"""PrivateAttend: Neural Network Architectures for Biometric Face Recognition.

Defines ArcFace-compatible deep feature extraction backbones (MobileFaceNet and
ResNet-based deep metric embeddings) in PyTorch, with ONNX export and runtime inference wrappers.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np


class ModelNotLoadedError(RuntimeError):
    """Raised when an inference operation is requested without loaded weights."""
    pass


class MobileFaceNet:
    """Lightweight MobileFaceNet backbone for on-device 512-d embedding extraction.

    Designed for 112x112 facial images, utilizing depthwise separable convolutions
    and linear bottleneck layers to achieve high speed and low memory on CPU.
    """

    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self._torch_model = None

    def build_torch_module(self):
        """Constructs the PyTorch nn.Module architecture."""
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            class ConvBN(nn.Module):
                def __init__(self, in_c, out_c, kernel_size, stride, padding):
                    super().__init__()
                    self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)
                    self.bn = nn.BatchNorm2d(out_c)
                    self.prelu = nn.PReLU(out_c)

                def forward(self, x):
                    return self.prelu(self.bn(self.conv(x)))

            class DepthwiseSeparable(nn.Module):
                def __init__(self, in_c, out_c, stride=1):
                    super().__init__()
                    self.dw = nn.Conv2d(in_c, in_c, 3, stride, 1, groups=in_c, bias=False)
                    self.bn1 = nn.BatchNorm2d(in_c)
                    self.prelu = nn.PReLU(in_c)
                    self.pw = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)
                    self.bn2 = nn.BatchNorm2d(out_c)

                def forward(self, x):
                    out = self.prelu(self.bn1(self.dw(x)))
                    return self.bn2(self.pw(out))

            class ArcFaceBackbone(nn.Module):
                def __init__(self, feat_dim=512):
                    super().__init__()
                    # Input: (B, 3, 112, 112)
                    self.conv1 = ConvBN(3, 64, 3, 2, 1)  # -> (B, 64, 56, 56)
                    self.dw1 = DepthwiseSeparable(64, 64, stride=1)
                    self.dw2 = DepthwiseSeparable(64, 128, stride=2)  # -> (B, 128, 28, 28)
                    self.dw3 = DepthwiseSeparable(128, 128, stride=1)
                    self.dw4 = DepthwiseSeparable(128, 256, stride=2)  # -> (B, 256, 14, 14)
                    self.dw5 = DepthwiseSeparable(256, 256, stride=1)
                    self.dw6 = DepthwiseSeparable(256, 512, stride=2)  # -> (B, 512, 7, 7)
                    self.conv_sep = nn.Conv2d(512, 512, 7, 1, 0, groups=512, bias=False)
                    self.bn_sep = nn.BatchNorm2d(512)
                    self.fc = nn.Linear(512, feat_dim, bias=False)
                    self.bn_fc = nn.BatchNorm1d(feat_dim)

                def forward(self, x):
                    x = self.conv1(x)
                    x = self.dw1(x)
                    x = self.dw2(x)
                    x = self.dw3(x)
                    x = self.dw4(x)
                    x = self.dw5(x)
                    x = self.dw6(x)
                    x = self.bn_sep(self.conv_sep(x))
                    x = x.flatten(1)
                    feat = self.bn_fc(self.fc(x))
                    # L2 Normalization enforcing ||e||_2 == 1.0 on hypersphere
                    return F.normalize(feat, p=2, dim=1)

            self._torch_model = ArcFaceBackbone(feat_dim=self.embedding_dim)
            self._torch_model.eval()
            return self._torch_model
        except ImportError as e:
            raise RuntimeError(f"PyTorch is required to build the PyTorch module: {e}")


class ONNXRecognitionInference:
    """High-performance ONNX Runtime inference engine for ArcFace models."""

    def __init__(self, model_path: Union[str, Path]):
        self.model_path = Path(model_path)
        self.session = None
        self.input_name = None
        self.output_name = None
        self._load_session()

    def _load_session(self) -> None:
        """Loads ONNX runtime session with CPU execution provider."""
        if not self.model_path.is_file():
            raise ModelNotLoadedError(
                f"ArcFace ONNX model weights not found at {self.model_path}. "
                "Inference requires trained model weights."
            )
        try:
            import onnxruntime as ort

            # Prefer CPU provider for deterministic on-device inference
            providers = ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(str(self.model_path), providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
        except Exception as e:
            raise ModelNotLoadedError(f"Failed to initialize ONNX Runtime session: {e}")

    def run(self, input_tensor: np.ndarray) -> np.ndarray:
        """Executes forward pass and returns L2-normalized 512-d feature vector.

        Args:
            input_tensor: Shape (B, 3, 112, 112) or (3, 112, 112), float32 [-1, 1].

        Returns:
            np.ndarray: Normalized feature vectors of shape (B, 512) or (512,).
        """
        if self.session is None:
            raise ModelNotLoadedError("ONNX session is not loaded.")

        orig_ndim = input_tensor.ndim
        if orig_ndim == 3:
            input_tensor = np.expand_dims(input_tensor, axis=0)

        # Ensure float32 format
        input_tensor = input_tensor.astype(np.float32)

        raw_out = self.session.run([self.output_name], {self.input_name: input_tensor})[0]

        # Enforce L2 normalization: e / ||e||_2
        norms = np.linalg.norm(raw_out, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        normalized = raw_out / norms

        if orig_ndim == 3:
            return normalized[0]
        return normalized
