"""PrivateAttend: Presentation Attack Detection (LivenessNet) Training Harness.

Provides training, validation, and ONNX export routines for the lightweight texture CNN,
evaluating ISO/IEC 30107-3 metrics (APCER, BPCER, ACER) at each epoch.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import config
from liveness import build_liveness_net, compute_pad_metrics


def train_liveness_pipeline(
    data_dir: Path,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    output_path: Optional[Path] = None,
    export_onnx: bool = True,
):
    """Executes PAD model training over bona fide and spoof face datasets."""
    out_file = output_path or (config.WEIGHTS_DIR / config.LIVENESS_WEIGHTS_FILE)

    if not data_dir.is_dir():
        print(f"[!] Dataset directory not found at: {data_dir}")
        print("    Liveness dataset is pending integration.")
        print("    Expected directory structure:")
        print(f"      {data_dir}/train/bona_fide/*.jpg")
        print(f"      {data_dir}/train/attacks/*.jpg (print, replay, video)")
        print(f"      {data_dir}/val/bona_fide/*.jpg")
        print(f"      {data_dir}/val/attacks/*.jpg")
        return

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, Dataset
        from torchvision import transforms
        from PIL import Image
    except ImportError as e:
        print(f"[!] PyTorch/Torchvision required for training: {e}")
        return

    class LivenessDataset(Dataset):
        def __init__(self, root_dir: Path, transform=None):
            self.samples = []
            self.transform = transform

            live_dir = root_dir / "bona_fide"
            attack_dir = root_dir / "attacks"

            if live_dir.is_dir():
                for p in live_dir.glob("*.*"):
                    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        self.samples.append((p, 1))  # 1 = live

            if attack_dir.is_dir():
                for p in attack_dir.glob("*.*"):
                    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        self.samples.append((p, 0))  # 0 = attack

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, label = self.samples[idx]
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label

    transform_train = transforms.Compose([
        transforms.Resize(config.FACE_TARGET_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(config.FACE_TARGET_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_ds = LivenessDataset(data_dir / "train", transform=transform_train)
    val_ds = LivenessDataset(data_dir / "val", transform=transform_val)

    if len(train_ds) == 0:
        print("[!] No training samples found. Aborting training.")
        return

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_liveness_net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_acer = 1.0

    print(f"[*] Starting PAD training on {device} ({len(train_ds)} train, {len(val_ds)} val)...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)

        train_loss = total_loss / len(train_ds)

        # Validation phase
        model.eval()
        all_labels = []
        all_scores = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                all_scores.extend(probs.tolist())
                all_labels.extend(labels.numpy().tolist())

        if all_labels:
            metrics = compute_pad_metrics(np.array(all_labels), np.array(all_scores))
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Loss: {train_loss:.4f} | "
                f"APCER: {metrics['apcer']:.2%} | "
                f"BPCER: {metrics['bpcer']:.2%} | "
                f"ACER: {metrics['acer']:.2%} | "
                f"AUC: {metrics['roc_auc']:.4f}"
            )

            if metrics["acer"] <= best_acer:
                best_acer = metrics["acer"]
                out_file.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), str(out_file))
                print(f"  [+] Saved best checkpoint (ACER: {best_acer:.2%}) to {out_file}")

    # Export to ONNX if requested
    if export_onnx and out_file.is_file():
        onnx_path = out_file.with_suffix(".onnx")
        dummy_input = torch.randn(1, 3, 112, 112).to(device)
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        )
        print(f"  [+] Exported ONNX model to: {onnx_path}")


def main():
    parser = argparse.ArgumentParser(description="PrivateAttend LivenessNet Training Script")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR / "liveness_dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=config.WEIGHTS_DIR / config.LIVENESS_WEIGHTS_FILE)
    args = parser.parse_args()

    train_liveness_pipeline(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
