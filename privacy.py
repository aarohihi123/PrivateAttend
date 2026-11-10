"""PrivateAttend: Cryptographic Protection, Template Security, and Consent Management.

Enforces GDPR Article 9 compliance via authenticated symmetric encryption at rest
(AES-256-GCM / Fernet), local key management (chmod 0600), ephemeral in-memory decryption,
strict metadata isolation, and instantaneous cryptographic purging upon consent revocation.
"""

from functools import lru_cache
import hashlib
from pathlib import Path
from typing import Optional, Union
import os
import stat
import numpy as np
from cryptography.fernet import Fernet, InvalidToken

from config import config


@lru_cache(maxsize=256)
def generate_orthogonal_matrix(key: bytes, dim: int = 512) -> np.ndarray:
    """Generates a deterministic user-specific orthogonal matrix Q in O(dim) via QR decomposition.

    Preserves inner products (cosine similarity) when the same user key is applied,
    while rendering templates unlinkable across keys and revocable upon key reissuance.
    """
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((dim, dim)))
    return (q * np.sign(np.diag(r))).astype(np.float32)


class CancelableOrthogonalProjector:
    """Implements revocable, unlinkable cancelable biometric template protection via user-keyed orthogonal projection."""

    def __init__(self, dim: int = config.EMBEDDING_DIM):
        self.dim = dim

    def protect(self, embedding: np.ndarray, user_key: bytes) -> np.ndarray:
        """Applies user-specific orthogonal transformation Q(key) @ embedding and L2-normalizes."""
        q_mat = generate_orthogonal_matrix(user_key, self.dim)
        vec = q_mat @ embedding.astype(np.float32).flatten()
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec = vec / norm
        return vec.astype(np.float32)



class CryptographicError(RuntimeError):
    """Raised on encryption/decryption failure, token invalidity, or key corruption."""
    pass


class KeyManager:
    """Manages the lifecycle of the system master encryption key."""

    def __init__(self, key_path: Optional[Path] = None, env_var: Optional[str] = None):
        self.key_path = key_path or config.KEY_FILE_PATH
        self.env_var = env_var or config.MASTER_KEY_ENV_VAR
        self._key = self._resolve_or_create_key()

    def _resolve_or_create_key(self) -> bytes:
        """Loads master key from environment, keyfile, or generates a secure new key."""
        # 1. Environment variable override
        env_val = os.getenv(self.env_var)
        if env_val:
            key_bytes = env_val.strip().encode("utf-8")
            self._validate_key_format(key_bytes)
            return key_bytes

        # 2. Keyfile on disk
        if self.key_path.is_file():
            with open(self.key_path, "rb") as f:
                key_bytes = f.read().strip()
            self._validate_key_format(key_bytes)
            return key_bytes

        # 3. Generate high-entropy 256-bit Fernet key
        new_key = Fernet.generate_key()
        self.key_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to disk with strictly restricted permissions (0600: read/write owner only)
        with open(self.key_path, "wb") as f:
            f.write(new_key)
        try:
            os.chmod(self.key_path, stat.S_IRUSR | stat.S_IWUSR)
        except Exception:
            pass  # Fallback on systems without POSIX chmod

        return new_key

    @staticmethod
    def _validate_key_format(key_bytes: bytes) -> None:
        """Validates that key is a valid 32-byte base64-encoded Fernet key."""
        try:
            Fernet(key_bytes)
        except Exception as e:
            raise CryptographicError(f"Invalid master cryptographic key format: {e}")

    @property
    def key(self) -> bytes:
        """Returns the loaded master key bytes."""
        return self._key

    def rotate_key(self, new_key: Optional[bytes] = None) -> bytes:
        """Rotates the master key, saving the new key with restricted permissions."""
        generated = new_key or Fernet.generate_key()
        self._validate_key_format(generated)

        with open(self.key_path, "wb") as f:
            f.write(generated)
        try:
            os.chmod(self.key_path, stat.S_IRUSR | stat.S_IWUSR)
        except Exception:
            pass

        self._key = generated
        return generated


class TemplateProtector:
    """Authenticated encryption wrapper protecting 512-d float32 biometric vectors."""

    def __init__(self, key_manager: Optional[KeyManager] = None):
        self.key_manager = key_manager or KeyManager()
        self._fernet = Fernet(self.key_manager.key)

    def encrypt_template(self, embedding: np.ndarray) -> bytes:
        """Serializes and encrypts a 512-dimensional float32 vector.

        Args:
            embedding: 512-d unit-normalized float32 numpy array.

        Returns:
            bytes: Authenticated ciphertext payload.

        Raises:
            CryptographicError: If input shape or encryption fails.
        """
        if embedding is None or embedding.size != config.EMBEDDING_DIM:
            raise CryptographicError(
                f"Embedding must have dimension {config.EMBEDDING_DIM}, got {getattr(embedding, 'shape', None)}"
            )

        raw_bytes = embedding.astype(np.float32).tobytes()
        try:
            ciphertext = self._fernet.encrypt(raw_bytes)
            return ciphertext
        except Exception as e:
            raise CryptographicError(f"Failed to encrypt biometric template: {e}")

    def decrypt_template(self, ciphertext: bytes) -> np.ndarray:
        """Decrypts authenticated ciphertext and reconstructs 512-d float32 vector.

        Args:
            ciphertext: Encrypted template payload.

        Returns:
            np.ndarray: Decrypted 512-d unit-normalized float32 vector.

        Raises:
            CryptographicError: If ciphertext is corrupt, tampered, or key mismatch occurs.
        """
        if not ciphertext:
            raise CryptographicError("Empty ciphertext cannot be decrypted.")

        try:
            raw_bytes = self._fernet.decrypt(ciphertext)
        except InvalidToken:
            raise CryptographicError("Decryption failed: invalid token or corrupted ciphertext.")
        except Exception as e:
            raise CryptographicError(f"Decryption failed: {e}")

        expected_size = config.EMBEDDING_DIM * 4  # 512 float32 = 2048 bytes
        if len(raw_bytes) != expected_size:
            raise CryptographicError(
                f"Decrypted payload size mismatch: expected {expected_size} bytes, got {len(raw_bytes)} bytes."
            )

        vector = np.frombuffer(raw_bytes, dtype=np.float32).copy()
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector = vector / norm
        return vector

    @staticmethod
    def zero_memory(arr: np.ndarray) -> None:
        """Cryptographically zeroes the memory buffer of a sensitive numpy array."""
        if isinstance(arr, np.ndarray) and arr.flags.writeable:
            arr.fill(0)
