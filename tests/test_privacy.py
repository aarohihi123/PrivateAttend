"""Unit tests for privacy.py (Cryptographic Protection & Key Management)."""

from pathlib import Path
import os
import stat
import numpy as np
import pytest

from privacy import CryptographicError, KeyManager, TemplateProtector


def test_key_manager_generation_and_permissions(tmp_path: Path):
    """Test secure key generation and 0600 POSIX permissions."""
    key_file = tmp_path / "test_key.key"
    km = KeyManager(key_path=key_file)

    assert key_file.is_file()
    assert len(km.key) > 0

    # Verify POSIX file permissions
    file_mode = stat.S_IMODE(os.stat(key_file).st_mode)
    assert file_mode & (stat.S_IRUSR | stat.S_IWUSR) != 0
    # Other users should not have read/write access
    assert file_mode & (stat.S_IRWXO | stat.S_IRWXG) == 0


def test_template_encryption_decryption_fidelity(template_protector: TemplateProtector, synthetic_embedding):
    """Test encryption and decryption roundtrip without numerical degradation."""
    orig = synthetic_embedding
    ciphertext = template_protector.encrypt_template(orig)

    assert isinstance(ciphertext, bytes)
    assert len(ciphertext) > len(orig.tobytes())  # Authenticated ciphertext expansion

    decrypted = template_protector.decrypt_template(ciphertext)
    assert decrypted.shape == (512,)
    assert decrypted.dtype == np.float32

    # Verify absolute numerical identity
    np.testing.assert_allclose(orig, decrypted, atol=1e-6)


def test_tampered_ciphertext_rejection(template_protector: TemplateProtector, synthetic_embedding):
    """Test rejection of corrupted or tampered ciphertext."""
    ciphertext = bytearray(template_protector.encrypt_template(synthetic_embedding))
    # Corrupt several bytes in payload
    ciphertext[20] ^= 0xFF
    ciphertext[21] ^= 0xAA

    with pytest.raises(CryptographicError):
        template_protector.decrypt_template(bytes(ciphertext))


def test_zero_memory(synthetic_embedding):
    """Test zero_memory wiping numpy buffer."""
    buf = synthetic_embedding.copy()
    assert not np.all(buf == 0.0)

    TemplateProtector.zero_memory(buf)
    assert np.all(buf == 0.0)
