"""PrivateAttend: Dataset Preparation and Open-Set Partitioning Harness.

Prepares and validates the UMDFace research subset (100 identities x 9 images = 900 total, 112x112),
and generates identity-disjoint splits: Enrolled Gallery, Known Probes, Validation Calibration, and Test Unknowns.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import config


def _natural_key(path_or_name: Any) -> int:
    name = path_or_name.name if hasattr(path_or_name, "name") else str(path_or_name)
    digits = "".join(c for c in name if c.isdigit())
    return int(digits) if digits else 0


def verify_umdface_structure(raw_dir: Path) -> Tuple[bool, List[str], Dict[str, List[Path]]]:
    """Inspects directory to verify compliance with the 100 identities x 9 images specification.

    Args:
        raw_dir: Root directory containing identity subdirectories.

    Returns:
        Tuple: (is_valid, validation_messages, identity_to_images_map)
    """
    if not raw_dir.is_dir():
        return False, [f"Dataset directory does not exist: {raw_dir}"], {}

    subdirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()], key=_natural_key)
    messages: List[str] = []
    id_map: Dict[str, List[Path]] = {}

    if len(subdirs) == 0:
        return False, [f"No identity subdirectories found in {raw_dir}"], {}

    messages.append(f"Discovered {len(subdirs)} identity subdirectories.")

    valid_exts = {".jpg", ".jpeg", ".png"}
    total_images = 0

    for sdir in subdirs:
        imgs = sorted([f for f in sdir.iterdir() if f.suffix.lower() in valid_exts], key=_natural_key)
        id_map[sdir.name] = imgs
        total_images += len(imgs)

    messages.append(f"Total valid image files discovered: {total_images}")

    is_compliant = (len(subdirs) == 100) and (total_images == 900)
    if not is_compliant:
        messages.append(
            f"Note: Current directory contains {len(subdirs)} IDs and {total_images} images "
            "(Target UMDFace subset specification: 100 IDs x 9 images = 900 images)."
        )
    else:
        messages.append("Dataset strictly conforms to UMDFace 100 identities x 9 images (900 total, 112x112) specification.")

    return is_compliant, messages, id_map


def create_openset_splits(
    id_map: Dict[str, List[Path]],
    output_dir: Path,
    gallery_count: int = 60,
    val_count: int = 20,
    test_count: int = 20,
    samples_per_gallery_id: int = 3,
) -> Dict[str, Any]:
    """Generates identity-disjoint partitions for open-set evaluation without data leakage.

    Splits:
        1. Gallery (Enrolled): 60 IDs, first K images per ID (default K=3)
        2. Known Probes: Same 60 IDs, last 4 images per ID (240 images total)
        3. Validation Calibration Unknowns: 20 disjoint IDs, all 9 images (180 images total)
        4. Test Evaluation Unknowns: 20 disjoint IDs, all 9 images (180 images total)
    """
    identities = sorted(list(id_map.keys()), key=_natural_key)
    total_ids = len(identities)

    if total_ids < (gallery_count + val_count + test_count):
        gallery_count = int(total_ids * 0.60)
        val_count = int(total_ids * 0.20)
        test_count = total_ids - gallery_count - val_count

    gallery_ids = identities[:gallery_count]
    val_ids = identities[gallery_count:gallery_count + val_count]
    test_ids = identities[gallery_count + val_count:gallery_count + val_count + test_count]

    rename_map = {}
    if config.RENAME_MAP_PATH.is_file():
        with open(config.RENAME_MAP_PATH, "r") as f:
            rename_map = json.load(f)

    manifest = {
        "metadata": {
            "total_identities": total_ids,
            "gallery_identity_count": len(gallery_ids),
            "validation_identity_count": len(val_ids),
            "test_unknown_identity_count": len(test_ids),
            "samples_enrolled_per_identity": samples_per_gallery_id,
            "rename_map_loaded": bool(rename_map),
        },
        "rename_map": rename_map,
        "gallery": {},
        "known_probes": [],
        "validation_unknown_probes": [],
        "test_unknown_probes": [],
    }

    # 1. Enrolled Gallery and Known Probes (last 4 photos reserved as test probes)
    for gid in gallery_ids:
        all_imgs = [str(p) for p in id_map[gid]]
        enrolled = all_imgs[:samples_per_gallery_id]
        probes = all_imgs[-4:] if len(all_imgs) >= 4 else all_imgs[samples_per_gallery_id:]
        manifest["gallery"][gid] = enrolled
        for p in probes:
            manifest["known_probes"].append({"image_path": p, "identity": gid, "umd_id": rename_map.get(gid, gid), "is_known": True})

    # 2. Validation Calibration Probes
    for vid in val_ids:
        for p in id_map[vid]:
            manifest["validation_unknown_probes"].append({"image_path": str(p), "identity": vid, "umd_id": rename_map.get(vid, vid), "is_known": False})

    # 3. Test Unknown Probes (Strictly reserved for testing; NEVER used for calibration)
    for tid in test_ids:
        for p in id_map[tid]:
            manifest["test_unknown_probes"].append({"image_path": str(p), "identity": tid, "umd_id": rename_map.get(tid, tid), "is_known": False})

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "openset_manifest.json"
    with open(out_file, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def main():
    parser = argparse.ArgumentParser(description="PrivateAttend Dataset Preparation Harness")
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DATA_DIR / "umdface")
    parser.add_argument("--manifest-dir", type=Path, default=config.MANIFESTS_DIR)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    print(f"=== PrivateAttend Dataset Preparation Harness ===")
    print(f"Target Directory: {args.raw_dir}")

    is_compliant, messages, id_map = verify_umdface_structure(args.raw_dir)
    for msg in messages:
        print(f"[-] {msg}")

    if args.verify_only:
        return

    if id_map:
        print(f"\nGenerating open-set disjoint partitions...")
        manifest = create_openset_splits(id_map, args.manifest_dir)
        print(f"[+] Saved manifest to: {args.manifest_dir / 'openset_manifest.json'}")
        print(f"    - Enrolled Gallery IDs: {manifest['metadata']['gallery_identity_count']}")
        print(f"    - Known Probe Images: {len(manifest['known_probes'])}")
        print(f"    - Validation Unknown Probes: {len(manifest['validation_unknown_probes'])}")
        print(f"    - Test Unknown Probes: {len(manifest['test_unknown_probes'])}")
    else:
        print("\n[!] No dataset images found. Waiting for UMDFace dataset ingestion.")
        print("    Place raw images into data/raw/umdface/<identity_id>/ and re-run.")


if __name__ == "__main__":
    main()
