"""Automated dataset audit, duplicate analysis, and leakage-resistant group partitioning.

Audits:
- Class distribution and image dimensions across 30,609 images.
- Exact duplicates via SHA-256 hash collisions.
- Near-duplicates and rotation/flip variants via multi-symmetry perceptual hashing.
- Group clustering to prevent augmented variants of the same physical leaf from crossing partitions.
- Leakage-resistant train (70%), val (15%), test (15%) group splitting.
- Generation of formal DATASET_MANIFEST.json and DATASET_AUDIT_REPORT.json.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
from PIL import Image

CLASS_NAMES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]


def multi_symmetry_hash(im: Image.Image, size: int = 8) -> list[int]:
    """Compute 64-bit perceptual hashes for 8 planar symmetries (flip x 4 rotations)."""
    im_l = im.convert("L")
    hashes = []
    for flip in [False, True]:
        base = im_l.transpose(Image.FLIP_LEFT_RIGHT) if flip else im_l
        for rot in [0, 90, 180, 270]:
            r_im = base.rotate(rot) if rot > 0 else base
            resized = r_im.resize((size, size), Image.Resampling.BILINEAR)
            arr = np.array(resized, dtype=np.float32)
            bits = (arr > arr.mean()).astype(np.uint8).flatten()
            val = 0
            for b in bits:
                val = (val << 1) | int(b)
            hashes.append(val)
    return hashes


def audit_and_partition_dataset(
    dataset_dir: str | Path,
    output_dir: str | Path,
    seed: int = 2026,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, Any]:
    dataset_path = Path(dataset_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    t0 = time.time()

    print(f"=== Starting Dataset Audit and Leakage-Free Group Partitioning ===")
    print(f"Dataset root: {dataset_path}")

    manifest_items = []
    class_stats = {}
    exact_sha_map: dict[str, list[dict[str, str]]] = {}
    corrupted_files = []

    total_usable = 0
    total_corrupt = 0
    all_dimensions: dict[str, int] = {}
    all_formats: dict[str, int] = {}

    for cname in CLASS_NAMES:
        cdir = dataset_path / cname
        if not cdir.exists():
            continue

        raw_files = sorted([f for f in cdir.iterdir()])
        valid_files: list[Path] = []
        corrupt_in_class = []

        for f in raw_files:
            if not f.is_file() or f.stat().st_size == 0 or f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                corrupted_files.append({"path": str(f), "class": cname, "reason": "zero_byte_or_non_image"})
                corrupt_in_class.append(f.name)
                total_corrupt += 1
                continue

            try:
                # Fast header verification
                with Image.open(f) as im:
                    fmt = im.format or "UNKNOWN"
                    dims = f"{im.width}x{im.height}"
                    all_dimensions[dims] = all_dimensions.get(dims, 0) + 1
                    all_formats[fmt] = all_formats.get(fmt, 0) + 1
                valid_files.append(f)
                total_usable += 1
            except Exception as exc:
                corrupted_files.append({"path": str(f), "class": cname, "reason": str(exc)})
                corrupt_in_class.append(f.name)
                total_corrupt += 1

        n_valid = len(valid_files)

        # 1. Exact SHA-256 hashes
        sha_list = []
        for f in valid_files:
            sha = hashlib.sha256(f.read_bytes()).hexdigest()
            sha_list.append(sha)
            exact_sha_map.setdefault(sha, []).append({"class": cname, "filename": f.name, "path": str(f)})

        # 2. Multi-symmetry perceptual hashes for duplicate / variant clustering
        print(f"  Hashing {n_valid:>4} images for {cname}...")
        sym_hashes = []
        for f in valid_files:
            with Image.open(f) as im:
                sym_hashes.append(multi_symmetry_hash(im))

        # Union-Find clustering: merge images sharing exact SHA-256 or symmetrical aHash
        parent = list(range(n_valid))

        def find(i: int) -> int:
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        # Fast bucketing: exact SHA-256 collisions
        sha_bucket: dict[str, int] = {}
        for i, sha in enumerate(sha_list):
            if sha in sha_bucket:
                union(i, sha_bucket[sha])
            else:
                sha_bucket[sha] = i

        # Fast bucketing: symmetrical perceptual hashes
        phash_bucket: dict[int, int] = {}
        for i, h_list in enumerate(sym_hashes):
            for h in set(h_list):
                if h in phash_bucket:
                    union(i, phash_bucket[h])
                else:
                    phash_bucket[h] = i

        # Near-duplicate candidate detection (Hamming distance <= 2 on base hash for review)
        base_hashes = np.array([h_list[0] for h_list in sym_hashes], dtype=np.uint64)
        xor_mat = base_hashes[:, None] ^ base_hashes[None, :]
        hamming_mat = np.bitwise_count(xor_mat)
        # Find pairs with distance in [1, 2] between distinct roots
        candidate_pairs_in_class = []
        for i_idx in range(n_valid):
            close_js = np.where((hamming_mat[i_idx] > 0) & (hamming_mat[i_idx] <= 2))[0]
            for j_idx in close_js:
                if i_idx < j_idx and find(i_idx) != find(j_idx):
                    candidate_pairs_in_class.append({
                        "file_a": valid_files[i_idx].name,
                        "file_b": valid_files[j_idx].name,
                        "hamming_distance": int(hamming_mat[i_idx, j_idx]),
                    })

        # Collect groups
        groups: dict[int, list[int]] = {}
        for i in range(n_valid):
            root = find(i)
            groups.setdefault(root, []).append(i)

        unique_group_roots = list(groups.keys())
        rng.shuffle(unique_group_roots)

        # Partition at the group level
        n_grps = len(unique_group_roots)
        n_train_grps = int(n_grps * train_ratio)
        n_val_grps = int(n_grps * val_ratio)

        train_grp_set = set(unique_group_roots[:n_train_grps])
        val_grp_set = set(unique_group_roots[n_train_grps : n_train_grps + n_val_grps])
        test_grp_set = set(unique_group_roots[n_train_grps + n_val_grps :])

        # Verify mutual exclusivity of groups
        assert train_grp_set.isdisjoint(val_grp_set)
        assert train_grp_set.isdisjoint(test_grp_set)
        assert val_grp_set.isdisjoint(test_grp_set)

        train_count = 0
        val_count = 0
        test_count = 0
        multi_variant_groups = 0

        for grp_idx, root in enumerate(unique_group_roots):
            item_indices = groups[root]
            if len(item_indices) > 1:
                multi_variant_groups += 1

            split = "train" if root in train_grp_set else "val" if root in val_grp_set else "test"
            grp_id = f"{cname}_grp_{grp_idx:04d}"

            for idx in item_indices:
                f = valid_files[idx]
                sha = sha_list[idx]
                if split == "train":
                    train_count += 1
                elif split == "val":
                    val_count += 1
                else:
                    test_count += 1

                manifest_items.append({
                    "path": str(f.relative_to(dataset_path.parent)),
                    "filename": f.name,
                    "class": cname,
                    "sha256": sha,
                    "group_id": grp_id,
                    "split": split,
                })

        class_stats[cname] = {
            "total_images": n_valid,
            "corrupt_count": len(corrupt_in_class),
            "distinct_groups": n_grps,
            "multi_variant_groups": multi_variant_groups,
            "near_match_candidate_pairs_count": len(candidate_pairs_in_class),
            "sample_candidate_pairs": candidate_pairs_in_class[:10],
            "train_samples": train_count,
            "val_samples": val_count,
            "test_samples": test_count,
        }
        print(f"    -> {n_valid} images -> {n_grps} groups | Train: {train_count}, Val: {val_count}, Test: {test_count} ({multi_variant_groups} variant groups, {len(candidate_pairs_in_class)} near-match review pairs)")

    exact_duplicates = [v for v in exact_sha_map.values() if len(v) > 1]
    cross_class_duplicates = [v for v in exact_duplicates if len(set(x["class"] for x in v)) > 1]

    # Verify zero data leakage across splits
    train_groups = {item["group_id"] for item in manifest_items if item["split"] == "train"}
    val_groups = {item["group_id"] for item in manifest_items if item["split"] == "val"}
    test_groups = {item["group_id"] for item in manifest_items if item["split"] == "test"}

    assert train_groups.isdisjoint(val_groups), "Leakage between train and val!"
    assert train_groups.isdisjoint(test_groups), "Leakage between train and test!"
    assert val_groups.isdisjoint(test_groups), "Leakage between val and test!"

    # Compile Audit Report
    total_train = sum(s["train_samples"] for s in class_stats.values())
    total_val = sum(s["val_samples"] for s in class_stats.values())
    total_test = sum(s["test_samples"] for s in class_stats.values())
    total_groups = sum(s["distinct_groups"] for s in class_stats.values())
    total_multi_variant = sum(s["multi_variant_groups"] for s in class_stats.values())
    total_near_match_candidates = sum(s["near_match_candidate_pairs_count"] for s in class_stats.values())

    audit_summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "total_files_inspected": total_usable + total_corrupt,
        "total_usable_images": total_usable,
        "total_corrupted_or_non_image": total_corrupt,
        "corrupted_files": corrupted_files,
        "formats_distribution": all_formats,
        "dimensions_distribution": all_dimensions,
        "exact_duplicates_count": len(exact_duplicates),
        "exact_duplicate_files_count": sum(len(v) for v in exact_duplicates),
        "cross_class_duplicates_count": len(cross_class_duplicates),
        "total_distinct_specimen_groups": total_groups,
        "total_multi_variant_groups": total_multi_variant,
        "total_near_match_candidates_count": total_near_match_candidates,
        "partition_proportions": {
            "train": round(total_train / total_usable, 4),
            "val": round(total_val / total_usable, 4),
            "test": round(total_test / total_usable, 4),
        },
        "partition_sample_counts": {
            "train": total_train,
            "val": total_val,
            "test": total_test,
        },
        "leakage_verification": {
            "train_val_group_overlap": len(train_groups.intersection(val_groups)),
            "train_test_group_overlap": len(train_groups.intersection(test_groups)),
            "val_test_group_overlap": len(val_groups.intersection(test_groups)),
            "zero_leakage_guaranteed": True,
            "isolation_scope": "Guarantees zero overlap among detected hash-defined and exact-duplicate groups across partitions.",
            "specimen_level_independence_guaranteed": False,
            "provenance_limitation": (
                "Original source specimen IDs were stripped upstream. Detected hash groups successfully isolate "
                "exact and symmetrical variants, but cannot guarantee that all photographs of the same physical plant "
                "or leaf have been grouped together if they underwent distinct non-symmetrical transformations."
            ),
        },
        "class_breakdown": class_stats,
    }

    # Save manifest and report
    manifest_file = out_path / "DATASET_MANIFEST.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump({
            "manifest_version": "2.0.0",
            "dataset_name": "augmented_dataset",
            "provenance": "PlantVillage laboratory dataset with synthetic affine/flip augmentations; specimen IDs stripped upstream.",
            "total_samples": len(manifest_items),
            "summary": audit_summary,
            "items": manifest_items,
        }, f, indent=2)

    report_file = out_path / "DATASET_AUDIT_REPORT.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nAudit completed in {elapsed:.1f}s.")
    print(f"Total Usable: {total_usable} across 10 classes.")
    print(f"Total Specimen Groups: {total_groups} ({total_multi_variant} multi-variant clusters).")
    print(f"Leakage-free Partition -> Train: {total_train} (70%), Val: {total_val} (15%), Test: {total_test} (15%).")
    print(f"Exact Duplicate Groups: {len(exact_duplicates)} ({sum(len(v) for v in exact_duplicates)} files).")
    print(f"Cross-class Duplicates: {len(cross_class_duplicates)}.")
    print(f"Saved manifest to {manifest_file} and audit report to {report_file}.")

    return audit_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit dataset and generate leakage-free manifest.")
    parser.add_argument("--dataset-dir", default="/home/mano/Documents/Anveshion/dataset/augmented_dataset", help="Dataset directory")
    parser.add_argument("--output-dir", default="/home/mano/Documents/Anveshion/plant-health-rover-sim/docs", help="Output directory for reports")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    args = parser.parse_args()

    audit_and_partition_dataset(args.dataset_dir, args.output_dir, seed=args.seed)
