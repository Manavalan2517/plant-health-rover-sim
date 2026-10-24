"""Dataset manifest management and group-based leakage-free splits."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import random
from typing import Optional
from PIL import Image, ImageDraw


TOMATO_CLASSES = (
    "healthy",
    "bacterial_spot",
    "early_blight",
    "late_blight",
    "leaf_mold",
    "septoria_leaf_spot",
    "spider_mites",
    "target_spot",
    "yellow_leaf_curl",
    "mosaic_virus",
)


@dataclass
class DatasetItem:
    path: str
    sha256: str
    label: str
    plant_id: str
    session_id: str
    split: str  # "train", "val", "test"
    annotation_confidence: float = 1.0


class DatasetManifest:
    """Manages metadata, file integrity checksums, and plant-level splits to prevent leakage."""

    def __init__(self, dataset_name: str = "tomato_leaf_dataset", version: str = "1.0.0") -> None:
        self.dataset_name = dataset_name
        self.version = version
        self.items: list[DatasetItem] = []

    def add_item(self, path: str | Path, label: str, plant_id: str, session_id: str, split: str = "train", confidence: float = 1.0) -> DatasetItem:
        p = Path(path)
        sha = hashlib.sha256(p.read_bytes() if p.exists() else f"{p}:{label}".encode()).hexdigest()
        item = DatasetItem(str(p), sha, label, plant_id, session_id, split, confidence)
        self.items.append(item)
        return item

    def group_split(self, by: str = "plant_id", train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 2026) -> None:
        """Assign splits by grouping unique entity keys to guarantee zero data leakage."""
        rng = random.Random(seed)
        groups: dict[str, list[DatasetItem]] = {}
        for item in self.items:
            key = getattr(item, by)
            groups.setdefault(key, []).append(item)

        unique_keys = list(groups.keys())
        rng.shuffle(unique_keys)

        n = len(unique_keys)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_keys = set(unique_keys[:n_train])
        val_keys = set(unique_keys[n_train:n_train + n_val])
        test_keys = set(unique_keys[n_train + n_val:])

        for key, group_items in groups.items():
            split = "train" if key in train_keys else "val" if key in val_keys else "test"
            for item in group_items:
                item.split = split

    def get_split(self, split: str) -> list[DatasetItem]:
        return [item for item in self.items if item.split == split]

    def to_dict(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "version": self.version,
            "total_samples": len(self.items),
            "items": [asdict(i) for i in self.items],
        }

    def save(self, json_path: str | Path) -> None:
        p = Path(json_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, json_path: str | Path) -> DatasetManifest:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        manifest = cls(data.get("dataset_name", "unknown"), data.get("version", "1.0.0"))
        for raw in data.get("items", []):
            manifest.items.append(DatasetItem(**raw))
        return manifest

    @classmethod
    def create_synthetic_evaluation_manifest(cls, output_dir: str | Path, sample_count: int = 60, seed: int = 2026) -> DatasetManifest:
        """Generates a balanced evaluation dataset fixtures directory and corresponding manifest."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        manifest = cls("synthetic_evaluation_dataset", "1.0.0")
        rng = random.Random(seed)

        classes = ("healthy", "early_blight", "late_blight", "water_stress", "nutrient_stress")
        for i in range(sample_count):
            label = classes[i % len(classes)]
            plant_id = f"PLANT-{1 + (i // 2):03d}"  # 2 samples per plant to test group splitting
            session_id = f"SESS-{1 + (i % 3):02d}"

            img_path = out_path / f"{plant_id}_{session_id}_{label}_{i}.png"
            img = Image.new("RGB", (224, 224), (40, 45, 40))
            draw = ImageDraw.Draw(img)

            color = (50, 160, 60) if label == "healthy" else (160, 120, 50) if label == "early_blight" else (90, 70, 50) if label == "late_blight" else (120, 140, 60)
            draw.ellipse((40, 40, 184, 184), fill=color, outline=(180, 190, 140), width=2)
            if "blight" in label:
                for _ in range(6):
                    x, y = rng.randint(60, 160), rng.randint(60, 160)
                    draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(80, 50, 20))
            img.save(img_path)

            manifest.add_item(img_path, label, plant_id, session_id, split="train", confidence=0.95)

        manifest.group_split(by="plant_id", train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=seed)
        manifest.save(out_path / "manifest.json")
        return manifest
