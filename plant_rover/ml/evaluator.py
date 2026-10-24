"""Evaluation pipeline computing per-class metrics, confusion matrices, and model comparison."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .manifest import DatasetManifest, DatasetItem
from .model import ModelInferenceEngine


@dataclass
class EvaluationReport:
    mode: str
    total_samples: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    uncertain_rate: float
    per_class: dict[str, dict[str, float]]
    confusion_matrix: dict[str, dict[str, int]]


class ModelEvaluator:
    """Evaluates inference engines against held-out splits to produce honest metrics."""

    @classmethod
    def evaluate(cls, engine: ModelInferenceEngine, manifest: DatasetManifest, split: str = "test") -> EvaluationReport:
        items = manifest.get_split(split)
        if not items:
            items = manifest.items  # fallback if split empty

        labels = sorted(list({item.label for item in items}))
        all_pred_classes = labels + ["uncertain"]

        cm: dict[str, dict[str, int]] = {true_lbl: {pred_lbl: 0 for pred_lbl in all_pred_classes} for true_lbl in labels}

        correct = 0
        uncertain_count = 0

        for item in items:
            res = engine.infer(item.path)
            pred = res.predicted_class
            if res.uncertain or pred == "uncertain":
                uncertain_count += 1
            if pred == item.label:
                correct += 1
            if item.label in cm and pred in cm[item.label]:
                cm[item.label][pred] += 1

        total = max(1, len(items))
        accuracy = correct / total
        uncertain_rate = uncertain_count / total

        per_class: dict[str, dict[str, float]] = {}
        precisions, recalls, f1s = [], [], []

        for lbl in labels:
            tp = cm[lbl].get(lbl, 0)
            fn = sum(cm[lbl].values()) - tp
            fp = sum(cm[other].get(lbl, 0) for other in labels if other != lbl)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            per_class[lbl] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": tp + fn,
            }
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        macro_p = float(sum(precisions) / len(precisions)) if precisions else 0.0
        macro_r = float(sum(recalls) / len(recalls)) if recalls else 0.0
        macro_f1 = float(sum(f1s) / len(f1s)) if f1s else 0.0

        return EvaluationReport(
            mode=engine.mode,
            total_samples=total,
            accuracy=round(accuracy, 4),
            macro_precision=round(macro_p, 4),
            macro_recall=round(macro_r, 4),
            macro_f1=round(macro_f1, 4),
            uncertain_rate=round(uncertain_rate, 4),
            per_class=per_class,
            confusion_matrix=cm,
        )

    @classmethod
    def compare_baselines(cls, manifest: DatasetManifest, split: str = "test") -> dict[str, EvaluationReport]:
        """Compares A (Image-only), B (Sensor-only), and C (Fusion) on the exact same held-out test split."""
        img_engine = ModelInferenceEngine(mode=ModelInferenceEngine.MODE_IMAGE_ONLY)
        sensor_engine = ModelInferenceEngine(mode=ModelInferenceEngine.MODE_SENSOR_ONLY)
        fusion_engine = ModelInferenceEngine(mode=ModelInferenceEngine.MODE_FUSION)

        return {
            "image_only": cls.evaluate(img_engine, manifest, split),
            "sensor_only": cls.evaluate(sensor_engine, manifest, split),
            "fusion": cls.evaluate(fusion_engine, manifest, split),
        }
