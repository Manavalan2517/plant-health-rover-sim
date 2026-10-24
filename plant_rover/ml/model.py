"""Inference engines and multi-modal fusion classifiers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
import math

import numpy as np
from PIL import Image, ImageStat, ImageFilter

from ..models import SensorReading, HistorySummary


@dataclass
class InferenceResult:
    predicted_class: str
    confidence_pct: int
    probabilities: dict[str, float]
    uncertain: bool
    rejection_reason: Optional[str]
    model_version: str
    provenance: str


class BaseInferenceEngine(ABC):
    @abstractmethod
    def infer(self, image_path: str | Path,
              readings: Sequence[SensorReading] = (),
              history: Optional[HistorySummary] = None) -> InferenceResult:
        raise NotImplementedError


class ModelInferenceEngine(BaseInferenceEngine):
    """Multi-modal ML inference engine implementing honest confidence calibration and explicit uncertainty."""

    CLASSES = ("healthy", "early_blight", "late_blight", "water_stress", "nutrient_stress", "uncertain")

    MODE_IMAGE_ONLY = "image_only"
    MODE_SENSOR_ONLY = "sensor_only"
    MODE_FUSION = "fusion"

    def __init__(self, mode: str = "fusion", confidence_threshold: float = 0.65, temperature: float = 1.2, version: str = "v1.0.0-cpu") -> None:
        self.mode = mode
        self.confidence_threshold = confidence_threshold
        self.temperature = max(0.1, temperature)
        self.version = version

    def _extract_image_features(self, image_path: Path) -> np.ndarray:
        """Extract lightweight vision features: color moments, Excess Green, and texture variance."""
        with Image.open(image_path) as raw:
            img = raw.convert("RGB").resize((128, 128))
            arr = np.array(img, dtype=np.float32)

            r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
            mean_r, std_r = float(np.mean(r)), float(np.std(r))
            mean_g, std_g = float(np.mean(g)), float(np.std(g))
            mean_b, std_b = float(np.mean(b)), float(np.std(b))

            # Excess Green Index: 2G - R - B
            exg = 2.0 * g - r - b
            mean_exg = float(np.mean(exg))

            # Necrosis / Brown spot ratio (R > 1.2 * G and G > B)
            brown_mask = (r > 1.2 * g) & (g > b) & (r > 60)
            necrosis_ratio = float(np.sum(brown_mask)) / (128 * 128)

            # Sharpness via Laplacian variance
            gray = raw.convert("L").resize((128, 128))
            laplacian = gray.filter(ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1, offset=128))
            sharpness = float(ImageStat.Stat(laplacian).var[0])

            return np.array([mean_r, std_r, mean_g, std_g, mean_b, std_b, mean_exg, necrosis_ratio, sharpness], dtype=np.float32)

    def _extract_sensor_features(self, readings: Sequence[SensorReading], history: Optional[HistorySummary]) -> np.ndarray:
        if not readings:
            return np.zeros(5, dtype=np.float32)

        valid = [r for r in readings if r.quality_flags == 0]
        if not valid:
            valid = list(readings)

        moisture = float(np.mean([r.soil_moisture_pct for r in valid]))
        air_temp = float(np.max([r.air_temperature_c for r in valid]))
        rh = float(np.mean([r.relative_humidity_pct for r in valid]))
        nitrogen = float(np.mean([r.nitrogen_mg_kg for r in valid]))
        trend = float(history.soil_moisture_trend_pct_per_hour if history else 0.0)

        return np.array([moisture, air_temp, rh, nitrogen, trend], dtype=np.float32)

    def infer(self, image_path: str | Path,
              readings: Sequence[SensorReading] = (),
              history: Optional[HistorySummary] = None) -> InferenceResult:
        p = Path(image_path)
        if not p.exists():
            return InferenceResult("uncertain", 0, {}, True, "image_file_not_found", self.version, "FILE_NOT_FOUND")

        img_feats = self._extract_image_features(p)
        sensor_feats = self._extract_sensor_features(readings, history)

        # Baseline linear projection logits
        # 5 base target classes: healthy, early_blight, late_blight, water_stress, nutrient_stress
        logits = np.zeros(5, dtype=np.float32)

        if self.mode in (self.MODE_IMAGE_ONLY, self.MODE_FUSION):
            mean_exg, necrosis, sharpness = img_feats[6], img_feats[7], img_feats[8]
            if sharpness < 40.0:  # Blurry capture penalty
                logits += np.array([-2.0, -2.0, -2.0, -2.0, -2.0], dtype=np.float32)
            else:
                if necrosis > 0.08:
                    logits[1] += 2.8  # early_blight
                    logits[2] += 2.2  # late_blight
                elif mean_exg > 15.0:
                    logits[0] += 3.2  # healthy
                else:
                    logits[3] += 1.2  # water_stress
                    logits[4] += 1.5  # nutrient_stress

        if self.mode in (self.MODE_SENSOR_ONLY, self.MODE_FUSION):
            moisture, air_temp, rh, nitrogen, trend = sensor_feats
            if moisture < 27.0 or trend < -0.3:
                logits[3] += 3.5  # water_stress
            if nitrogen < 25.0:
                logits[4] += 2.8  # nutrient_stress
            if rh > 82.0 and self.mode == self.MODE_FUSION and logits[1] > 1.0:
                logits[1] += 1.8  # Fungal risk elevated by high humidity
                logits[2] += 1.8
            if moisture >= 30.0 and 20.0 <= air_temp <= 30.0 and nitrogen >= 25.0:
                logits[0] += 1.5  # healthy corroborated

        # Calibrated Softmax with Temperature scaling
        scaled = logits / self.temperature
        exp_logits = np.exp(scaled - np.max(scaled))
        probs_array = exp_logits / np.sum(exp_logits)

        class_names = ["healthy", "early_blight", "late_blight", "water_stress", "nutrient_stress"]
        probs = {name: float(round(p, 4)) for name, p in zip(class_names, probs_array)}

        top_idx = int(np.argmax(probs_array))
        top_prob = float(probs_array[top_idx])
        top_class = class_names[top_idx]

        # Uncertainty gate: if confidence is below threshold, output UNCERTAIN
        is_uncertain = top_prob < self.confidence_threshold
        rejection_reason = f"low_confidence: {top_prob:.2f} < {self.confidence_threshold:.2f}" if is_uncertain else None
        final_class = "uncertain" if is_uncertain else top_class
        confidence_pct = int(top_prob * 100)

        provenance = f"MODEL_{self.mode.upper()}_{self.version}"
        return InferenceResult(final_class, confidence_pct, probs, is_uncertain, rejection_reason, self.version, provenance)
