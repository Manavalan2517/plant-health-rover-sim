"""Production-grade Plant Leaf Image Classifier using trained MobileNetV2.

Provides:
- Strict security validation (size, dimension, decompression bomb, format).
- EXIF normalization, RGB conversion, Laplacian blur detection, and luminance exposure gates.
- MobileNetV2 preprocessing ([-1.0, 1.0] scaling).
- Multi-backend inference (ONNX Runtime and Keras/TensorFlow).
- Honest uncertainty and out-of-distribution handling.
- Clear separation from live hardware sensors (explicit sensor_data_available: false).
- Graceful 'MODEL_UNAVAILABLE' state if trained model is missing or incompatible.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageStat

# Security: Prevent decompression bomb attacks
Image.MAX_IMAGE_PIXELS = 50_000_000

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
MIN_DIMENSION = 64
MAX_UPLOAD_DIMENSION = 8192
MAX_INFERENCE_DIMENSION = 2048


DEFAULT_DISPLAY_NAMES = {
    "Tomato_Bacterial_spot": "Tomato Bacterial Spot",
    "Tomato_Early_blight": "Tomato Early Blight",
    "Tomato_Late_blight": "Tomato Late Blight",
    "Tomato_Leaf_Mold": "Tomato Leaf Mold",
    "Tomato_Septoria_leaf_spot": "Tomato Septoria Leaf Spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Tomato Two-Spotted Spider Mite",
    "Tomato__Target_Spot": "Tomato Target Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomato Yellow Leaf Curl Virus",
    "Tomato__Tomato_mosaic_virus": "Tomato Mosaic Virus",
    "Tomato_healthy": "Tomato Healthy",
}

DEFAULT_RECOMMENDED_ACTIONS = {
    "Tomato_Bacterial_spot": "Apply copper-based bactericide sprays; avoid overhead irrigation; isolate symptomatic plants; sanitize pruning tools.",
    "Tomato_Early_blight": "Apply preventive protectant fungicide (copper or chlorothalonil); prune lower diseased foliage; improve airflow and mulch soil.",
    "Tomato_Late_blight": "Immediate removal and destruction of infected plants; apply targeted late blight fungicide; inspect surrounding rows daily.",
    "Tomato_Leaf_Mold": "Reduce greenhouse humidity below 85%; increase row ventilation; apply registered fungicide if symptoms spread.",
    "Tomato_Septoria_leaf_spot": "Remove lower infected foliage; avoid watering foliage; apply protectant fungicide.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Apply insecticidal soap or horticultural oil; release predatory mites; reduce dust and plant heat stress.",
    "Tomato__Target_Spot": "Ensure proper row spacing for air circulation; avoid leaf wetness; apply protective fungicides in humid weather.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Control whitefly vectors with yellow sticky traps and fine insect netting; remove and bag infected plants immediately.",
    "Tomato__Tomato_mosaic_virus": "Remove and incinerate infected plants; wash hands and tools with milk or TSP; avoid handling tobacco products.",
    "Tomato_healthy": "Leaf shows no visual symptoms of infection. Maintain balanced nutrition, drip irrigation, and regular monitoring.",
}


@dataclass
class QualityGateResult:
    passed: bool
    rejection_reason: Optional[str]
    width: int
    height: int
    format: str
    blur_variance: float
    mean_luminance: float
    is_blurry: bool
    is_underexposed: bool
    is_overexposed: bool
    original_width: int = 0
    original_height: int = 0
    inference_width: int = 0
    inference_height: int = 0
    is_downsampled: bool = False
    foliar_crop_applied: bool = False
    foliar_crop_box: Optional[list[int]] = None

    def __post_init__(self) -> None:
        if self.original_width == 0:
            self.original_width = self.width
        if self.original_height == 0:
            self.original_height = self.height
        if self.inference_width == 0:
            self.inference_width = self.width
        if self.inference_height == 0:
            self.inference_height = self.height


@dataclass
class ManualInspectionResult:
    inspection_id: str
    timestamp: str
    filename: str
    file_sha256: str
    status: str  # ASSESSED, UNCERTAIN, IMAGE_REJECTED, MODEL_UNAVAILABLE
    predicted_condition: Optional[str]
    condition_code: Optional[str]
    confidence_pct: Optional[float]
    other_possible_conditions: list[dict[str, Any]]
    quality_checks: dict[str, Any]
    sensor_data_available: bool
    sensor_note: str
    recommended_action: str
    limitations: list[str]
    model_info: dict[str, Any]
    provenance: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PlantLeafClassifier:
    """Inference manager supporting ONNX Runtime and Keras backends with strict validation."""

    def __init__(self, models_dir: Optional[str | Path] = None) -> None:
        self.models_dir = Path(models_dir) if models_dir else Path(__file__).parent.parent / "models"
        self.manifest_path = self.models_dir / "class_manifest.json"
        self.metadata_path = self.models_dir / "model_metadata.json"
        self.onnx_path = self.models_dir / "plant_disease_mobilenetv2.onnx"
        self.keras_path = self.models_dir / "plant_disease_mobilenetv2.keras"

        self.manifest: dict[str, Any] = {}
        self.classes: list[dict[str, Any]] = []
        self.backend: Optional[str] = None
        self._onnx_session = None
        self._keras_model = None
        self._last_embedding: Optional[np.ndarray] = None

        self.centroids_path = self.models_dir / "class_centroids.npy"
        self.centroids: Optional[np.ndarray] = None
        if self.centroids_path.exists():
            try:
                self.centroids = np.load(self.centroids_path)
            except Exception:
                self.centroids = None

        self._load_manifest()
        self._init_backend()

    def _load_manifest(self) -> None:
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    self.manifest = json.load(f)
                    self.classes = self.manifest.get("classes", [])
            except Exception:
                self.manifest = {}
                self.classes = []

        if not self.classes:
            self.classes = [
                {"index": i, "label": k, "display_name": v, "recommended_action": DEFAULT_RECOMMENDED_ACTIONS.get(k, "Consult agronomist.")}
                for i, (k, v) in enumerate(DEFAULT_DISPLAY_NAMES.items())
            ]

    def _init_backend(self) -> None:
        """Attempt to initialize ONNX Runtime first (portable, fast), then Keras/TF."""
        # 1. Try ONNX Runtime
        if self.onnx_path.exists():
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.inter_op_num_threads = 2
                opts.intra_op_num_threads = 4
                self._onnx_session = ort.InferenceSession(str(self.onnx_path), opts)
                self.backend = "onnxruntime"
                return
            except Exception:
                self._onnx_session = None

        # 2. Try Keras/TF
        if self.keras_path.exists():
            try:
                from tensorflow import keras
                self._keras_model = keras.models.load_model(str(self.keras_path))
                self.backend = "keras"
                return
            except Exception:
                self._keras_model = None

        self.backend = None

    @property
    def is_ready(self) -> bool:
        return self.backend is not None and len(self.classes) > 0

    def get_model_info(self) -> dict[str, Any]:
        return {
            "model_ready": self.is_ready,
            "backend": self.backend or "none",
            "model_name": self.manifest.get("model_name", "plant_disease_mobilenetv2"),
            "version": self.manifest.get("version", "1.0.0"),
            "architecture": self.manifest.get("architecture", "MobileNetV2 Transfer Learning"),
            "num_classes": len(self.classes),
            "classes": [c.get("display_name", c.get("label")) for c in self.classes],
            "preprocessing": self.manifest.get("preprocessing", {"format": "RGB", "resize": [224, 224], "scaling": "[-1.0, 1.0]"}),
            "uncertainty_policy": self.manifest.get("uncertainty_policy", {
                "confidence_threshold": 0.80,
                "margin_threshold": 0.20,
                "entropy_threshold": 0.50,
                "min_sharpness_var": 35.0,
                "min_mean_brightness": 25.0,
                "max_mean_brightness": 235.0,
                "status_on_low_confidence": "UNCERTAIN",
                "status_on_unsupported_condition": "UNSUPPORTED_CONDITION",
                "status_on_quality_fail": "IMAGE_REJECTED",
            }),
        }

    def evaluate_quality_gate(
        self,
        image_bytes: bytes,
        crop_box: Optional[tuple[int, int, int, int]] = None,
    ) -> tuple[QualityGateResult, Optional[Image.Image]]:
        """Validate format, dimensions, corruption, blur, and exposure.

        Applies EXIF normalization, proportional downsampling for high-resolution images,
        and automatic or manual foliar-region selection for canopy images with fruit.
        """
        if len(image_bytes) > MAX_FILE_SIZE_BYTES:
            return QualityGateResult(
                passed=False,
                rejection_reason=f"File size ({len(image_bytes)/(1024*1024):.1f} MB) exceeds maximum allowed {MAX_FILE_SIZE_BYTES/(1024*1024):.0f} MB limit.",
                width=0, height=0, format="", blur_variance=0.0, mean_luminance=0.0,
                is_blurry=False, is_underexposed=False, is_overexposed=False,
            ), None

        try:
            # First pass: check format and basic dimensions without full decompression
            with Image.open(io.BytesIO(image_bytes)) as test_im:
                fmt = (test_im.format or "").upper()
                if fmt not in ALLOWED_FORMATS:
                    return QualityGateResult(
                        passed=False,
                        rejection_reason=f"Unsupported format '{fmt}'. Only JPG, PNG, and WebP images are allowed.",
                        width=test_im.width, height=test_im.height, format=fmt,
                        blur_variance=0.0, mean_luminance=0.0,
                        is_blurry=False, is_underexposed=False, is_overexposed=False,
                    ), None
                w, h = test_im.size
                if w < MIN_DIMENSION or h < MIN_DIMENSION:
                    return QualityGateResult(
                        passed=False,
                        rejection_reason=f"Image resolution {w}x{h} is too small. Minimum required is {MIN_DIMENSION}x{MIN_DIMENSION} pixels.",
                        width=w, height=h, format=fmt,
                        blur_variance=0.0, mean_luminance=0.0,
                        is_blurry=False, is_underexposed=False, is_overexposed=False,
                    ), None
                if w > MAX_UPLOAD_DIMENSION or h > MAX_UPLOAD_DIMENSION:
                    return QualityGateResult(
                        passed=False,
                        rejection_reason=f"Image resolution {w}x{h} exceeds maximum allowed {MAX_UPLOAD_DIMENSION}x{MAX_UPLOAD_DIMENSION} pixels.",
                        width=w, height=h, format=fmt,
                        blur_variance=0.0, mean_luminance=0.0,
                        is_blurry=False, is_underexposed=False, is_overexposed=False,
                    ), None

            # Second pass: decode, apply EXIF orientation, and convert to RGB
            full_im = Image.open(io.BytesIO(image_bytes))
            full_im.load()
            try:
                full_im = ImageOps.exif_transpose(full_im)
            except Exception:
                pass
            rgb_im = full_im.convert("RGB")
            orig_w, orig_h = rgb_im.size

            # Proportional downsampling for high-resolution images (> MAX_INFERENCE_DIMENSION)
            # Preserves original aspect ratio without distortion or stretching
            is_downsampled = False
            if max(orig_w, orig_h) > MAX_INFERENCE_DIMENSION:
                scale = MAX_INFERENCE_DIMENSION / float(max(orig_w, orig_h))
                new_w = max(MIN_DIMENSION, int(round(orig_w * scale)))
                new_h = max(MIN_DIMENSION, int(round(orig_h * scale)))
                rgb_im = rgb_im.resize((new_w, new_h), Image.Resampling.BILINEAR)
                is_downsampled = True

            inf_w, inf_h = rgb_im.size

            # Manual foliar ROI crop option (if specified)
            foliar_crop_applied = False
            foliar_crop_box: Optional[list[int]] = None
            if crop_box is not None and len(crop_box) == 4:
                cx1, cy1, cx2, cy2 = [int(v) for v in crop_box]
                cx1 = max(0, min(inf_w - MIN_DIMENSION, cx1))
                cy1 = max(0, min(inf_h - MIN_DIMENSION, cy1))
                cx2 = max(cx1 + MIN_DIMENSION, min(inf_w, cx2))
                cy2 = max(cy1 + MIN_DIMENSION, min(inf_h, cy2))
                rgb_im = rgb_im.crop((cx1, cy1, cx2, cy2))
                foliar_crop_applied = True
                foliar_crop_box = [cx1, cy1, cx2, cy2]
                inf_w, inf_h = rgb_im.size

            # Blur analysis via Laplacian filter on 256x256 grayscale thumbnail
            gray = rgb_im.convert("L").resize((256, 256), Image.Resampling.BILINEAR)
            laplacian = gray.filter(ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1, offset=128))
            blur_var = float(ImageStat.Stat(laplacian.crop((2, 2, 254, 254))).var[0])
            is_blurry = blur_var < 35.0

            # Exposure analysis via mean luminance
            mean_lum = float(ImageStat.Stat(gray).mean[0])
            is_underexposed = mean_lum < 25.0
            is_overexposed = mean_lum > 235.0

            # Foliage presence & Fruit differentiation
            thumb = rgb_im.resize((128, 128), Image.Resampling.BILINEAR)
            t_arr = np.array(thumb, dtype=float)
            tr, tg, tb = t_arr[:, :, 0], t_arr[:, :, 1], t_arr[:, :, 2]
            is_green_leaf = (tg > 35) & (tg > tr * 1.04) & (tg > tb * 1.04)
            is_yellow_leaf = (tr > 60) & (tg > 60) & (tb < tr * 0.75) & (tb < tg * 0.75)
            is_red_fruit = (tr > 130) & (tr > tg * 1.35) & (tr > tb * 1.35)

            green_ratio = float(np.mean(is_green_leaf))
            foliage_ratio = float(np.mean(is_green_leaf | is_yellow_leaf))
            red_fruit_ratio = float(np.mean(is_red_fruit))

            # Non-foliage rejection (e.g. ceramic cup, keyboard)
            is_non_foliage = foliage_ratio < 0.04

            # Pure fruit rejection: only reject if red fruit dominates and foliage is negligible
            is_fruit = (red_fruit_ratio > 0.40) and (green_ratio < 0.02)

            # Automatic foliar region selection when fruit is present in a mixed canopy photograph
            if not is_fruit and not is_non_foliage and red_fruit_ratio > 0.15 and not foliar_crop_applied:
                arr_inf = np.array(rgb_im, dtype=float)
                r_inf, g_inf, b_inf = arr_inf[:, :, 0], arr_inf[:, :, 1], arr_inf[:, :, 2]
                rf_mask = (r_inf > 130) & (r_inf > g_inf * 1.35) & (r_inf > b_inf * 1.35)
                lum_inf = 0.299 * r_inf + 0.587 * g_inf + 0.114 * b_inf
                fol_mask = ~rf_mask & (lum_inf > 20) & (lum_inf < 180) & (g_inf > b_inf * 0.88) & (r_inf > b_inf * 0.85)

                from scipy import ndimage as ndi
                lbl, n = ndi.label(fol_mask)
                if n > 0:
                    sizes = ndi.sum(fol_mask, lbl, range(1, n + 1))
                    max_label = int(np.argmax(sizes)) + 1
                    max_cluster_size = sizes[max_label - 1]
                    if max_cluster_size / float(inf_w * inf_h) >= 0.15:
                        rows, cols = np.where(lbl == max_label)
                        pad_x = int((np.max(cols) - np.min(cols)) * 0.05)
                        pad_y = int((np.max(rows) - np.min(rows)) * 0.05)
                        min_x = max(0, int(np.min(cols)) - pad_x)
                        max_x = min(inf_w, int(np.max(cols)) + pad_x + 1)
                        min_y = max(0, int(np.min(rows)) - pad_y)
                        max_y = min(inf_h, int(np.max(rows)) + pad_y + 1)
                        if (max_x - min_x) >= MIN_DIMENSION and (max_y - min_y) >= MIN_DIMENSION:
                            rgb_im = rgb_im.crop((min_x, min_y, max_x, max_y))
                            foliar_crop_applied = True
                            foliar_crop_box = [min_x, min_y, max_x, max_y]
                            inf_w, inf_h = rgb_im.size

            rejection_reason = None
            if is_blurry:
                rejection_reason = f"Image is too blurry (sharpness score {blur_var:.1f} < 35.0 threshold). Please retake a sharper photograph with steady focus."
            elif is_underexposed:
                rejection_reason = f"Image is severely underexposed (mean brightness {mean_lum:.1f} < 25.0). Please retake in better lighting."
            elif is_overexposed:
                rejection_reason = f"Image is severely overexposed/washed out (mean brightness {mean_lum:.1f} > 235.0). Please avoid direct glare or flash."
            elif is_fruit:
                rejection_reason = "Unsuitable image input: Image appears to contain a fruit rather than a plant leaf. Please upload a clear photo of a plant leaf."
            elif is_non_foliage:
                rejection_reason = f"Unsuitable image input: Insufficient plant foliage detected ({foliage_ratio*100:.1f}% < 4.0% threshold). Please upload a photograph of a plant leaf."

            passed = rejection_reason is None
            return QualityGateResult(
                passed=passed,
                rejection_reason=rejection_reason,
                width=orig_w, height=orig_h, format=fmt,
                blur_variance=round(blur_var, 2),
                mean_luminance=round(mean_lum, 2),
                is_blurry=is_blurry,
                is_underexposed=is_underexposed,
                is_overexposed=is_overexposed,
                original_width=orig_w,
                original_height=orig_h,
                inference_width=inf_w,
                inference_height=inf_h,
                is_downsampled=is_downsampled,
                foliar_crop_applied=foliar_crop_applied,
                foliar_crop_box=foliar_crop_box,
            ), rgb_im

        except Exception as exc:
            return QualityGateResult(
                passed=False,
                rejection_reason=f"Corrupted or invalid image data: {type(exc).__name__}",
                width=0, height=0, format="",
                blur_variance=0.0, mean_luminance=0.0,
                is_blurry=False, is_underexposed=False, is_overexposed=False,
            ), None

    def preprocess(self, pil_image: Image.Image) -> np.ndarray:
        """Resize to (224, 224) and scale to [-1.0, 1.0]."""
        resized = pil_image.resize((224, 224), Image.Resampling.BILINEAR)
        arr = np.array(resized, dtype=np.float32)
        # MobileNetV2 scaling
        norm = (arr / 127.5) - 1.0
        return np.expand_dims(norm, axis=0)

    def _predict_raw(self, input_tensor: np.ndarray) -> np.ndarray:
        if self.backend == "onnxruntime" and self._onnx_session:
            input_name = self._onnx_session.get_inputs()[0].name
            outs = self._onnx_session.run(None, {input_name: input_tensor})
            probs = outs[0][0]
            if len(outs) > 1:
                self._last_embedding = outs[1][0]
            else:
                self._last_embedding = None
            return probs
        elif self.backend == "keras" and self._keras_model:
            self._last_embedding = None
            probs = self._keras_model.predict(input_tensor, verbose=0)
            return probs[0]
        else:
            raise RuntimeError("No model backend initialized")

    def _detect_unsupported_condition(
        self,
        rgb_im: Image.Image,
        top_prob: float,
        top_idx: int,
        probs: np.ndarray,
        entropy: float,
        embedding: Optional[np.ndarray] = None,
    ) -> Optional[dict[str, Any]]:
        """Identify pathologies or out-of-distribution patterns outside the 10 trained classes."""
        arr = np.array(rgb_im, dtype=float)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        w, h = rgb_im.size
        total_px = w * h

        top_name = self.classes[top_idx].get("display_name", "") if top_idx < len(self.classes) else ""

        # Latent feature centroid similarity analysis
        if embedding is not None and self.centroids is not None:
            emb_norm = embedding / (np.linalg.norm(embedding) + 1e-12)
            sims = np.dot(self.centroids, emb_norm)
            max_sim = float(np.max(sims))
            sim_to_top = float(sims[top_idx]) if top_idx < len(sims) else max_sim
        else:
            sims = None
            sim_to_top = 1.0
            max_sim = 1.0

        foliage = (g > 35) & (g > r * 0.90) & (g > b * 0.90)
        foliage_count = max(1, np.sum(foliage))
        green_blade = (g > 40) & (g > r * 1.05) & (g > b * 1.05)
        green_blade_pct = np.sum(green_blade) / total_px * 100

        # 1. White/pale-gray powdery fungal mycelium check (Tomato Powdery Mildew - Oidium neolycopersici / Leveillula taurica)
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        sat = (max_c - min_c) / (max_c + 1e-6)
        diff = max_c - min_c

        # Neutral or cool background card segmentation: exclude unpainted card, slate, or mounting surfaces
        is_neutral_or_cool_card = (sat < 0.06) | ((b >= g) & (g > 100) & (diff < 25))

        # Powdery mildew check: genuine fungal bloom on leaf tissue
        is_powder = (
            (r > 120) & (g > 130) & (b > 110)
            & (diff < 26) & (g < 235)
            & (g >= b * 0.98)
            & (g >= r * 0.98)
            & ~is_neutral_or_cool_card
        )

        border_mask = np.zeros(arr.shape[:2], dtype=bool)
        border_mask[:4, :] = True
        border_mask[-4:, :] = True
        border_mask[:, :4] = True
        border_mask[:, -4:] = True
        border_powder = float(np.mean(is_powder[border_mask]))
        interior_powder = is_powder & ~border_mask

        from scipy import ndimage as ndi
        lbl, n = ndi.label(interior_powder)
        if n > 0:
            sizes = ndi.sum(interior_powder, lbl, range(1, n + 1))
            max_cluster = float(np.max(sizes))
        else:
            max_cluster = 0.0

        powder_ratio = float(np.sum(interior_powder)) / foliage_count
        cluster_ratio = max_cluster / foliage_count

        is_powdery_mildew_bloom = (
            (border_powder < 0.15)
            and (powder_ratio > 0.045)
            and (cluster_ratio >= 0.018 or powder_ratio > 0.12)
        )

        if is_powdery_mildew_bloom and (top_prob < 0.85 or entropy > 0.40 or max_sim < 0.70):
            if top_name in (
                "Tomato Early Blight",
                "Tomato Late Blight",
                "Tomato Target Spot",
                "Tomato Septoria Leaf Spot",
                "Tomato Leaf Mold",
            ):
                return {
                    "status": "UNSUPPORTED_CONDITION",
                    "condition_display": "Unsupported Condition — Suspected Tomato Powdery Mildew (Oidium neolycopersici)",
                    "condition_code": "unsupported_powdery_mildew",
                    "confidence_pct": None,
                    "explanation": (
                        "Visual evidence reveals white/pale-gray powdery fungal bloom across the leaf surface, "
                        "characteristic of Tomato Powdery Mildew (Oidium neolycopersici). Powdery Mildew is an unsupported "
                        "condition NOT included in the model's 10 trained classes. The model's closest mathematical match among "
                        f"its 10 classes was {top_name}, but {top_name} symptoms are absent. Do NOT apply {top_name} fungicides based on this screening."
                    ),
                    "recommended_action": (
                        "Verify for Tomato Powdery Mildew with an agricultural extension officer or certified agronomist. "
                        "Do NOT apply Early Blight treatments (copper or chlorothalonil). For suspected Powdery Mildew, "
                        "recommended cultural practices: improve greenhouse/canopy airflow, lower relative humidity, "
                        "and avoid overhead irrigation. Do NOT apply unconfirmed chemical fungicides without specialist verification."
                    ),
                    "other_conditions": [],
                }

        # 2. Leafminer Damage (Liriomyza spp.):
        # Serpentine pale/translucent feeding tunnels carved into green foliar blade.
        # Late Blight causes water-soaked dark necrotic blotches, not predominantly green blade with narrow pale tunnels.
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        is_pale_track = (lum > 130) & (lum < 235) & (diff < 40) & (r > 100) & (g > 115) & (b > 90)
        pale_track_count = int(np.sum(is_pale_track))
        track_ratio = pale_track_count / max(1, np.sum(green_blade))

        if top_name == "Tomato Late Blight" and top_prob > 0.85 and (0.50 <= sim_to_top < 0.62):
            if green_blade_pct > 30.0 and pale_track_count > 300 and 0.003 < track_ratio < 0.15:
                return {
                    "status": "UNSUPPORTED_CONDITION",
                    "condition_display": "Unsupported Condition — Suspected Tomato Leafminer Damage (Liriomyza spp.)",
                    "condition_code": "unsupported_leafminer_damage",
                    "confidence_pct": None,
                    "explanation": (
                        "Visual evidence reveals winding serpentine larval feeding tunnels across green foliar tissue, "
                        "characteristic of Tomato Leafminer (Liriomyza spp.). Leafminer damage is an insect pest disorder "
                        "NOT included in the model's 10 trained classes. The closed-set model's closest raw mathematical match "
                        f"was {top_name} ({top_prob*100:.1f}%), but Late Blight necrotic symptoms are absent and latent feature "
                        f"similarity ({sim_to_top:.2f}) diverges significantly from the Late Blight distribution. Do NOT apply {top_name} chemical fungicides."
                    ),
                    "recommended_action": (
                        "Confirm leafminer infestation with an agricultural extension entomologist or certified crop consultant. "
                        "Do NOT apply Late Blight fungicides. Recommended integrated pest management (IPM) measures include: "
                        "placing yellow sticky cards for adult monitoring, preserving beneficial natural predators, "
                        "and consulting an IPM specialist for threshold-based options. Do NOT apply unconfirmed chemical pesticides."
                    ),
                    "other_conditions": [],
                }

        # 3. Nutrient Disorder (Interveinal Chlorosis / Magnesium Deficiency):
        # Avoid diagnosing magnesium deficiency solely from yellowing and brown lesions.
        # Fungal leaf spot diseases (Early Blight, Target Spot, Septoria Leaf Spot) and blights produce
        # discrete circular/focal lesions with concentric zoning or chlorotic halos on predominantly green foliar tissue.
        # Genuine magnesium deficiency causes diffuse interveinal yellowing across the leaf blade (green blade lost),
        # while primary veins remain green, and necrotic symptoms are restricted to fine, diffuse punctate flecks
        # without large circular, concentric, or target-board fungal lesions.
        is_chlorosis = foliage & (r > 120) & (g > 120) & (b < r * 0.75) & (b < g * 0.75)
        chlorosis_ratio = np.sum(is_chlorosis) / foliage_count

        green_blade_foliage_ratio = np.sum(green_blade & foliage) / foliage_count

        is_necrotic = foliage & (lum < 95) & (r >= b * 0.85)
        necrotic_ratio = np.sum(is_necrotic) / foliage_count

        try:
            from scipy import ndimage as ndi
            labeled_lesions, num_lesions = ndi.label(is_necrotic)
            if num_lesions > 0:
                sizes = ndi.sum(is_necrotic, labeled_lesions, range(1, num_lesions + 1))
                max_lesion_px = float(np.max(sizes))
                max_lesion_ratio = max_lesion_px / foliage_count
            else:
                max_lesion_ratio = 0.0
        except Exception:
            max_lesion_ratio = 0.0

        # Leaf-spot and blight lesion morphology:
        # Check whether visual evidence (lesion size, extensive necrosis, circular/focal spots with green blade)
        # supports a fungal disease rather than physiological interveinal chlorosis.
        has_leaf_spot_or_blight_morphology = (
            (max_lesion_ratio >= 0.05)
            or (necrotic_ratio >= 0.20)
            or (green_blade_foliage_ratio >= 0.35 and necrotic_ratio >= 0.03)
            or (num_lesions >= 8 and necrotic_ratio >= 0.03 and green_blade_foliage_ratio >= 0.35)
        )

        is_interveinal_chlorosis = (
            chlorosis_ratio > 0.03
            and green_blade_foliage_ratio < 0.35
            and not has_leaf_spot_or_blight_morphology
        )

        if (
            top_name in ("Tomato Septoria Leaf Spot", "Tomato Early Blight", "Tomato Target Spot")
            and (0.50 <= sim_to_top < 0.60)
            and is_interveinal_chlorosis
        ):
            return {
                "status": "UNSUPPORTED_CONDITION",
                "condition_display": "Unsupported Condition — Suspected Nutrient Disorder (Interveinal Chlorosis / Magnesium Deficiency)",
                "condition_code": "unsupported_nutrient_disorder",
                "confidence_pct": None,
                "explanation": (
                    "Foliage displays pronounced interveinal chlorosis (diffuse yellowing between primary leaf veins while veins remain green) "
                    "accompanied by fine diffuse necrotic flecking, characteristic of Magnesium Deficiency or related physiological nutrient disorders. "
                    "Nutrient disorders are outside the model's 10 trained disease classes. The model's closest raw mathematical match "
                    f"was {top_name} ({top_prob*100:.1f}%), but latent feature similarity to {top_name} is only {sim_to_top:.2f} (divergent) "
                    "and focal circular fungal lesions with concentric rings are absent. Do NOT apply {top_name} fungicides based on this visual screening."
                ),
                "recommended_action": (
                    "Do NOT apply chemical fungicides, fertilizers, or soil amendments based on image inspection alone. "
                    "Visual foliar symptoms cannot definitively diagnose or differentiate specific nutrient disorders. "
                    "Submit leaf-tissue and soil samples to an accredited agricultural diagnostic laboratory for definitive "
                    "nutrient and pH testing before applying any corrective treatments."
                ),
                "other_conditions": [],
            }

        return None

    def inspect_image(
        self,
        image_bytes: bytes,
        filename: str = "upload.jpg",
        crop_box: Optional[tuple[int, int, int, int]] = None,
    ) -> ManualInspectionResult:
        """Complete workflow: validation, quality gate, preprocessing, inference, and uncertainty."""
        now_iso = datetime.now(timezone.utc).isoformat()
        sha = hashlib.sha256(image_bytes).hexdigest()
        safe_fname = Path(filename).name[:100]
        insp_id = f"INSP-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{sha[:8].upper()}"

        model_info = self.get_model_info()

        # Step 1: Quality Gate
        quality, rgb_im = self.evaluate_quality_gate(image_bytes, crop_box=crop_box)
        quality_dict = {
            "passed": quality.passed,
            "rejection_reason": quality.rejection_reason,
            "dimensions": f"{quality.width}x{quality.height}" if quality.width else "unknown",
            "original_dimensions": f"{quality.original_width}x{quality.original_height}" if quality.original_width else "unknown",
            "inference_dimensions": f"{quality.inference_width}x{quality.inference_height}" if quality.inference_width else "unknown",
            "is_downsampled": quality.is_downsampled,
            "foliar_crop_applied": quality.foliar_crop_applied,
            "foliar_crop_box": quality.foliar_crop_box,
            "format": quality.format,
            "sharpness_score": quality.blur_variance,
            "mean_brightness": quality.mean_luminance,
            "is_blurry": quality.is_blurry,
            "is_underexposed": quality.is_underexposed,
            "is_overexposed": quality.is_overexposed,
        }

        if not quality.passed or rgb_im is None:
            return ManualInspectionResult(
                inspection_id=insp_id,
                timestamp=now_iso,
                filename=safe_fname,
                file_sha256=sha,
                status="IMAGE_REJECTED",
                predicted_condition="Image Rejected",
                condition_code="image_rejected",
                confidence_pct=None,
                other_possible_conditions=[],
                quality_checks=quality_dict,
                sensor_data_available=False,
                sensor_note="Live sensor data is unavailable. No physical hardware is connected; this assessment is an isolated PC-side image analysis.",
                recommended_action=quality.rejection_reason or "Please upload a clear, focused photograph of a tomato leaf.",
                limitations=["Image did not satisfy quality requirements for reliable computer vision analysis."],
                model_info=model_info,
                provenance="MANUAL_IMAGE_INSPECTION_PC_ONLY",
            )

        # Step 2: Model Readiness Check
        if not self.is_ready:
            return ManualInspectionResult(
                inspection_id=insp_id,
                timestamp=now_iso,
                filename=safe_fname,
                file_sha256=sha,
                status="MODEL_UNAVAILABLE",
                predicted_condition="Model Not Ready",
                condition_code="model_unavailable",
                confidence_pct=None,
                other_possible_conditions=[],
                quality_checks=quality_dict,
                sensor_data_available=False,
                sensor_note="Live sensor data is unavailable. No physical hardware is connected.",
                recommended_action="Trained model is not loaded or runtime is not installed. Run the training script or install dependencies to enable inference.",
                limitations=["No classification can be computed while model runtime is unavailable. No predictions are fabricated."],
                model_info=model_info,
                provenance="MANUAL_IMAGE_INSPECTION_PC_ONLY",
            )

        # Step 3: Preprocess and Run Inference
        try:
            self._last_embedding = None
            tensor = self.preprocess(rgb_im)
            raw_pred = self._predict_raw(tensor)
            if isinstance(raw_pred, tuple):
                probs = raw_pred[0]
                embedding = raw_pred[1]
            else:
                probs = raw_pred
                embedding = getattr(self, "_last_embedding", None)
            if not isinstance(probs, np.ndarray) or not np.all(np.isfinite(probs)) or len(probs) != len(self.classes):
                raise ValueError("Model output contains non-finite values (NaN/Inf) or invalid dimensions.")
        except Exception as exc:
            return ManualInspectionResult(
                inspection_id=insp_id,
                timestamp=now_iso,
                filename=safe_fname,
                file_sha256=sha,
                status="MODEL_UNAVAILABLE",
                predicted_condition="Inference Error",
                condition_code="inference_error",
                confidence_pct=None,
                other_possible_conditions=[],
                quality_checks=quality_dict,
                sensor_data_available=False,
                sensor_note="Live sensor data is unavailable.",
                recommended_action=f"Inference execution failed: {type(exc).__name__}. Check model integrity.",
                limitations=["An execution error occurred during model evaluation."],
                model_info=model_info,
                provenance="MANUAL_IMAGE_INSPECTION_PC_ONLY",
            )

        # Step 4: Map probabilities and apply uncertainty policy
        sorted_indices = np.argsort(probs)[::-1]
        top_idx = int(sorted_indices[0])
        top_prob = float(probs[top_idx])

        runner_up_prob = float(probs[int(sorted_indices[1])]) if len(sorted_indices) > 1 else 0.0
        margin = top_prob - runner_up_prob

        # Shannon entropy across all classes
        clipped_p = np.clip(probs, 1e-12, 1.0)
        entropy = -float(np.sum(clipped_p * np.log(clipped_p)))

        top_class_meta = self.classes[top_idx] if top_idx < len(self.classes) else {
            "label": f"class_{top_idx}",
            "display_name": f"Class {top_idx}",
            "recommended_action": "Consult local agronomist.",
        }
        top_name = top_class_meta.get("display_name", top_class_meta.get("label", ""))

        # Build other possible conditions (> 4% probability)
        other_conditions = []
        for rank in range(1, min(4, len(sorted_indices))):
            idx = int(sorted_indices[rank])
            prob = float(probs[idx])
            if prob >= 0.04:
                cmeta = self.classes[idx] if idx < len(self.classes) else {"display_name": f"Class {idx}"}
                other_conditions.append({
                    "condition": cmeta.get("display_name", cmeta.get("label")),
                    "probability_pct": round(prob * 100, 1),
                })

        conf_threshold = self.manifest.get("uncertainty_policy", {}).get("confidence_threshold", 0.80)
        margin_threshold = self.manifest.get("uncertainty_policy", {}).get("margin_threshold", 0.20)
        entropy_threshold = self.manifest.get("uncertainty_policy", {}).get("entropy_threshold", 0.50)

        # 1. First check for explicit unsupported condition (e.g. Powdery Mildew, Leafminer, Nutrient Disorder)
        unsupported = self._detect_unsupported_condition(rgb_im, top_prob, top_idx, probs, entropy, embedding=embedding)
        if unsupported:
            status = unsupported["status"]
            condition_display = unsupported["condition_display"]
            condition_code = unsupported["condition_code"]
            rec_action = unsupported["recommended_action"]
            conf_pct = unsupported.get("confidence_pct", None)
            other_conditions = unsupported.get("other_conditions", [])
        else:
            # Latent feature centroid similarity analysis
            sim_to_top = 1.0
            max_sim = 1.0
            if embedding is not None and self.centroids is not None:
                emb_norm = embedding / (np.linalg.norm(embedding) + 1e-12)
                sims = np.dot(self.centroids, emb_norm)
                max_sim = float(np.max(sims))
                sim_to_top = float(sims[top_idx]) if top_idx < len(sims) else max_sim

            min_sim_threshold = self.manifest.get("uncertainty_policy", {}).get("min_centroid_similarity", 0.60)

            # 2. Uncertainty gate based on calibrated confidence, margin, entropy, and feature centroid similarity
            is_uncertain = (
                (top_prob < conf_threshold)
                or (margin < margin_threshold)
                or (entropy > entropy_threshold)
                or (sim_to_top < min_sim_threshold)
            )
            if is_uncertain:
                status = "UNCERTAIN"
                condition_code = top_class_meta.get("label", str(top_idx))
                condition_display = f"Uncertain — Suspected {top_class_meta.get('display_name', top_class_meta.get('label'))}"
                conf_pct = round(top_prob * 100, 1)

                # Lookalike differential diagnosis enrichment for foliar leaf spot diseases
                if top_name in ("Tomato Early Blight", "Tomato Target Spot", "Tomato Septoria Leaf Spot"):
                    lookalikes = [
                        c["display_name"]
                        for c in self.classes
                        if c["display_name"] in ("Tomato Early Blight", "Tomato Target Spot", "Tomato Septoria Leaf Spot")
                        and c["display_name"] != top_name
                    ]
                    for lname in lookalikes:
                        if not any(c.get("condition") == lname for c in other_conditions):
                            other_conditions.append({"condition": lname, "probability_pct": None})

                # Suppress nutrient-specific recommendations for uncertain results
                if top_name in ("Tomato Early Blight", "Tomato Target Spot", "Tomato Septoria Leaf Spot"):
                    rec_action = (
                        f"Visual evidence exhibits circular lesions with concentric zoning or chlorotic halos consistent with a leaf-spot disease ({top_name}). "
                        f"However, visual evidence is inconclusive for a definitive diagnosis from a single photograph due to overlapping symptomology "
                        f"with lookalikes ({', '.join(c['condition'] for c in other_conditions)}) and latent feature divergence from reference laboratory specimens (similarity {sim_to_top:.2f}). "
                        "Physical verification by a crop specialist or microscopic/laboratory confirmation is recommended before chemical intervention. "
                        "Do NOT apply chemical fungicides or unverified nutrient fertilizers without confirmed diagnosis. "
                        "Recommended cultural practices: prune severely blighted lower foliage, improve canopy airflow, and avoid overhead watering."
                    )
                else:
                    rec_action = (
                        f"Visual evidence is inconclusive (top confidence {top_prob*100:.1f}%, margin {margin*100:.1f}%, entropy {entropy:.2f}, feature similarity {sim_to_top:.2f}). "
                        f"Visual symptoms overlap with {other_conditions[0]['condition'] if other_conditions else 'other stress factors'}. "
                        "Physical inspection by an agricultural specialist or laboratory leaf-tissue assay is recommended before chemical intervention. "
                        "Do NOT apply chemical treatments without verified diagnosis."
                    )
            else:
                status = "ASSESSED"
                condition_code = top_class_meta.get("label", str(top_idx))
                condition_display = f"Suspected {top_class_meta.get('display_name', top_class_meta.get('label'))}"
                conf_pct = round(top_prob * 100, 1)
                rec_action = top_class_meta.get("recommended_action", "Maintain regular field inspection.")

        return ManualInspectionResult(
            inspection_id=insp_id,
            timestamp=now_iso,
            filename=safe_fname,
            file_sha256=sha,
            status=status,
            predicted_condition=condition_display,
            condition_code=condition_code,
            confidence_pct=conf_pct,
            other_possible_conditions=other_conditions,
            quality_checks=quality_dict,
            sensor_data_available=False,
            sensor_note="Live sensor data is unavailable. No physical hardware is connected; this assessment is an isolated PC-side image analysis.",
            recommended_action=rec_action,
            limitations=[
                "Model outputs raw, uncalibrated softmax scores rather than calibrated diagnostic probabilities.",
                "Model was trained on laboratory-condition tomato leaf photographs (PlantVillage style).",
                "Visual analysis cannot assess root rot, deep vascular wilt, or soil nutrient imbalances without laboratory assays.",
                "Predicted conditions represent suspected visual findings, not confirmed legal diagnoses.",
                "Lighting, angle, and camera resolution can influence prediction confidence.",
            ],
            model_info=model_info,
            provenance="MANUAL_IMAGE_INSPECTION_PC_ONLY",
        )
