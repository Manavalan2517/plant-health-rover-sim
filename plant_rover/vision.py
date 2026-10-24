"""PC-only camera and executable image-quality gate.

The generated PNGs are synthetic. The fixture cue is test metadata, not an AI prediction.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFilter, ImageStat

from .hal import BaseCamera


@dataclass(frozen=True)
class QualityResult:
    accepted: bool
    score: int
    sharpness_variance: float
    mean_brightness: float
    clipped_fraction: float
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


class ImageQualityGate:
    """Simple deterministic quality gate; thresholds are engineering placeholders."""

    def __init__(self, min_sharpness_variance: float = 160.0, min_brightness: float = 28.0,
                 max_brightness: float = 215.0, max_clipped_fraction: float = 0.38) -> None:
        self.min_sharpness_variance = min_sharpness_variance
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.max_clipped_fraction = max_clipped_fraction

    def evaluate(self, image_path: str | Path) -> QualityResult:
        try:
            with Image.open(image_path) as source:
                image = source.convert("L")
                pixels = list(image.get_flattened_data() if hasattr(image, "get_flattened_data") else image.getdata())
                if not pixels:
                    return QualityResult(False, 0, 0, 0, 1, "empty_image")
                brightness = sum(pixels) / len(pixels)
                clipped = sum(1 for p in pixels if p <= 5 or p >= 250) / len(pixels)
                laplacian = image.filter(ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1, offset=128))
                sharpness = ImageStat.Stat(laplacian).var[0]
        except (OSError, ValueError) as exc:
            return QualityResult(False, 0, 0, 0, 1, f"unreadable_image:{type(exc).__name__}")

        reasons = []
        if sharpness < self.min_sharpness_variance:
            reasons.append("blur")
        if brightness < self.min_brightness:
            reasons.append("underexposed")
        if brightness > self.max_brightness:
            reasons.append("overexposed")
        if clipped > self.max_clipped_fraction:
            reasons.append("too_many_clipped_pixels")
        score = max(0, min(100, int(100 - (self.min_sharpness_variance / max(sharpness, 0.1)) * 10
                            - abs(brightness - 128) * 0.18 - clipped * 40)))
        return QualityResult(not reasons, score, float(sharpness), float(brightness), float(clipped),
                             "accepted" if not reasons else ",".join(reasons))


class SyntheticCamera(BaseCamera):
    """Creates visibly synthetic tomato-leaf-like test fixtures, never ground truth."""

    def __init__(self, output_dir: str | Path, seed: int = 2026) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed

    def capture(self, session_id: str, plant_id: str, cue: str, condition: str = "normal", attempt: int = 1) -> Path | None:
        if condition == "capture_failure":
            return None
        seed = self.seed + sum(ord(ch) for ch in f"{session_id}:{plant_id}:{attempt}:{cue}")
        rng = random.Random(seed)
        image = Image.new("RGB", (320, 240), (39, 46, 38))
        draw = ImageDraw.Draw(image)
        # A hood-like, fixed background and one stylised leaf. It is clearly synthetic.
        draw.rounded_rectangle((18, 14, 302, 226), radius=18, fill=(56, 63, 47), outline=(126, 131, 86), width=2)
        leaf_color = {
            "healthy_looking_fixture": (56, 152, 65),
            "spot_like_fixture": (67, 128, 53),
            "wilt_like_fixture": (100, 137, 63),
            "yellowing_like_fixture": (178, 169, 60),
            "unknown": (76, 104, 65),
        }.get(cue, (56, 152, 65))
        draw.ellipse((78, 32, 244, 207), fill=leaf_color, outline=(176, 188, 133), width=3)
        draw.line((160, 198, 160, 43), fill=(195, 202, 139), width=3)
        for n in range(9):
            y = 55 + n * 15
            draw.line((160, y, 125 + rng.randint(-8, 8), y - 9), fill=(151, 184, 105), width=2)
            draw.line((160, y + 4, 195 + rng.randint(-8, 8), y - 6), fill=(151, 184, 105), width=2)
        if cue == "spot_like_fixture":
            for _ in range(10):
                x, y = rng.randint(105, 214), rng.randint(54, 181)
                radius = rng.randint(2, 7)
                draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=(116, 83, 37), outline=(217, 184, 70))
        if cue == "yellowing_like_fixture":
            draw.arc((92, 45, 229, 195), 25, 150, fill=(226, 212, 92), width=9)
        if condition == "no_leaf":
            image = Image.new("RGB", (320, 240), (78, 86, 75))
        if condition == "underexposed":
            image = Image.eval(image, lambda px: int(px * 0.18))
        elif condition == "overexposed":
            image = Image.eval(image, lambda px: min(255, int(px * 2.8 + 75)))
        elif condition == "blur":
            image = image.filter(ImageFilter.GaussianBlur(radius=5.5))
        path = self.output_dir / session_id / f"{plant_id}-attempt-{attempt}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        return path


class RecordedImageCamera(BaseCamera):
    """Loads actual recorded images from a directory, fulfilling Master Prompt Section 6.3."""

    def __init__(self, recorded_dir: str | Path, output_dir: str | Path) -> None:
        self.recorded_dir = Path(recorded_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def capture(self, session_id: str, plant_id: str, cue: str, condition: str = "normal", attempt: int = 1) -> Path | None:
        if condition == "capture_failure":
            return None
        # Check if an exact image or cue-specific image exists in recorded_dir
        candidate = None
        if self.recorded_dir.exists():
            specific = list(self.recorded_dir.glob(f"*{cue}*")) or list(self.recorded_dir.glob("*.jpg")) or list(self.recorded_dir.glob("*.png"))
            if specific:
                candidate = specific[0]

        dest = self.output_dir / session_id / f"{plant_id}-attempt-{attempt}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if candidate and candidate.exists():
            with Image.open(candidate) as img:
                img.save(dest)
            return dest

        # Fallback to a marked test frame if recorded file not found
        fallback = Image.new("RGB", (320, 240), (60, 60, 60))
        draw = ImageDraw.Draw(fallback)
        draw.text((20, 110), f"RECORDED_FRAME:{plant_id}:{cue}", fill=(220, 220, 220))
        fallback.save(dest)
        return dest

