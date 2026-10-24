"""Profiling tool for measuring PC inference latency and resource footprint."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any
import numpy as np

from .model import ModelInferenceEngine


def profile_inference(engine: ModelInferenceEngine, sample_image_path: str | Path, iterations: int = 50) -> dict[str, Any]:
    """Measures latency statistics across multiple inference iterations on the PC CPU."""
    p = Path(sample_image_path)
    if not p.exists():
        raise FileNotFoundError(f"Sample image not found: {p}")

    # Warm-up run
    engine.infer(p)

    latencies_ms: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        engine.infer(p)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt)

    arr = np.array(latencies_ms)
    return {
        "engine_mode": engine.mode,
        "model_version": engine.version,
        "iterations": iterations,
        "mean_latency_ms": round(float(np.mean(arr)), 2),
        "median_latency_ms": round(float(np.median(arr)), 2),
        "p95_latency_ms": round(float(np.percentile(arr, 95)), 2),
        "min_latency_ms": round(float(np.min(arr)), 2),
        "max_latency_ms": round(float(np.max(arr)), 2),
        "fps_estimate": round(1000.0 / max(0.1, float(np.mean(arr))), 1),
    }
