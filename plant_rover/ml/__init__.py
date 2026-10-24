"""Machine learning and evaluation package for Plant-Health Rover."""
from .manifest import DatasetManifest, DatasetItem
from .model import ModelInferenceEngine, InferenceResult
from .evaluator import ModelEvaluator, EvaluationReport
from .profiler import profile_inference
from .image_classifier import PlantLeafClassifier, ManualInspectionResult, QualityGateResult

__all__ = [
    "DatasetManifest",
    "DatasetItem",
    "ModelInferenceEngine",
    "InferenceResult",
    "ModelEvaluator",
    "EvaluationReport",
    "profile_inference",
    "PlantLeafClassifier",
    "ManualInspectionResult",
    "QualityGateResult",
]
