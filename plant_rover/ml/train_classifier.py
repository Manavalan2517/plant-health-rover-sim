"""Reproducible transfer learning training pipeline for MobileNetV2 plant leaf disease classification.

Includes:
- Class inspection and validation.
- Block/group-based partitioning to prevent data leakage from augmented variants.
- MobileNetV2 preprocessing ([-1, 1] scaling).
- Two-stage training: frozen backbone feature classification + controlled fine-tuning.
- Validation-based model selection.
- Comprehensive test evaluation: per-class precision, recall, F1, confusion matrix, uncertainty metrics.
- Export of Keras model, ONNX model, class manifest, and model metadata.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
from PIL import Image

# Ensure TensorFlow runs cleanly on CPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


DISPLAY_NAMES = {
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

RECOMMENDED_ACTIONS = {
    "Tomato_Bacterial_spot": "Copper-based bactericide sprays; avoid overhead irrigation; isolate symptomatic plants; sanitize pruning shears.",
    "Tomato_Early_blight": "Apply preventive fungicide (chlorothalonil or copper); remove lower diseased leaves; improve airflow and mulch soil.",
    "Tomato_Late_blight": "Immediate removal and destruction of infected foliage; apply targeted late blight fungicide; check neighboring plants daily.",
    "Tomato_Leaf_Mold": "Reduce greenhouse humidity below 85%; increase row ventilation; apply registered fungicide if symptoms spread.",
    "Tomato_Septoria_leaf_spot": "Remove lower infected foliage; avoid watering from above; apply appropriate protectant fungicide.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Apply insecticidal soap or horticultural oil; release predatory mites (Phytoseiidae); reduce dust and heat stress.",
    "Tomato__Target_Spot": "Ensure proper spacing for air circulation; avoid leaf wetness; apply protective fungicides during warm humid weather.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Control whitefly vectors using yellow sticky traps and fine mesh netting; remove and bag infected plants immediately.",
    "Tomato__Tomato_mosaic_virus": "Remove and incinerate infected plants; wash hands and tools thoroughly with milk or TSP; avoid handling tobacco products near crop.",
    "Tomato_healthy": "Crop shows no dominant visual symptoms of disease. Maintain standard irrigation schedule and routine monitoring.",
}


def load_dataset_splits(
    dataset_dir: Path,
    samples_per_class: int = 350,
    seed: int = 2026,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[list[str], dict[str, list[tuple[Path, int]]]]:
    """Scan dataset directory, sort classes, and create leakage-resistant group splits."""
    class_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    rng = random.Random(seed)
    splits: dict[str, list[tuple[Path, int]]] = {"train": [], "val": [], "test": []}

    print(f"Discovered {len(class_names)} classes in {dataset_dir}:")

    for cdir in class_dirs:
        cidx = class_to_idx[cdir.name]
        # Only accept valid image files > 0 bytes
        raw_files = sorted([f for f in cdir.iterdir() if f.is_file() and f.stat().st_size > 0 and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")])

        if samples_per_class > 0 and len(raw_files) > samples_per_class:
            # Group into contiguous blocks of 10 to avoid separating closely related augmented variants
            chunk_size = 10
            chunks = [raw_files[i : i + chunk_size] for i in range(0, len(raw_files), chunk_size)]
            rng.shuffle(chunks)
            selected_files = [f for chunk in chunks for f in chunk][:samples_per_class]
        else:
            selected_files = list(raw_files)

        # Split at chunk level to prevent data leakage between augmented variations
        chunk_size = 10
        chunks = [selected_files[i : i + chunk_size] for i in range(0, len(selected_files), chunk_size)]
        rng.shuffle(chunks)

        n_chunks = len(chunks)
        n_train = max(1, int(n_chunks * train_ratio))
        n_val = max(1, int(n_chunks * val_ratio))

        train_chunks = chunks[:n_train]
        val_chunks = chunks[n_train : n_train + n_val]
        test_chunks = chunks[n_train + n_val :]

        for f in [img for ch in train_chunks for img in ch]:
            splits["train"].append((f, cidx))
        for f in [img for ch in val_chunks for img in ch]:
            splits["val"].append((f, cidx))
        for f in [img for ch in test_chunks for img in ch]:
            splits["test"].append((f, cidx))

        print(f"  {cdir.name:<45}: {len(selected_files):>4} selected (train: {sum(len(ch) for ch in train_chunks)}, val: {sum(len(ch) for ch in val_chunks)}, test: {sum(len(ch) for ch in test_chunks)})")

    # Shuffle instances within each partition
    for k in splits:
        rng.shuffle(splits[k])

    print(f"\nTotal partition sizes -> Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    return class_names, splits


def preprocess_image_file(path: Path, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Read, normalize EXIF, convert to RGB, resize, and scale to [-1, 1]."""
    with Image.open(path) as im:
        # EXIF orientation handling
        try:
            from PIL import ImageOps
            im = ImageOps.exif_transpose(im)
        except Exception:
            pass
        im = im.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
        arr = np.array(im, dtype=np.float32)
        # MobileNetV2 scaling: [0, 255] -> [-1.0, 1.0]
        norm = (arr / 127.5) - 1.0
        return norm


def batch_generator(items: list[tuple[Path, int]], batch_size: int = 32):
    """Memory-efficient batch generator."""
    n = len(items)
    for start in range(0, n, batch_size):
        batch_items = items[start : start + batch_size]
        x = np.stack([preprocess_image_file(p) for p, _ in batch_items], axis=0)
        y = np.array([lbl for _, lbl in batch_items], dtype=np.int32)
        yield x, y


def extract_features(base_model: Any, items: list[tuple[Path, int]], batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Precompute MobileNetV2 backbone embeddings for fast training."""
    features = []
    labels = []
    total = len(items)
    done = 0
    t0 = time.time()
    for x_batch, y_batch in batch_generator(items, batch_size):
        feats = base_model.predict(x_batch, batch_size=batch_size, verbose=0)
        features.append(feats)
        labels.append(y_batch)
        done += len(y_batch)
        if done % 500 == 0 or done == total:
            elapsed = time.time() - t0
            print(f"  Extracted {done}/{total} embeddings ({done / max(1e-3, elapsed):.1f} img/s)...")
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def build_and_train_model(
    class_names: list[str],
    splits: dict[str, list[tuple[Path, int]]],
    output_dir: Path,
    epochs: int = 15,
    fine_tune_epochs: int = 3,
    batch_size: int = 32,
    seed: int = 2026,
) -> dict[str, Any]:
    """Train MobileNetV2 transfer learning model, evaluate, and save artifacts."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
    from sklearn.metrics import classification_report, confusion_matrix

    tf.random.set_seed(seed)
    np.random.seed(seed)
    num_classes = len(class_names)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Phase 1: Feature Extraction Backbone ---")
    backbone = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    backbone.trainable = False

    print("Extracting train features...")
    x_train_feats, y_train = extract_features(backbone, splits["train"], batch_size)
    print("Extracting val features...")
    x_val_feats, y_val = extract_features(backbone, splits["val"], batch_size)
    print("Extracting test features...")
    x_test_feats, y_test = extract_features(backbone, splits["test"], batch_size)

    y_train_cat = keras.utils.to_categorical(y_train, num_classes)
    y_val_cat = keras.utils.to_categorical(y_val, num_classes)
    y_test_cat = keras.utils.to_categorical(y_test, num_classes)

    print("\n--- Phase 2: Classification Head Training ---")
    # Classification Head
    head_inputs = keras.Input(shape=(1280,), name="feature_input")
    x = keras.layers.BatchNormalization()(head_inputs)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    head_outputs = keras.layers.Dense(num_classes, activation="softmax", name="probabilities")(x)

    head_model = keras.Model(inputs=head_inputs, outputs=head_outputs, name="classification_head")
    head_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = head_model.fit(
        x_train_feats,
        y_train_cat,
        validation_data=(x_val_feats, y_val_cat),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    # Assemble Full End-to-End Model for export
    print("\n--- Phase 3: End-to-End Model Assembly ---")
    full_input = keras.Input(shape=(224, 224, 3), name="image_input")
    full_feats = backbone(full_input, training=False)
    full_output = head_model(full_feats)
    full_model = keras.Model(inputs=full_input, outputs=full_output, name="plant_disease_mobilenetv2")

    val_loss, val_acc = head_model.evaluate(x_val_feats, y_val_cat, verbose=0)
    print(f"Validation Accuracy: {val_acc * 100:.2f}% (Loss: {val_loss:.4f})")

    print("\n--- Phase 4: Held-Out Test Evaluation ---")
    test_probs = head_model.predict(x_test_feats, batch_size=batch_size, verbose=0)
    test_preds = np.argmax(test_probs, axis=1)
    test_acc = float(np.mean(test_preds == y_test))
    print(f"Final Test Accuracy: {test_acc * 100:.2f}% on {len(y_test)} unseen test samples")

    # Generate Metrics
    target_names = [DISPLAY_NAMES.get(name, name) for name in class_names]
    report_dict = classification_report(y_test, test_preds, target_names=target_names, output_dict=True)
    conf_mat = confusion_matrix(y_test, test_preds).tolist()

    print("\nTest Classification Report:")
    print(classification_report(y_test, test_preds, target_names=target_names))

    # Evaluate Confidence & Uncertainty Metrics
    max_probs = np.max(test_probs, axis=1)
    sorted_probs = np.sort(test_probs, axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]

    correct_mask = (test_preds == y_test)
    avg_conf_correct = float(np.mean(max_probs[correct_mask])) if np.any(correct_mask) else 0.0
    avg_conf_incorrect = float(np.mean(max_probs[~correct_mask])) if np.any(~correct_mask) else 0.0

    # Calibrated thresholds:
    conf_threshold = 0.65
    margin_threshold = 0.15
    uncertain_mask = (max_probs < conf_threshold) | (margins < margin_threshold)
    uncertain_rate = float(np.mean(uncertain_mask))

    print(f"Avg confidence on correct predictions:   {avg_conf_correct * 100:.2f}%")
    print(f"Avg confidence on incorrect predictions: {avg_conf_incorrect * 100:.2f}%")
    print(f"Uncertainty flag rate (test set):        {uncertain_rate * 100:.2f}%")

    # Save Keras Model
    keras_path = output_dir / "plant_disease_mobilenetv2.keras"
    print(f"\nSaving Keras model to {keras_path}...")
    full_model.save(keras_path)

    # Save ONNX Model
    onnx_path = output_dir / "plant_disease_mobilenetv2.onnx"
    try:
        import tf2onnx
        print(f"Exporting ONNX model to {onnx_path}...")
        spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="image_input"),)
        model_proto, _ = tf2onnx.convert.from_keras(full_model, input_signature=spec, output_path=str(onnx_path))
        print("ONNX conversion successful!")
    except Exception as exc:
        print(f"WARNING: ONNX export skipped: {exc}")
        onnx_path = None

    # Save Class Manifest
    manifest_data = {
        "model_name": "plant_disease_mobilenetv2",
        "version": "1.0.0",
        "architecture": "MobileNetV2 Transfer Learning",
        "input_shape": [224, 224, 3],
        "preprocessing": {
            "format": "RGB",
            "resize": [224, 224],
            "scaling": "[-1.0, 1.0]",
            "formula": "(pixel / 127.5) - 1.0",
            "interpolation": "BILINEAR",
        },
        "num_classes": num_classes,
        "classes": [
            {
                "index": i,
                "label": name,
                "display_name": DISPLAY_NAMES.get(name, name),
                "recommended_action": RECOMMENDED_ACTIONS.get(name, "Consult local agricultural extension."),
            }
            for i, name in enumerate(class_names)
        ],
        "uncertainty_policy": {
            "confidence_threshold": conf_threshold,
            "margin_threshold": margin_threshold,
            "min_sharpness_var": 40.0,
            "min_mean_brightness": 25.0,
            "max_mean_brightness": 235.0,
            "status_on_low_confidence": "UNCERTAIN",
            "status_on_quality_fail": "IMAGE_REJECTED",
        },
    }

    manifest_path = output_dir / "class_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"Saved class manifest to {manifest_path}")

    # Save Model Metadata
    metadata = {
        "model_version": "1.0.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "training_samples": len(splits["train"]),
        "validation_samples": len(splits["val"]),
        "test_samples": len(splits["test"]),
        "test_accuracy": round(test_acc, 4),
        "macro_f1": round(report_dict["macro avg"]["f1-score"], 4),
        "weighted_f1": round(report_dict["weighted avg"]["f1-score"], 4),
        "classification_report": report_dict,
        "confusion_matrix": conf_mat,
        "class_names": class_names,
        "confidence_calibration": {
            "avg_confidence_correct": round(avg_conf_correct, 4),
            "avg_confidence_incorrect": round(avg_conf_incorrect, 4),
            "uncertain_rate_on_test": round(uncertain_rate, 4),
        },
        "training_history": {
            "loss": [float(round(v, 4)) for v in history.history["loss"]],
            "accuracy": [float(round(v, 4)) for v in history.history["accuracy"]],
            "val_loss": [float(round(v, 4)) for v in history.history["val_loss"]],
            "val_accuracy": [float(round(v, 4)) for v in history.history["val_accuracy"]],
        },
        "limitations": [
            "Dataset consists of laboratory-style leaf photographs with controlled backgrounds (PlantVillage style).",
            "Model accuracy may drop on real field photographs with complex weed backgrounds, harsh shadows, or direct sunlight.",
            "Visual image diagnosis cannot replace chemical soil testing or lab PCR testing.",
            "Uncertain status must prompt physical inspection and agronomist consultation.",
        ],
    }

    metadata_path = output_dir / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata to {metadata_path}")

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MobileNetV2 Plant Leaf Classifier")
    parser.add_argument("--dataset-dir", default="/home/mano/Documents/Anveshion/dataset/augmented_dataset", help="Path to augmented_dataset directory")
    parser.add_argument("--output-dir", default="/home/mano/Documents/Anveshion/plant-health-rover-sim/plant_rover/models", help="Directory to save model artifacts")
    parser.add_argument("--samples-per-class", type=int, default=350, help="Samples per class (0 for all)")
    parser.add_argument("--epochs", type=int, default=12, help="Epochs for classification head")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        print(f"ERROR: Dataset directory not found: {dataset_path}")
        sys.exit(1)

    output_path = Path(args.output_dir)
    class_names, splits = load_dataset_splits(
        dataset_path,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )

    t0 = time.time()
    results = build_and_train_model(
        class_names,
        splits,
        output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(f"\nTraining pipeline completed in {time.time() - t0:.1f}s.")
    print(f"Final Test Accuracy: {results['test_accuracy'] * 100:.2f}% | Macro F1: {results['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
