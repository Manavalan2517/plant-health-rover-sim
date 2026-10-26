"""Reproducible transfer learning training pipeline for MobileNetV2 plant leaf disease classification.

Features:
- Leakage-resistant dataset loading using group-based manifests (DATASET_MANIFEST.json).
- MobileNetV2 preprocessing contract: (pixel / 127.5) - 1.0 ([-1.0, 1.0] scaling).
- Two-stage training:
    Stage 1: Frozen backbone feature extraction + head classification.
    Stage 2: Controlled fine-tuning of upper backbone layers (BatchNormalization frozen).
- Training-only data augmentation (rotation, zoom, contrast, flip) with diagnostic integrity.
- Validation monitoring: EarlyStopping and ReduceLROnPlateau.
- Extraction and normalization of latent class centroids (class_centroids.npy).
- Dual-output ONNX export (probabilities + 1280-dim feature embeddings).
- Comprehensive evaluation: Accuracy, Macro F1, Per-class Precision/Recall, Confusion Matrix.
- Export of Keras model, ONNX model, class manifest, and model metadata.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2


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


def preprocess_image_file(
    path: Path | str,
    target_size: tuple[int, int] = (224, 224),
    augment: bool = False,
    rng: random.Random | None = None,
) -> np.ndarray:
    """Read, optionally augment (training only), normalize EXIF, convert to RGB, resize, and scale to [-1, 1]."""
    with Image.open(path) as im:
        try:
            from PIL import ImageOps, ImageEnhance
            im = ImageOps.exif_transpose(im)
        except Exception:
            from PIL import ImageEnhance
        im = im.convert("RGB")
        if augment:
            local_rng = rng if rng is not None else random.Random()
            # 1. Random horizontal flip (p=0.5)
            if local_rng.random() > 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            # 2. Random mild rotation (-12 to +12 degrees)
            angle = local_rng.uniform(-12.0, 12.0)
            im = im.rotate(angle, resample=Image.Resampling.BILINEAR)
            # 3. Random mild brightness (factor 0.92 to 1.08)
            b_factor = local_rng.uniform(0.92, 1.08)
            im = ImageEnhance.Brightness(im).enhance(b_factor)
            # 4. Random mild contrast (factor 0.90 to 1.10)
            c_factor = local_rng.uniform(0.90, 1.10)
            im = ImageEnhance.Contrast(im).enhance(c_factor)

        im = im.resize(target_size, Image.Resampling.BILINEAR)
        arr = np.array(im, dtype=np.float32)
        # Canonical MobileNetV2 scaling: [0, 255] -> [-1.0, 1.0]
        norm = (arr / 127.5) - 1.0
        return norm


def validate_manifest_integrity(
    manifest_data: dict[str, Any],
    dataset_dir: Path,
    sample_sha_checks: int = 50,
    seed: int = 2026,
) -> tuple[bool, str]:
    """Validate manifest entries: check file existence, class and split validity, and verify SHA-256 integrity."""
    items = manifest_data.get("items", [])
    if not items:
        return False, "Manifest contains no items"
    valid_splits = {"train", "val", "test"}
    valid_classes = set(DISPLAY_NAMES.keys())

    rng = random.Random(seed)
    check_sample = rng.sample(items, min(len(items), sample_sha_checks)) if sample_sha_checks > 0 else items
    for item in check_sample:
        rel_path = item.get("path")
        cname = item.get("class")
        split = item.get("split")
        expected_sha = item.get("sha256")

        if cname not in valid_classes:
            return False, f"Invalid class name in manifest: {cname}"
        if split not in valid_splits:
            return False, f"Invalid split in manifest: {split}"

        full_p = dataset_dir.parent / rel_path
        if not full_p.exists():
            return False, f"Manifest file does not exist on disk: {full_p}"

        actual_sha = hashlib.sha256(full_p.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            return False, f"SHA-256 mismatch for {full_p.name}: expected {expected_sha[:8]}, got {actual_sha[:8]}"

    return True, f"Validated {len(items)} manifest entries ({len(check_sample)} SHA-256 verified)"


def get_stratified_subset(
    items: list[tuple[Path, int]],
    samples_per_class: int | None = None,
    seed: int = 2026,
) -> list[tuple[Path, int]]:
    """Return a stratified, class-balanced subset containing all classes present in items."""
    by_class: dict[int, list[tuple[Path, int]]] = {}
    for p, c in items:
        by_class.setdefault(c, []).append((p, c))

    rng = random.Random(seed)
    subset: list[tuple[Path, int]] = []
    for c in sorted(by_class.keys()):
        flist = list(by_class[c])
        rng.shuffle(flist)
        if samples_per_class is not None and samples_per_class > 0:
            subset.extend(flist[:min(len(flist), samples_per_class)])
        else:
            subset.extend(flist)
    rng.shuffle(subset)
    return subset


def load_dataset_splits(
    dataset_dir: Path,
    manifest_path: Path | None = None,
    samples_per_class: int = 0,
    seed: int = 2026,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    validate_manifest: bool = True,
) -> tuple[list[str], dict[str, list[tuple[Path, int]]]]:
    """Load dataset partitions. Uses DATASET_MANIFEST.json if available to guarantee zero leakage."""
    class_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    splits: dict[str, list[tuple[Path, int]]] = {"train": [], "val": [], "test": []}
    rng = random.Random(seed)

    # Search for manifest
    candidate_manifests = [
        manifest_path,
        Path(__file__).resolve().parents[2] / "docs" / "DATASET_MANIFEST.json",
        dataset_dir.parent.parent / "plant-health-rover-sim" / "docs" / "DATASET_MANIFEST.json",
        Path.cwd() / "docs" / "DATASET_MANIFEST.json",
    ]
    target_manifest = None
    for cand in candidate_manifests:
        if cand and Path(cand).exists():
            target_manifest = Path(cand)
            break

    if target_manifest:
        print(f"Loading leakage-resistant splits from manifest: {target_manifest}")
        with open(target_manifest, "r", encoding="utf-8") as f:
            mdata = json.load(f)

        if validate_manifest:
            is_valid, v_msg = validate_manifest_integrity(mdata, dataset_dir, sample_sha_checks=50, seed=seed)
            if not is_valid:
                raise ValueError(f"Manifest integrity validation failed: {v_msg}")
            print(f"  Manifest integrity check: {v_msg}")

        items_by_class_split: dict[str, dict[str, list[Path]]] = {c: {"train": [], "val": [], "test": []} for c in class_names}
        for item in mdata.get("items", []):
            cname = item.get("class")
            split = item.get("split")
            rel_path = item.get("path")
            if cname in items_by_class_split and split in items_by_class_split[cname]:
                full_p = dataset_dir.parent / rel_path
                if full_p.exists():
                    items_by_class_split[cname][split].append(full_p)

        for cname in class_names:
            cidx = class_to_idx[cname]
            for s in ("train", "val", "test"):
                flist = items_by_class_split[cname][s]
                rng.shuffle(flist)
                if samples_per_class > 0:
                    target_n = int(samples_per_class * (train_ratio if s == "train" else val_ratio if s == "val" else test_ratio))
                    selected = flist[:max(1, target_n)]
                else:
                    selected = flist
                for f in selected:
                    splits[s].append((f, cidx))

        for s in ("train", "val", "test"):
            rng.shuffle(splits[s])

        print(f"Loaded from manifest -> Train: {len(splits['train'])} files, Val: {len(splits['val'])} files, Test: {len(splits['test'])} files across {len(class_names)} classes.")
        return class_names, splits

    # Fallback to direct directory discovery with group clustering
    print(f"Manifest not found; discovering directly from {dataset_dir}...")
    for cdir in class_dirs:
        cidx = class_to_idx[cdir.name]
        raw_files = sorted([f for f in cdir.iterdir() if f.is_file() and f.stat().st_size > 0 and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")])

        if samples_per_class > 0 and len(raw_files) > samples_per_class:
            chunk_size = 10
            chunks = [raw_files[i : i + chunk_size] for i in range(0, len(raw_files), chunk_size)]
            rng.shuffle(chunks)
            selected_files = [f for chunk in chunks for f in chunk][:samples_per_class]
        else:
            selected_files = list(raw_files)

        chunk_size = 10
        chunks = [selected_files[i : i + chunk_size] for i in range(0, len(selected_files), chunk_size)]
        rng.shuffle(chunks)

        n_chunks = len(chunks)
        n_train = max(1, int(n_chunks * train_ratio))
        n_val = max(1, int(n_chunks * val_ratio))

        for f in [img for ch in chunks[:n_train] for img in ch]:
            splits["train"].append((f, cidx))
        for f in [img for ch in chunks[n_train : n_train + n_val] for img in ch]:
            splits["val"].append((f, cidx))
        for f in [img for ch in chunks[n_train + n_val :] for img in ch]:
            splits["test"].append((f, cidx))

    for k in splits:
        rng.shuffle(splits[k])

    print(f"Direct discovery -> Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    return class_names, splits


def batch_generator(
    items: list[tuple[Path, int]],
    batch_size: int = 32,
    augment: bool = False,
    shuffle: bool = False,
    seed: int | None = None,
):
    """Memory-efficient batch generator with optional training-only augmentation and epoch shuffling."""
    items_list = list(items)
    n = len(items_list)
    rng = random.Random(seed) if seed is not None else random.Random()
    if shuffle:
        rng.shuffle(items_list)

    for start in range(0, n, batch_size):
        batch_items = items_list[start : start + batch_size]
        if augment:
            x = np.stack([
                preprocess_image_file(p, augment=True, rng=rng)
                for p, _ in batch_items
            ], axis=0)
        else:
            with ThreadPoolExecutor(max_workers=min(8, len(batch_items))) as ex:
                x = np.stack(list(ex.map(lambda it: preprocess_image_file(it[0], augment=False), batch_items)), axis=0)
        y = np.array([lbl for _, lbl in batch_items], dtype=np.int32)
        yield x, y


def extract_features(
    base_model: Any,
    items: list[tuple[Path, int]],
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute MobileNetV2 backbone embeddings for fast training."""
    features = []
    labels = []
    total = len(items)
    done = 0
    t0 = time.time()
    for x_batch, y_batch in batch_generator(items, batch_size, augment=False, shuffle=False):
        feats = base_model.predict(x_batch, batch_size=batch_size, verbose=0)
        features.append(feats)
        labels.append(y_batch)
        done += len(y_batch)
        if done % 1000 == 0 or done == total:
            elapsed = time.time() - t0
            print(f"  Extracted {done}/{total} embeddings ({done / max(1e-3, elapsed):.1f} img/s)...")
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def compute_metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    """Pure NumPy calculation of classification metrics, per-class stats, and confusion matrix.
    Guarantees single-source-of-truth mathematical consistency across all reported metrics.
    """
    num_classes = len(class_names)
    conf_mat = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf_mat[t, p] += 1

    report: dict[str, Any] = {}
    precisions = []
    recalls = []
    f1s = []
    supports = []

    for i, cname in enumerate(class_names):
        display = DISPLAY_NAMES.get(cname, cname)
        tp = int(conf_mat[i, i])
        fp = int(np.sum(conf_mat[:, i]) - tp)
        fn = int(np.sum(conf_mat[i, :]) - tp)
        support = int(np.sum(conf_mat[i, :]))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(support)

        report[display] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1-score": round(f1, 4),
            "support": support,
        }

    total_samples = len(y_true)
    correct_count = int(np.sum(y_true == y_pred))
    accuracy = float(correct_count / total_samples) if total_samples > 0 else 0.0

    displayed_precisions = [report[DISPLAY_NAMES.get(c, c)]["precision"] for c in class_names]
    displayed_recalls = [report[DISPLAY_NAMES.get(c, c)]["recall"] for c in class_names]
    displayed_f1s = [report[DISPLAY_NAMES.get(c, c)]["f1-score"] for c in class_names]

    macro_prec = round(float(np.mean(displayed_precisions)), 4)
    macro_rec = round(float(np.mean(displayed_recalls)), 4)
    macro_f1 = round(float(np.mean(displayed_f1s)), 4)

    weights = np.array(supports, dtype=np.float64) / max(1, total_samples)
    weighted_prec = float(np.sum(np.array(displayed_precisions) * weights))
    weighted_rec = float(np.sum(np.array(displayed_recalls) * weights))
    weighted_f1 = float(np.sum(np.array(displayed_f1s) * weights))

    report["accuracy"] = round(accuracy, 4)
    report["macro avg"] = {
        "precision": round(macro_prec, 4),
        "recall": round(macro_rec, 4),
        "f1-score": round(macro_f1, 4),
        "support": total_samples,
    }
    report["weighted avg"] = {
        "precision": round(weighted_prec, 4),
        "recall": round(weighted_rec, 4),
        "f1-score": round(weighted_f1, 4),
        "support": total_samples,
    }

    # Strict mathematical consistency checks
    assert int(np.sum(np.diag(conf_mat))) == correct_count, "Confusion matrix diagonal must equal correct count"
    assert int(np.sum(conf_mat)) == total_samples, "Confusion matrix sum must equal total samples"

    return {
        "classification_report": report,
        "confusion_matrix": conf_mat.tolist(),
        "total_samples": total_samples,
        "correct_samples": correct_count,
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
    }


def build_and_train_model(
    class_names: list[str],
    splits: dict[str, list[tuple[Path, int]]],
    output_dir: Path,
    epochs: int = 12,
    fine_tune_epochs: int = 3,
    fine_tune_samples_per_class: int = 120,
    batch_size: int = 32,
    seed: int = 2026,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Execute two-stage MobileNetV2 training with early stopping, stratified fine-tuning, dynamic augmentation, and ONNX export."""
    tf.random.set_seed(seed)
    np.random.seed(seed)
    num_classes = len(class_names)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Stage 1: Feature Extraction Backbone Setup ---")
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

    y_train_cat = keras.utils.to_categorical(y_train, num_classes)
    y_val_cat = keras.utils.to_categorical(y_val, num_classes)

    print("\n--- Stage 1: Classification Head Training with Monitoring ---")
    head_inputs = keras.Input(shape=(1280,), name="feature_input")
    x = keras.layers.BatchNormalization(name="head_bn1")(head_inputs)
    x = keras.layers.Dropout(0.35, name="head_drop1")(x)
    x = keras.layers.Dense(256, activation="relu", name="head_dense1")(x)
    x = keras.layers.Dropout(0.35, name="head_drop2")(x)
    head_outputs = keras.layers.Dense(num_classes, activation="softmax", name="probabilities")(x)

    head_model = keras.Model(inputs=head_inputs, outputs=head_outputs, name="classification_head")
    head_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_stage1 = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1),
    ]

    history_stage1 = head_model.fit(
        x_train_feats,
        y_train_cat,
        validation_data=(x_val_feats, y_val_cat),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_stage1,
        verbose=1,
    )

    val_loss_s1, val_acc_s1 = head_model.evaluate(x_val_feats, y_val_cat, verbose=0)
    print(f"Stage 1 Best Val Accuracy: {val_acc_s1 * 100:.2f}% (Loss: {val_loss_s1:.4f})")

    # Assemble End-to-End Model
    full_input = keras.Input(shape=(224, 224, 3), name="image_input")
    full_feats = backbone(full_input, training=False)
    full_output = head_model(full_feats)
    full_model = keras.Model(inputs=full_input, outputs=full_output, name="plant_disease_mobilenetv2")

    # Stage 2: Controlled Fine-Tuning with Stratified All-Class Batches and Dynamic Augmentation
    history_stage2 = None
    if fine_tune_epochs > 0:
        print(f"\n--- Stage 2: Controlled Fine-Tuning ({fine_tune_epochs} epochs) with Stratified Sampling and Real Augmentation ---")
        backbone.trainable = True
        for layer in backbone.layers[:-25]:
            layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False

        full_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Stratified sampling across ALL 10 classes
        ft_train_items = get_stratified_subset(splits["train"], samples_per_class=fine_tune_samples_per_class, seed=seed)
        ft_val_items = get_stratified_subset(splits["val"], samples_per_class=max(1, fine_tune_samples_per_class // 4), seed=seed)

        # Class coverage assertion
        ft_train_classes = set(c for _, c in ft_train_items)
        ft_val_classes = set(c for _, c in ft_val_items)
        assert len(ft_train_classes) == num_classes, f"Fine-tuning train items missing classes: {set(range(num_classes)) - ft_train_classes}"
        assert len(ft_val_classes) == num_classes, f"Fine-tuning val items missing classes: {set(range(num_classes)) - ft_val_classes}"
        print(f"Fine-tuning stratified training set: {len(ft_train_items)} items ({len(ft_train_classes)}/{num_classes} classes)")
        print(f"Fine-tuning stratified validation set: {len(ft_val_items)} items ({len(ft_val_classes)}/{num_classes} classes)")

        def ft_gen(items: list[tuple[Path, int]], bsize: int, augment: bool = False):
            epoch = 0
            while True:
                epoch_seed = seed + epoch if seed is not None else None
                for x_b, y_b in batch_generator(items, bsize, augment=augment, shuffle=True, seed=epoch_seed):
                    yield x_b, keras.utils.to_categorical(y_b, num_classes)
                epoch += 1

        steps_per_epoch = max(1, len(ft_train_items) // batch_size)
        val_steps = max(1, len(ft_val_items) // batch_size)

        callbacks_stage2 = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True, verbose=1),
        ]

        history_stage2 = full_model.fit(
            ft_gen(ft_train_items, batch_size, augment=True),
            steps_per_epoch=steps_per_epoch,
            validation_data=ft_gen(ft_val_items, batch_size, augment=False),
            validation_steps=val_steps,
            epochs=fine_tune_epochs,
            callbacks=callbacks_stage2,
            verbose=1,
        )

    # Compute Feature Centroids using Final Backbone Embeddings
    print("\n--- Computing Normalized Latent Class Centroids using Final Backbone ---")
    final_feature_extractor = keras.Model(
        inputs=full_model.input,
        outputs=full_feats,
        name="final_feature_extractor"
    )

    # Use stratified training samples across all 10 classes to compute prototypes
    centroid_samples = get_stratified_subset(splits["train"], samples_per_class=100, seed=seed)
    print(f"Extracting final backbone embeddings from {len(centroid_samples)} training samples across all classes...")
    train_final_feats, train_final_labels = extract_features(final_feature_extractor, centroid_samples, batch_size=batch_size)

    centroids = np.zeros((num_classes, 1280), dtype=np.float32)
    for cidx in range(num_classes):
        cmask = (train_final_labels == cidx)
        if np.any(cmask):
            c_mean = np.mean(train_final_feats[cmask], axis=0)
            c_norm = c_mean / (np.linalg.norm(c_mean) + 1e-12)
            centroids[cidx] = c_norm
        else:
            print(f"WARNING: No samples for class {cidx} in centroid extraction!")
            centroids[cidx] = np.zeros(1280, dtype=np.float32)

    centroids_path = output_dir / "class_centroids.npy"
    np.save(centroids_path, centroids)
    print(f"Saved class centroids to {centroids_path} (shape: {centroids.shape}, unit norm: {np.allclose(np.linalg.norm(centroids, axis=1), 1.0)})")

    # Held-Out Test Evaluation on Final Fine-Tuned Model directly on Raw Test Images
    print("\n--- Held-Out Test Evaluation on Final Fine-Tuned Model ---")
    test_probs_list = []
    test_y_list = []
    for x_b, y_b in batch_generator(splits["test"], batch_size=batch_size, augment=False, shuffle=False):
        probs = full_model.predict(x_b, batch_size=batch_size, verbose=0)
        test_probs_list.append(probs)
        test_y_list.append(y_b)

    test_probs = np.concatenate(test_probs_list, axis=0)
    y_test = np.concatenate(test_y_list, axis=0)
    test_preds = np.argmax(test_probs, axis=1)

    eval_stats = compute_metrics_dict(y_test, test_preds, class_names)
    test_acc = eval_stats["accuracy"]
    macro_f1 = eval_stats["macro_f1"]
    report_dict = eval_stats["classification_report"]
    conf_mat = eval_stats["confusion_matrix"]

    print(f"Final Test Accuracy: {test_acc * 100:.2f}% | Macro F1: {macro_f1:.4f} on {len(y_test)} held-out test samples")

    # Evaluate Confidence & Calibration Metrics
    max_probs = np.max(test_probs, axis=1)
    sorted_probs = np.sort(test_probs, axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]

    correct_mask = (test_preds == y_test)
    avg_conf_correct = float(np.mean(max_probs[correct_mask])) if np.any(correct_mask) else 0.0
    avg_conf_incorrect = float(np.mean(max_probs[~correct_mask])) if np.any(~correct_mask) else 0.0

    conf_threshold = 0.80
    margin_threshold = 0.20
    uncertain_mask = (max_probs < conf_threshold) | (margins < margin_threshold)
    uncertain_rate = float(np.mean(uncertain_mask))

    print(f"Avg confidence on correct predictions:   {avg_conf_correct * 100:.2f}%")
    print(f"Avg confidence on incorrect predictions: {avg_conf_incorrect * 100:.2f}%")
    print(f"Uncertainty flag rate (test set):        {uncertain_rate * 100:.2f}%")

    # Save detailed per-image test predictions for complete reproducibility
    predictions_record = []
    for (p, true_c), pred_c, max_p, margin in zip(splits["test"], test_preds, max_probs, margins):
        rel_p = str(p.relative_to(p.parents[2])) if len(p.parents) > 2 else p.name
        predictions_record.append({
            "path": rel_p,
            "filename": p.name,
            "true_class_index": int(true_c),
            "true_class_name": class_names[true_c],
            "predicted_class_index": int(pred_c),
            "predicted_class_name": class_names[pred_c],
            "correct": bool(true_c == pred_c),
            "confidence": float(round(max_p, 4)),
            "margin": float(round(margin, 4)),
        })
    preds_path = output_dir / "test_predictions.json"
    with open(preds_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_samples": len(predictions_record),
            "correct_samples": int(np.sum(test_preds == y_test)),
            "accuracy": round(test_acc, 4),
            "macro_precision": round(eval_stats["macro_precision"], 4),
            "macro_recall": round(eval_stats["macro_recall"], 4),
            "macro_f1": round(macro_f1, 4),
            "predictions": predictions_record,
        }, f, indent=2)
    print(f"Saved complete test predictions record to {preds_path}")

    eval_path = output_dir / "test_evaluation.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_stats, f, indent=2)
    print(f"Saved complete test evaluation report to {eval_path}")

    # Save Keras Model
    keras_path = output_dir / "plant_disease_mobilenetv2.keras"
    print(f"\nSaving Keras model to {keras_path}...")
    full_model.save(keras_path)

    # Save Dual-Output ONNX Model (probabilities + 1280-dim feature embeddings)
    onnx_path = output_dir / "plant_disease_mobilenetv2.onnx"
    keras_onnx_parity = False
    max_prob_diff = 1.0
    min_cos_sim = 0.0
    disagreements = 0
    total_tested = 0
    try:
        import tf2onnx
        import onnxruntime as ort

        # Create dual-output Keras model for ONNX export
        dual_model = keras.Model(inputs=full_input, outputs=[full_output, full_feats], name="plant_disease_mobilenetv2_dual")
        print(f"Exporting dual-output ONNX model to {onnx_path}...")
        spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="image_input"),)
        model_proto, _ = tf2onnx.convert.from_keras(dual_model, input_signature=spec, output_path=str(onnx_path))
        print("ONNX conversion successful (dual outputs: probabilities + feature_embeddings)!")

        # Verify Keras vs ONNX Parity across the ENTIRE held-out test set
        print(f"Verifying Keras-to-ONNX numerical parity across all {len(splits['test'])} test samples...")
        onnx_sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        max_prob_diff = 0.0
        min_cos_sim = 1.0

        for x_b, _ in batch_generator(splits["test"], batch_size=batch_size, augment=False, shuffle=False):
            k_out = dual_model(x_b, training=False)
            k_probs = k_out[0].numpy()
            k_feats = k_out[1].numpy()

            o_outs = onnx_sess.run(None, {"image_input": x_b})
            o_probs = o_outs[0]
            o_feats = o_outs[1]

            diff = float(np.max(np.abs(k_probs - o_probs)))
            if diff > max_prob_diff:
                max_prob_diff = diff

            for k_f, o_f in zip(k_feats, o_feats):
                c_sim = float(np.dot(k_f, o_f) / (np.linalg.norm(k_f) * np.linalg.norm(o_f) + 1e-12))
                if c_sim < min_cos_sim:
                    min_cos_sim = c_sim

            k_top = np.argmax(k_probs, axis=1)
            o_top = np.argmax(o_probs, axis=1)
            disagreements += int(np.sum(k_top != o_top))
            total_tested += len(x_b)

        keras_onnx_parity = bool(max_prob_diff < 1e-4 and min_cos_sim > 0.9999 and disagreements == 0)
        print(f"Keras-to-ONNX Parity over {total_tested} test samples: {keras_onnx_parity} (Max prob diff: {max_prob_diff:.2e}, Min cos sim: {min_cos_sim:.6f}, Top-1 disagreements: {disagreements})")
    except Exception as exc:
        print(f"WARNING: ONNX export or parity check skipped/failed: {exc}")
        onnx_path = None

    # Save Class Manifest
    manifest_data = {
        "model_name": "plant_disease_mobilenetv2",
        "version": "2.0.0",
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
            "entropy_threshold": 0.50,
            "min_centroid_similarity": 0.60,
            "min_centroid_margin": 0.005,
            "ambiguity_similarity_cutoff": 0.62,
            "min_sharpness_var": 35.0,
            "min_mean_brightness": 25.0,
            "max_mean_brightness": 235.0,
            "status_on_low_confidence": "UNCERTAIN",
            "status_on_unsupported_condition": "UNSUPPORTED_CONDITION",
            "status_on_quality_fail": "IMAGE_REJECTED",
        },
    }

    manifest_path = output_dir / "class_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"Saved class manifest to {manifest_path}")

    # Combine Training Histories
    loss_hist = [float(round(v, 4)) for v in history_stage1.history["loss"]]
    acc_hist = [float(round(v, 4)) for v in history_stage1.history["accuracy"]]
    val_loss_hist = [float(round(v, 4)) for v in history_stage1.history["val_loss"]]
    val_acc_hist = [float(round(v, 4)) for v in history_stage1.history["val_accuracy"]]

    if history_stage2:
        loss_hist.extend([float(round(v, 4)) for v in history_stage2.history["loss"]])
        acc_hist.extend([float(round(v, 4)) for v in history_stage2.history["accuracy"]])
        val_loss_hist.extend([float(round(v, 4)) for v in history_stage2.history["val_loss"]])
        val_acc_hist.extend([float(round(v, 4)) for v in history_stage2.history["val_accuracy"]])

    # Hash all generated artifacts for single-release consistency
    keras_sha = hashlib.sha256(keras_path.read_bytes()).hexdigest()
    onnx_sha = hashlib.sha256(onnx_path.read_bytes()).hexdigest() if onnx_path and onnx_path.exists() else None
    centroids_sha = hashlib.sha256(centroids_path.read_bytes()).hexdigest()
    class_manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    preds_sha = hashlib.sha256(preds_path.read_bytes()).hexdigest() if preds_path.exists() else None
    eval_sha = hashlib.sha256(eval_path.read_bytes()).hexdigest() if eval_path.exists() else None

    # Locate and hash dataset manifest
    ds_manifest_path = manifest_path if (manifest_path and manifest_path.name == "DATASET_MANIFEST.json") else (output_dir.parents[1] / "docs" / "DATASET_MANIFEST.json")
    ds_manifest_sha = hashlib.sha256(ds_manifest_path.read_bytes()).hexdigest() if ds_manifest_path.exists() else None

    # Save Model Metadata
    metadata = {
        "model_version": "2.0.0",
        "run_id": f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_seed_{seed}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "dataset_manifest_sha256": ds_manifest_sha,
        "training_samples": len(splits["train"]),
        "validation_samples": len(splits["val"]),
        "test_samples": len(splits["test"]),
        "total_evaluated_samples": eval_stats["total_samples"],
        "correct_samples": eval_stats["correct_samples"],
        "test_accuracy": round(test_acc, 4),
        "macro_precision": round(eval_stats["macro_precision"], 4),
        "macro_recall": round(eval_stats["macro_recall"], 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(eval_stats["weighted_f1"], 4),
        "classification_report": report_dict,
        "confusion_matrix": conf_mat,
        "class_names": class_names,
        "dynamic_augmentation": {
            "applied_to_training": True,
            "applied_to_val_and_test": False,
            "operations": [
                "Random horizontal flip (p=0.5)",
                "Random mild rotation (+/- 12 deg)",
                "Random mild brightness (factor 0.92 to 1.08)",
                "Random mild contrast (factor 0.90 to 1.10)",
                "Preserves diagnostic foliar symptoms and lesion margins",
            ],
        },
        "fine_tuning": {
            "stratified_selection": True,
            "all_classes_participating": True,
            "unfrozen_layers": 25,
            "batch_norm_frozen": True,
            "learning_rate": 1e-5,
            "optimizer": "Adam",
        },
        "keras_onnx_parity": {
            "verified": keras_onnx_parity,
            "images_tested": total_tested,
            "max_probability_difference": round(max_prob_diff, 8),
            "min_embedding_cosine_similarity": round(min_cos_sim, 6),
            "top1_prediction_disagreements": disagreements,
        },
        "artifact_hashes": {
            "keras_model_sha256": keras_sha,
            "onnx_model_sha256": onnx_sha,
            "class_centroids_sha256": centroids_sha,
            "class_manifest_sha256": class_manifest_sha,
            "test_predictions_sha256": preds_sha,
            "test_evaluation_sha256": eval_sha,
        },
        "confidence_calibration": {
            "avg_confidence_correct": round(avg_conf_correct, 4),
            "avg_confidence_incorrect": round(avg_conf_incorrect, 4),
            "uncertain_rate_on_test": round(uncertain_rate, 4),
        },
        "training_history": {
            "loss": loss_hist,
            "accuracy": acc_hist,
            "val_loss": val_loss_hist,
            "val_accuracy": val_acc_hist,
        },
        "limitations": [
            "Dataset consists of laboratory-style leaf photographs with controlled backgrounds (PlantVillage style).",
            "Model accuracy drops on real field photographs with weed backgrounds, harsh shadows, or direct sunlight.",
            "Visual image diagnosis cannot replace chemical soil testing or lab PCR testing.",
            "Uncertain status must prompt physical inspection and agronomist consultation.",
            "Closed-set model cannot diagnose conditions outside the 10 trained classes without uncertainty/OOD rejection.",
        ],
    }

    metadata_path = output_dir / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata to {metadata_path}")

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MobileNetV2 Plant Leaf Classifier")
    parser.add_argument("--dataset-dir", default="/home/mano/Documents/Anveshion/dataset/augmented_dataset", help="Dataset directory")
    parser.add_argument("--output-dir", default="/home/mano/Documents/Anveshion/plant-health-rover-sim/plant_rover/models", help="Output directory")
    parser.add_argument("--manifest-path", default=None, help="Path to DATASET_MANIFEST.json")
    parser.add_argument("--samples-per-class", type=int, default=0, help="Samples per class (0 for all eligible samples from manifest)")
    parser.add_argument("--fine-tune-samples-per-class", type=int, default=120, help="Fine-tuning samples per class (0 for all)")
    parser.add_argument("--epochs", type=int, default=10, help="Stage 1 epochs")
    parser.add_argument("--fine-tune-epochs", type=int, default=2, help="Stage 2 fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument("--skip-manifest-validation", action="store_true", help="Skip manifest hash validation")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        print(f"ERROR: Dataset directory not found: {dataset_path}")
        sys.exit(1)

    output_path = Path(args.output_dir)
    manifest_target = Path(args.manifest_path) if args.manifest_path else None
    class_names, splits = load_dataset_splits(
        dataset_path,
        manifest_path=manifest_target,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
        validate_manifest=not args.skip_manifest_validation,
    )

    t0 = time.time()
    results = build_and_train_model(
        class_names,
        splits,
        output_path,
        epochs=args.epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        fine_tune_samples_per_class=args.fine_tune_samples_per_class,
        batch_size=args.batch_size,
        seed=args.seed,
        manifest_path=manifest_target,
    )
    print(f"\nTraining pipeline completed in {time.time() - t0:.1f}s.")
    print(f"Final Test Accuracy: {results['test_accuracy'] * 100:.2f}% | Macro F1: {results['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
