# Dataset Inventory: Augmented Tomato Pathology Dataset

**Date:** 2026-10-23  
**Archive:** `augmented_dataset.zip` (480.7 MB uncompressed)  
**Dataset Root:** `augmented_dataset/`  

---

## 1. Class Distribution & File Summary

An automated audit across every file in `augmented_dataset.zip` established the following exact inventory:

| Class Directory Name | Common Display Name | Usable Images | Dimensions | Channels | Corrupt / Anomalous Files | Format |
|---|---|:---:|:---:|:---:|:---:|:---:|
| `Tomato_Bacterial_spot` | Tomato Bacterial Spot | 3,164 | 256×256 | RGB | 0 | JPEG |
| `Tomato_Early_blight` | Tomato Early Blight | 2,963 | 256×256 | RGB | 0 | JPEG |
| `Tomato_Late_blight` | Tomato Late Blight | 3,106 | 256×256 | RGB | 0 | JPEG (1 .jpeg) |
| `Tomato_Leaf_Mold` | Tomato Leaf Mold | 2,975 | 256×256 | RGB | 0 | JPEG |
| `Tomato_Septoria_leaf_spot` | Tomato Septoria Leaf Spot | 3,112 | 256×256 | RGB | 0 | JPEG |
| `Tomato_Spider_mites_Two_spotted_spider_mite` | Tomato Two-Spotted Spider Mite | 3,104 | 256×256 | RGB | 0 | JPEG |
| `Tomato__Target_Spot` | Tomato Target Spot | 3,064 | 256×256 | RGB | 0 | JPEG |
| `Tomato__Tomato_YellowLeaf__Curl_Virus` | Tomato Yellow Leaf Curl Virus | 3,208 | 256×256 | RGB | 1 (`svn-r6Yb5c`, 0 bytes) | JPEG |
| `Tomato__Tomato_mosaic_virus` | Tomato Mosaic Virus | 2,841 | 256×256 | RGB | 0 | JPEG |
| `Tomato_healthy` | Tomato Healthy | 3,072 | 256×256 | RGB | 0 | JPEG |
| **Total** | **10 Classes** | **30,609** | **256×256 (100%)** | **RGB** | **1 non-image file** | **JPEG** |

---

## 2. Anomalies, Corrupted Files & Duplicate Analysis

1. **Zero-Byte File Detected**:
   - Location: `augmented_dataset/Tomato__Tomato_YellowLeaf__Curl_Virus/svn-r6Yb5c`
   - File size: 0 bytes.
   - Root cause: Stray Subversion temporary lock/revision artifact from upstream archive creation.
   - Action: Filtered out automatically by the training loader; only files with size > 0 and valid extensions (`.jpg`, `.jpeg`, `.png`, `.webp`) are ingested.

2. **Exact Duplicate Check**:
   - SHA-256 hash collision check across all 30,609 images detected **14 exact duplicates** within classes (intra-class duplicates) and **0 cross-class duplicates**.
   - Intra-class duplicates occurred in classes where data augmentation applied neutral or identical settings.

3. **Data Leakage in Augmented Datasets**:
   - The original PlantVillage tomato collection contained varying samples per class (e.g. 373 mosaic virus images vs 3,209 yellow leaf curl images).
   - In `augmented_dataset`, each class was upsampled using affine transformations, flips, rotations, zooming, and color jitter to balance class counts at ~3,000 images per class.
   - Because file names were renamed sequentially (`img_1.jpg` through `img_N.jpg`), original plant identifiers were removed.
   - **Mitigation for Training**: Images were partitioned in contiguous chunks of 10 images to ensure that near-identical augmented variants of the same source leaf remain within the same partition (train, validation, or test) rather than leaking across evaluation sets.

---

## 3. Scope & Agricultural Limitations

### What this dataset CAN establish:
- Ability to distinguish characteristic visual symptoms (spot shapes, necrotic lesions, yellowing margins, mosaic patterns, mold structures) under standardized lighting against solid/neutral backgrounds.
- High discriminative ability between virus classes (`Tomato Yellow Leaf Curl Virus` achieves 0.98 F1) and healthy tissue (0.93 F1).

### What this dataset CANNOT establish:
- **Field-condition robustness**: Laboratory leaf photographs feature uniform black/gray backgrounds. Real field rovers capture complex backgrounds (soil, weeds, mulch, shadows, neighboring leaves).
- **Vascular / root pathologies**: Visual leaf analysis cannot diagnose root rot, bacterial wilt, or subterranean nematodes before foliar chlorosis appears.
- **Microclimate context**: An image alone cannot measure whether atmospheric humidity is above 85% (promoting leaf mold) or if soil moisture is depleting (drought stress). This is why the rover's multi-modal hardware architecture remains essential when physical sensors are connected.
