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

3. **Data Leakage and Specimen Independence in Augmented Datasets**:
   - The original PlantVillage tomato collection contained varying samples per class (e.g. 373 mosaic virus images vs 3,209 yellow leaf curl images).
   - In `augmented_dataset`, each class was upsampled using affine transformations, flips, rotations, zooming, and color jitter to balance class counts at ~3,000 images per class.
   - Because file names were renamed sequentially (`img_1.jpg` through `img_N.jpg`), original plant specimen identifiers and EXIF tags were discarded during upstream preprocessing.
   - **Auditor Evaluation of Partitioning**:
     * In `train_classifier.py`, image files were sorted lexicographically before grouping into contiguous chunks of 10.
     * Standard string sorting orders files as `img_1.jpg`, `img_10.jpg`, `img_100.jpg`, `img_1000.jpg` rather than natural numeric order (`img_1.jpg`, `img_2.jpg`, ...). Even if sequential numbers originally mapped to shared source specimens, lexicographical chunking scrambles that sequence.
     * More critically, because no source-image metadata exists in `augmented_dataset`, there is no evidence that adjacent files share parents or that separated files do not.
     * **Auditor Finding**: The current partition does **NOT** genuinely prevent augmented variants of the same physical leaf from crossing train, validation, and test splits.
     * **Evaluation Scope**: The reported test accuracy (83.33%) and macro F1 (83.34%) must therefore be treated as preliminary benchmark metrics on the existing synthetic partition, representing an upper-bound performance under potential intra-specimen correlation.
     * **Metadata Recovery**: Identifiable source-image metadata cannot be recovered from `augmented_dataset` alone because EXIF and original filenames were permanently stripped upstream.
     * **Independent Evaluation Protocol**: Rigorous validation of true out-of-distribution and real-world diagnostic performance requires an independent evaluation benchmark using external, botanically confirmed leaf photographs (such as university agricultural extension services, USDA ARS, EPPO, and fresh field captures with varied soil, weed, and lighting backgrounds).

---

## 3. Scope & Agricultural Limitations

### What this dataset CAN establish:
- Ability to distinguish characteristic visual symptoms (spot shapes, necrotic lesions, yellowing margins, mosaic patterns, mold structures) under standardized lighting against solid/neutral backgrounds.
- High discriminative ability between virus classes (`Tomato Yellow Leaf Curl Virus` achieves 0.98 F1) and healthy tissue (0.93 F1) on the synthetic partition.

### What this dataset CANNOT establish:
- **Specimen-independent generalization**: Because specimen IDs were lost, zero-leakage independence between train and test sets cannot be guaranteed on `augmented_dataset`.
- **Field-condition robustness**: Laboratory leaf photographs feature uniform black/gray backgrounds. Real field rovers capture complex backgrounds (soil, weeds, mulch, shadows, neighboring leaves) where visual accuracy drops.
- **Vascular / root pathologies**: Visual leaf analysis cannot diagnose root rot, bacterial wilt, or subterranean nematodes before foliar chlorosis appears.
- **Nutrient deficiencies and non-modeled pests**: Visual symptoms of magnesium deficiency (`pub0006537_76781.webp`) and leafminer tunneling (`leaf_miner_pest.jpg`) are outside the 10 trained classes and cause closed-set overconfidence failures unless detected as out-of-distribution.
- **Microclimate context**: An image alone cannot measure whether atmospheric humidity is above 85% (promoting leaf mold) or if soil moisture is depleting (drought stress). This is why the rover's multi-modal hardware architecture remains essential when physical sensors are connected.

---

## 4. Multi-Symmetry Hash Clustering and Leakage-Free Manifest (`DATASET_MANIFEST.json`)

To resolve the lexicographical chunking flaw and eliminate cross-partition variant leakage, the dataset was audited using 64-bit multi-symmetry perceptual hashing (8 planar symmetries) and SHA-256 collision mapping (`plant_rover/ml/dataset_audit.py`):

1. **Clustering Results:**
   - **Total Usable Images:** 30,609 across 10 classes
   - **Corrupted / Non-Image:** 1 (`svn-r6Yb5c`, 0 bytes)
   - **Exact Duplicate Groups:** 14 (28 files)
   - **Total Distinct Specimen Groups:** 30,196
   - **Multi-Variant Groups:** 344 clusters (containing up to 11 synthetic affine/flip variants)
   - **Near-Match Candidate Review Pairs:** Perceptual hash Hamming distance $\le 2$ candidate pairs are logged in `DATASET_AUDIT_REPORT.json` for manual agricultural review rather than collapsed automatically, preventing false-positive merging of distinct leaves.

2. **Group-Level Partitioning (`DATASET_MANIFEST.json`):**
   - **Train Split:** 21,421 images (69.98% of dataset, 21,137 groups)
   - **Validation Split:** 4,592 images (15.00% of dataset, 4,529 groups)
   - **Held-Out Test Split:** 4,596 images (15.02% of dataset, 4,530 groups)
   - **Group Leakage:** 0 groups cross split boundaries ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$).

3. **Isolation Scope & Verification Boundary:**
   - **Guaranteed:** Zero overlap among detected hash-defined groups and exact duplicate files across train, validation, and test partitions.
   - **Limitation:** Specimen-level biological independence cannot be proven from filenames or hash metrics alone because upstream preprocessing stripped physical plant tags and EXIF data. Real-world diagnostic reliability must be verified using genuinely independent field photographs.


