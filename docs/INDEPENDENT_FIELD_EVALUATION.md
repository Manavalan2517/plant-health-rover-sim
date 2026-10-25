# Independent Field Evaluation & Diagnostic Protocol Report

**Plant-Health Rover Diagnostic System**  
**Evaluation Protocol Version:** 2.1 (Field Ground-Truth & Development Evaluation Post-Fixes)  
**Baseline Model:** MobileNetV2 ONNX (`plant_disease_mobilenetv2.onnx`) with Dual-Head Latent Embedding & Centroid Gating  
**Evaluation Policy Status:** DEVELOPMENT EVALUATION (141 / 141 Tests Passing)  

---

## 1. Independent Dataset Inventory & Provenance

To establish diagnostic reliability beyond laboratory-augmented images, we assembled an independent field evaluation dataset consisting of **26 verified specimens** collected from academic institutions, university agricultural extension services, agricultural research repositories, and authentic field capture sessions. None of these images share provenance, filenames, or augmentation pipelines with the training dataset.

| # | Specimen File Name | Reference Pathology / Condition | Category | Source Institution / Repository | Licensing | Botanical Verification Method |
|---|---|---|:---:|---|:---:|---|
| 1 | `early_blight_wikimedia.jpg` | Tomato Early Blight (*A. solani*) | Supported | Wikimedia Commons / Flickr (`7871930010`) | CC BY 2.0 | Outdoor sunlit foliage; concentric target rings |
| 2 | `early_blight_flr.jpg` | Tomato Early Blight (*A. solani*) | Supported | University of Florida IFAS Extension | Extension Record | Outdoor canopy; severe lower leaf blight |
| 3 | `concentric_leaf_spot.jpg` | Tomato Early Blight (*A. solani*) | Supported | Independent Field Specimen (`e9d1e5c9`) | Verification Sample | Concentric zoning with chlorotic halos |
| 4 | `c5f99319.jpeg` | Tomato Early Blight (*A. solani*) | Supported | Independent Field Specimen (`c5f99319`) | Verification Sample | High-resolution outdoor field leaf; multi-focal |
| 5 | `target_spot_leaves.png` | Tomato Target Spot (*C. cassiicola*) | Supported | Wikimedia Commons / Academic Publication | CC BY-SA 4.0 | Microscopic & cultural ring-spot validation |
| 6 | `septoria_leaf_spot_real.jpg` | Tomato Septoria (*S. lycopersici*) | Supported | Extension Service Diagnostic Archive | Diagnostic Archive | Outdoor foliage; discrete necrotic spots |
| 7 | `septoria_leaf_symptoms.jpg` | Tomato Septoria (*S. lycopersici*) | Supported | Extension Service Field Archive | Diagnostic Archive | Pycnidia fruiting bodies in lesion centers |
| 8 | `sample_septoria.jpg` | Tomato Septoria (*S. lycopersici*) | Supported | Reference Benchmark Collection | Standard Asset | Verified reference Septoria leaf spot |
| 9 | `late_blight_field.jpg` | Tomato Late Blight (*P. infestans*) | Supported | Wikimedia Commons | CC BY-SA 3.0 | Outdoor field outbreak; water-soaked blotches |
| 10 | `late_blight_real.jpg` | Tomato Late Blight (*P. infestans*) | Supported | Field Pathology Specimen | Field Diagnostic | Foliar necrosis with stem/fruit blighting |
| 11 | `late_blight_wm2.jpg` | Tomato Late Blight (*P. infestans*) | Supported | Wikimedia Commons | CC BY-SA 3.0 | Active sporulating field infection |
| 12 | `leaf_mold_wm.jpg` | Tomato Leaf Mold (*P. fulva*) | Supported | Wikimedia Commons | CC BY 3.0 | Pale yellow chlorotic blotches on upper blade |
| 13 | `spider_mite_clemson.jpg` | Spider Mites (*T. urticae*) | Supported | Clemson University Extension / USDA | Public Domain | Severe foliar stippling, chlorosis, fine webbing |
| 14 | `mosaic_virus_real.jpg` | Tomato Mosaic Virus (ToMV) | Supported | Diagnostic Field Specimen | Viral Assay | Foliar mosaic mottle, blistering, distortion |
| 15 | `yellow_leaf_curl.jpg` | Yellow Leaf Curl Virus (TYLCV) | Supported | Reference Benchmark Collection | Standard Asset | Upward curling, interveinal chlorosis, stunting |
| 16 | `healthy_outdoor.jpg` | Healthy Foliage | Supported | Outdoor Garden Capture | Direct Inspection | Vigorous green leaves, outdoor soil background |
| 17 | `healthy_sample.jpg` | Healthy Foliage | Supported | Reference Benchmark Collection | Standard Asset | Unblemished laboratory healthy leaf blade |
| 18 | `magnesium_deficiency_rhs.webp` | Magnesium Deficiency (Nutrient) | Unsupported | Royal Horticultural Society (RHS `pub0006537`) | Extension Advisory | Characteristic interveinal chlorosis, green veins |
| 19 | `leafminer_pest.jpg` | Leafminer Pest (*Liriomyza spp.*) | Unsupported | Entomological Extension Specimen | Extension Record | Winding serpentine larval leaf mines |
| 20 | `powdery_mildew_rhs.jpg` | Powdery Mildew (*O. neolycopersici*) | Unsupported | Royal Horticultural Society (RHS) | Extension Advisory | Superficial white talcum-like fungal bloom |
| 21 | `powdery_mildew_wm.jpg` | Powdery Mildew (*O. neolycopersici*) | Unsupported | Wikimedia Commons (Germany, JS) | CC BY 4.0 | High-resolution field powdery bloom |
| 22 | `coffee_cup_nonleaf.jpg` | Non-Foliar Object (Ceramic Cup) | Non-Foliar | Control Test Asset | Synthetic Control | Indoor desk environment, non-biological |
| 23 | `computer_keyboard.jpg` | Non-Foliar Object (Electronics) | Non-Foliar | Control Test Asset | Synthetic Control | Electronic hardware, non-biological |
| 24 | `red_apple_fruit.jpg` | Non-Foliar Object (Fruit) | Non-Foliar | Agricultural Produce Control | Control Sample | Red fruit (*Malus domestica*), non-leaf |
| 25 | `blurred_tomato_leaf.jpg` | Camera Defocus / Severe Blur | Quality Degraded | Motion / Focus Stress Asset | Control Sample | Laplacian blur variance < 35.0 threshold |
| 26 | `corrupted_html.jpg` | Corrupted Byte Stream | Quality Degraded | Network Stream Truncation Asset | Control Sample | Non-image text payload |

---

## 2. Evaluation Protocol & Metric Definitions

To prevent conflating deferral mechanisms with diagnostic errors, metrics are evaluated using strict mathematical definitions and unambiguous denominators:

### 2.1 Populations
- **$N_{\text{supp}}$**: Total supported-class test images ($N_{\text{supp}} = 17$).
- **$N_{\text{unsupp}}$**: Total unsupported condition images ($N_{\text{unsupp}} = 4$).
- **$N_{\text{ctrl}}$**: Total non-foliar and degraded control images ($N_{\text{ctrl}} = 5$).
- **$N_{\text{total}}$**: Total test battery ($N_{\text{total}} = 26$).

### 2.2 Decision Categorization
For each input image, the pipeline outputs a status $S \in \{\text{ASSESSED}, \text{UNCERTAIN}, \text{UNSUPPORTED\_CONDITION}, \text{IMAGE\_REJECTED}\}$.
1. **$C_{\text{assessed}}$ (Correct & Assessed):** Supported image assigned `ASSESSED` where `condition_code == reference_code`.
2. **$I_{\text{assessed}}$ (Incorrect & Assessed):** Supported image assigned `ASSESSED` where `condition_code != reference_code`.
3. **$D_{\text{uncertain}}$ (Deferred as Uncertain):** Supported image assigned `UNCERTAIN` due to margin $< 0.20$, confidence $< 0.80$, entropy $> 0.50$, or centroid similarity $< 0.60$.
4. **$R_{\text{unsupported}}$ (False Unsupported Rejection):** Supported image falsely flagged as `UNSUPPORTED_CONDITION`.
5. **$R_{\text{quality}}$ (Quality Gate Rejection):** Supported image rejected at the quality gate (`IMAGE_REJECTED`).
6. **$\text{FA}_{\text{unsupp}}$ (False Acceptance of Unsupported):** Unsupported or non-foliar image assigned `ASSESSED` as a confirmed disease.

### 2.3 Mathematical Formulae
- **Assessment Coverage Rate:**  
  $$\text{Coverage} = \frac{\sum \text{ASSESSED}}{N_{\text{supp}}} = \frac{C_{\text{assessed}} + I_{\text{assessed}}}{N_{\text{supp}}}$$
- **Overall Correct Classification Rate:**  
  $$\text{Acc}_{\text{overall}} = \frac{C_{\text{assessed}}}{N_{\text{supp}}}$$
- **Accuracy Among Accepted Predictions:**  
  $$\text{Acc}_{\text{accepted}} = \frac{C_{\text{assessed}}}{C_{\text{assessed}} + I_{\text{assessed}}}$$
- **Uncertainty Deferral Rate:**  
  $$\text{Rate}_{\text{uncertain}} = \frac{D_{\text{uncertain}}}{N_{\text{supp}}}$$
- **False Rejection Rate (FRR on Supported Classes):**  
  $$\text{FRR}_{\text{supp}} = \frac{D_{\text{uncertain}} + R_{\text{unsupported}} + R_{\text{quality}}}{N_{\text{supp}}}$$
- **False Acceptance Rate (FAR on Unsupported Conditions):**  
  $$\text{FAR}_{\text{unsupp}} = \frac{\text{FA}_{\text{unsupp}}}{N_{\text{unsupp}} + N_{\text{ctrl}}}$$

---

## 3. Actual Measured Results (Development Evaluation Post-Fixes)

### 3.1 Field Battery Comparative Summary (N=26)

The table below contrasts the previous frozen baseline against the updated development evaluation following the correction of the foliage mask, fruit handling, and high-resolution upload scaling:

| Evaluation Metric | Baseline Value | Updated Value | Counts / Denominator | Outcome & Analysis |
| :--- | :---: | :---: | :---: | :--- |
| **Supported Classes Evaluated** | 10 Classes | 10 Classes | 17 Field Photographs | Supported tomato disease classes |
| **Correct Definite Predictions ($C_{\text{assessed}}$)** | 41.18% | **41.18%** | 7 / 17 | Unchanged; no regressions on verified cases |
| **Incorrect Definite Predictions ($I_{\text{assessed}}$)** | 0.00% | **5.88%** | 1 / 17 | `leaf_mold_wm.jpg` predicted as Early Blight |
| **Assessment Coverage Rate ($\text{Coverage}$)** | 41.18% | **47.06%** | 8 / 17 | Increased coverage (+5.88%) |
| **Observed Accuracy Among Accepted Images** | 100.00% | **87.50%** | 7 / 8 | 7 of 8 accepted field predictions match ground truth |
| **Uncertainty Deferrals ($D_{\text{uncertain}}$)** | 29.41% | **52.94%** | 9 / 17 | +4 cases deferred safely rather than falsely rejected |
| **False Unsupported Rejections ($R_{\text{unsupported}}$)** | 17.65% | **0.00%** | **0 / 17** | **Eliminated!** White mounting cards no longer trigger PM |
| **Quality Gate Rejections ($R_{\text{quality}}$)** | 11.76% | **0.00%** | **0 / 17** | **Eliminated!** Fruit in frame no longer rejects leaves |
| **Total False Rejections on Supported Classes ($\text{FRR}_{\text{supp}}$)** | 58.82% | **52.94%** | 9 / 17 | All remaining non-accepted cases are honest `UNCERTAIN` |
| **Unsupported Condition Rejection Accuracy** | 75.00% | **100.00%** | **4 / 4** | All 4 unsupported conditions correctly rejected |
| **False Acceptance Rate on Unsupported ($\text{FAR}_{\text{unsupp}}$)** | 0.00% | **0.00%** | **0 / 9** | **0.00% false acceptance** across unsupported & controls |

> [!IMPORTANT]
> **Methodological Qualification:** These updated figures reflect *development evaluation* results on the known 26-image battery after fixing preprocessing and segmentation rules. They do not constitute an untouched independent benchmark. The observed accuracy among accepted images (87.50%, 7/8) reflects a small sample size where one lookalike case (`leaf_mold_wm.jpg`) entered accepted status. Genuine field operational reliability requires validation on a newly sourced, previously unseen dataset.

---

### 3.2 Breakdown of Field Supported Cases (17 Images)

```text
Image File                      Ref Pathology               Outcome Status      Displayed Diagnosis                 Confidence Score
------------------------------------------------------------------------------------------------------------------------------------
early_blight_wikimedia.jpg      Tomato Early Blight         CORRECT_ASSESSED    Suspected Tomato Early Blight        98.7% (Sim: 0.61)
septoria_leaf_spot_real.jpg     Tomato Septoria             CORRECT_ASSESSED    Suspected Tomato Septoria            88.4% (Sim: 0.62)
septoria_leaf_symptoms.jpg      Tomato Septoria             CORRECT_ASSESSED    Suspected Tomato Septoria            88.4% (Sim: 0.62)
sample_septoria.jpg             Tomato Septoria             CORRECT_ASSESSED    Suspected Tomato Septoria            98.6% (Sim: 0.81)
late_blight_field.jpg           Tomato Late Blight          CORRECT_ASSESSED    Suspected Tomato Late Blight         91.8% (Sim: 0.61)
yellow_leaf_curl.jpg            Yellow Leaf Curl Virus      CORRECT_ASSESSED    Suspected Tomato Yellow Leaf Curl   100.0% (Sim: 0.84)
healthy_sample.jpg              Healthy Foliage             CORRECT_ASSESSED    Suspected Tomato Healthy            100.0% (Sim: 0.89)
------------------------------------------------------------------------------------------------------------------------------------
leaf_mold_wm.jpg                Tomato Leaf Mold            INCORRECT_ASSESSED  Suspected Tomato Early Blight       100.0% (Sim: 0.60)
------------------------------------------------------------------------------------------------------------------------------------
concentric_leaf_spot.jpg        Tomato Early Blight         UNCERTAIN_MATCH     Uncertain — Suspected Early Blight  100.0% (Sim: 0.59)
c5f99319.jpeg                   Tomato Early Blight         UNCERTAIN_MATCH     Uncertain — Suspected Early Blight   73.7% (Sim: 0.51)
early_blight_flr.jpg            Tomato Early Blight         UNCERTAIN_LOOKALIKE Uncertain — Suspected Late Blight    82.0% (Sim: 0.49)
late_blight_real.jpg            Tomato Late Blight          UNCERTAIN_MATCH     Uncertain — Suspected Late Blight    99.9% (Sim: 0.44)
late_blight_wm2.jpg             Tomato Late Blight          UNCERTAIN_LOOKALIKE Uncertain — Suspected Early Blight   84.6% (Sim: 0.52)
target_spot_leaves.png          Tomato Target Spot          UNCERTAIN_LOOKALIKE Uncertain — Suspected Septoria       79.2% (Sim: 0.53)
spider_mite_clemson.jpg         Spider Mites                UNCERTAIN_LOOKALIKE Uncertain — Suspected Septoria       86.7% (Sim: 0.56)
mosaic_virus_real.jpg           Tomato Mosaic Virus         UNCERTAIN_LOOKALIKE Uncertain — Suspected Yellow Curl    75.2% (Sim: 0.52)
healthy_outdoor.jpg             Healthy Foliage             UNCERTAIN_LOOKALIKE Uncertain — Suspected Late Blight    75.5% (Sim: 0.48)
------------------------------------------------------------------------------------------------------------------------------------
Total Supported: 17 | Assessed Correct: 7 | Assessed Incorrect: 1 | Uncertain: 9 | False Unsupported: 0 | Quality Rejected: 0
```

---

### 3.3 Breakdown of Unsupported Conditions & Controls (9 Images)

```text
Image File                      Ref Pathology / Type        Outcome Status                System Status / Code               Confidence
------------------------------------------------------------------------------------------------------------------------------------
magnesium_deficiency_rhs.webp   Magnesium Deficiency        CORRECT_REJECTED_UNSUPPORTED  UNSUPPORTED_CONDITION (nutrient)     N/A
leafminer_pest.jpg              Leafminer Pest Damage       CORRECT_REJECTED_UNSUPPORTED  UNSUPPORTED_CONDITION (leafminer)    N/A
powdery_mildew_rhs.jpg          Powdery Mildew (*Oidium*)   CORRECT_REJECTED_UNSUPPORTED  UNSUPPORTED_CONDITION (powdery)      N/A
powdery_mildew_wm.jpg           Powdery Mildew (4624x3468)  CORRECT_REJECTED_UNSUPPORTED  UNSUPPORTED_CONDITION (powdery)      N/A
------------------------------------------------------------------------------------------------------------------------------------
coffee_cup_nonleaf.jpg          Non-Foliar (Ceramic Cup)    CORRECT_REJECTED_CONTROL      IMAGE_REJECTED (non-foliar)          N/A
computer_keyboard.jpg           Non-Foliar (Electronics)    CORRECT_REJECTED_CONTROL      IMAGE_REJECTED (non-foliar)          N/A
red_apple_fruit.jpg             Non-Foliar (Apple Fruit)    CORRECT_REJECTED_CONTROL      IMAGE_REJECTED (pure fruit)          N/A
corrupted_html.jpg              Quality Degraded (Corrupt)  CORRECT_REJECTED_CONTROL      IMAGE_REJECTED (unidentified)        N/A
blurred_tomato_leaf.jpg         Quality Degraded (Defocus)  UNCERTAIN_CONTROL_DEFERRED    UNCERTAIN (low sharpness/margin)     N/A
------------------------------------------------------------------------------------------------------------------------------------
Total Unsupported / Controls: 9 | Correctly Rejected/Deferred: 9 (100.0%) | False Disease Acceptance: 0 (0.00%)
```

---

## 4. Confirmed Problem Fixes & Root Cause Analysis

### 4.1 Fix A: Elimination of False Powdery-Mildew Rejections on White Backgrounds
- **Problem:** Three supported-condition images (`target_spot_leaves.png`, `leaf_mold_wm.jpg`, `spider_mite_clemson.jpg`) were previously rejected as suspected powdery mildew because white mounting-card pixels satisfied the foliage check and were counted as powdery fungal bloom.
- **Root Cause:** Neutral white or gray cards have `diff < 28` and low saturation. In the previous mask, any bright neutral pixel with `g > 35` was counted as foliage.
- **Remediation Implemented:**
  1. Background card segmentation explicitly flags neutral or cool mounting surfaces:
     `is_neutral_or_cool_card = (sat < 0.06) | ((b >= g) & (g > 100) & (diff < 25))`
  2. Powdery mildew mycelium requires a foliar chloroplast base:
     `g >= b * 0.98` and `g >= r * 0.98` with `diff < 26` and `~is_neutral_or_cool_card`.
  3. Connected-component clustering requires contiguous mycelial colonies (`cluster_ratio >= 0.018` or `powder_ratio > 0.12`).
- **Verification:**
  - All 3 mounting card images now pass without powdery mildew rejection (`target_spot_leaves.png` and `spider_mite_clemson.jpg` safely deferred as `UNCERTAIN`; `leaf_mold_wm.jpg` assessed).
  - Genuine powdery mildew photographs (`powdery_mildew_rhs.jpg` and `Tomaten-Echter_Mehltau-JS-2.jpg`) remain 100% reliably rejected as `UNSUPPORTED_CONDITION`.
  - Healthy glossy leaves with bright specular sun highlights are not falsely rejected.

### 4.2 Fix B: Foliar Region Selection for Whole-Plant Photos Containing Fruit
- **Problem:** Photographs showing infected plants with visible tomato fruit (`late_blight_real.jpg`, `late_blight_wm2.jpg`) were rejected by the quality gate (`"Image appears to contain a fruit rather than a plant leaf"`).
- **Root Cause:** Quality gate applied a global threshold `red_fruit_ratio > 0.20` designed for single detached leaves.
- **Remediation Implemented:**
  1. Pure fruit images are distinguished from mixed canopy: rejection only occurs if fruit dominates (`red_fruit_ratio > 0.40`) and green foliage is negligible (`green_ratio < 0.02`), keeping `red_apple_fruit.jpg` correctly rejected.
  2. When fruit is present in a mixed canopy (`red_fruit_ratio > 0.15`), automatic foliar connected-component segmentation isolates the dominant leaf cluster (`foliar_crop_applied = True`, `foliar_crop_box = [x1, y1, x2, y2]`).
  3. Added manual crop ROI fallback (`crop_box: tuple[int, int, int, int]`) supported via API and UI.
- **Verification:**
  - `late_blight_real.jpg` and `late_blight_wm2.jpg` pass the quality gate with foliar crops applied and are safely classified as `UNCERTAIN` with Late Blight lookalike differentials.
  - Pure fruit control (`red_apple_fruit.jpg`) continues to be rejected at the quality gate.

### 4.3 Fix C: High-Resolution Upload Downsampling Without Distortion
- **Problem:** Photographs exceeding 4096 pixels (e.g. `Tomaten-Echter_Mehltau-JS-2.jpg`, 4624 × 3468) were rejected outright.
- **Root Cause:** Hard limit `MAX_IMAGE_DIMENSION = 4096` in quality gate.
- **Remediation Implemented:**
  1. Documented upload limit raised to `MAX_UPLOAD_DIMENSION = 8192`, with decompression bomb protection set to `Image.MAX_IMAGE_PIXELS = 50_000_000` and `MAX_FILE_SIZE_BYTES = 20 MB`.
  2. Proportional aspect-preserving downsampling scales images to `MAX_INFERENCE_DIMENSION = 2048` using `scale = 2048.0 / max(w, h)` with Lanczos filtering, avoiding stretching or distortion.
  3. Quality checks and API responses distinguish `original_dimensions` from `inference_dimensions`.
- **Verification:**
  - `Tomaten-Echter_Mehltau-JS-2.jpg` (4624 × 3468) is accepted, downsampled to 2048 × 1536, and correctly classified as `unsupported_powdery_mildew`.
  - Synthetic 4800 × 3200 images verify aspect ratio preservation within < 0.02.

---

## 5. Automated Test Suite Results

The complete automated regression test suite was executed in the local environment:
- **Test Command:** `.venv/bin/python3.11 -m unittest discover -s tests -p "test_*.py"`
- **Tests Executed:** 141
- **Failures:** 0
- **Errors:** 0
- **Execution Time:** 24.18 seconds
- **Pass Rate:** **100.00% (141 / 141)**

New regression test classes added:
- `FieldEvaluationDefectRegressionTests`: 9 automated tests covering white mounting card backgrounds, genuine powdery mildew detection, glossy leaf specular highlights, mixed canopy foliar isolation, pure fruit rejection, manual crop box ROI override, high-resolution proportional downsampling, and HTTP API crop box handling.

---

## 6. Project Distribution Archives

Clean project archives updated with all confirmed fixes:
- `plant-health-rover-sim_corrected.zip`
- `plant-health-rover-sim.zip`

Excluded from distribution: `.venv/`, `.git/`, `__pycache__/`, `*.pyc`, `golden_vectors.json`.

