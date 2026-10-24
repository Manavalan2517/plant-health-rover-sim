# Dataset, annotation and evaluation plan

## Current data status

No labelled paired field dataset or trained model was provided. The current app makes **synthetic images and synthetic sensor data** for software integration only. Scenario-generated image cues are not model predictions, and generated sensor readings are not agronomic measurements. Do not merge them into a real-world evaluation set or use them as ground truth.

## Candidate public datasets

- **PlantVillage:** candidate for an initial controlled-image benchmark. Check the downloaded release's licence/terms and attribution requirements; do not treat a GitHub URL or “public/open access” wording as blanket redistribution permission.
- **PlantDoc:** candidate for more varied in-scene images. Check the exact repository/release licence, annotations and provenance. It is not a paired dataset for our field-node time series out of the box.
- **Own paired dataset:** essential for validation under the actual fixed camera distance, LED hood, colour card, camera settings, target tomato varieties and field conditions.

Store source metadata per image: stable ID, file checksum, dataset/release, original URL/source, licence, class label as supplied, plant identifier/group, collection session, timestamp, cultivar if known, growth stage, camera/lens/lighting, field/site identifier and annotation provenance. Keep a copy of each dataset's licensing text and terms in a dataset-specific `README` before distributing derived files or model weights.

## Data collection protocol for physical trial

1. Assign immutable plot, row, plant and session IDs. Tags (ArUco/RFID) must be verified against manual field notes.
2. Fix hood/camera distance, angle, exposure strategy, LED level and colour-card position. Save raw capture and quality metrics; don't keep only preprocessed images.
3. Timestamp captures in UTC plus local-time metadata. Record clock source and time uncertainty.
4. Associate sensor node and zone identifiers explicitly. Pair images with sensor/history windows using documented tolerances (for example, nearest recent sample and a defined past-24-hour window), not an implicit global “latest value.”
5. Capture context: cultivar, plant stage, watering/irrigation event, recent weather when available, fertiliser events, visible symptoms, management actions and notes about occlusion.
6. Collect both symptomatic and apparently healthy samples over different days, lighting and conditions. A sample should include uncertain/ambiguous observations rather than force a single class.
7. Record follow-up/independent confirmation where feasible (expert assessment, lab result or documented agronomist judgement). Label confidence and confirmation method separately.
8. Use duplicate and near-duplicate checks. Repeated images from the same plant/session must remain in the same data split.

## Annotation ontology (initial proposal; confirm with agronomist)

Store separate fields instead of one forced diagnosis:

- `visual_label`: visually described class/symptom (e.g. healthy appearance, early-blight-like lesions, wilting/curling, yellowing, other, unidentifiable, uncertain).
- `reference_diagnosis`: only when independently confirmed; include method and reviewer.
- `abiotic_context`: water/heat/nutrient or other stress evidence, with source and confidence.
- `sensor_data_quality`: valid/calibrated, questionable, missing, stale.
- `annotation_confidence`: ordinal plus free-text evidence.
- `review_status`: unreviewed, reviewed, disagreement, confirmed.

“Disease-like symptoms” are visual observations, not a definitive pathogen identification. Do not label synthetic generated leaves as ground truth labels. Do not label a low synthetic NPK value as a confirmed plant nutrient deficiency.

## Splits and leakage prevention

- Split by **plant identity** and preferably site/session/time group so the test plants were not part of training or tuning.
- Never allow images of one plant or neighbouring frames from the same capture burst to cross train/validation/test boundaries.
- Apply flips, rotation, shift, zoom, colour augmentation and cropping only inside the training pipeline after grouping is set. Document all transformations.
- Fit normalisation, feature scaling, oversampling, feature selection and calibration only using the training split/validation protocol.
- Keep a final holdout untouched until the model/threshold is locked.
- Maintain a second evaluation on real rover-hood data even if public dataset test results are strong.

## Experiments, in order

A. Image-only MobileNetV2 baseline with reproducible preprocessing and class mapping.

B. Sensor-only feature baseline using validated sample windows/trends.

C. Fused model using paired image and sensor/history data. Compare C to A using same held-out plants, time windows and evaluation criteria.

D. Deployment version (quantized TFLite) evaluated against its FP32 counterpart with fixed images and a published model file/hash.

Report per-class precision/recall, macro F1, balanced accuracy, confusion matrix, coverage vs reject/uncertain rate, calibration error/reliability diagram, subgroup results and qualitative failure cases. Report sample/plant counts and confidence intervals where practical. High disease recall may matter, but also count false alarms; threshold selection should reflect farmer cost and be evaluated on held-out data.

## Model card required before any real recommendation

- Intended use and users, out-of-scope use, supported crop/stage/disease classes.
- Data sources/licences, exact splits, collection sites and domain gaps.
- Preprocessing/model architecture/model and class-index hashes.
- Metrics per class, uncertainty/reject policy, calibration and known limitations.
- On-target Pi latency/RAM/temperature and quantisation impact.
- Human review rules, escalation and feedback collection.
- Explicit statement that an image/sensor prediction is not a confirmed diagnosis unless the reference process supports that label.
