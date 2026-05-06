# British Airways Customer Booking Prediction — Rebuilt

> Binary classification predicting booking completion on **50,000 BA records**. Rebuilt with a leak-free `ImbPipeline`, `class_weight='balanced'`, stratified cross-validation, and AUC-ROC evaluation — improving class 1 recall from **0.01 → 0.52** and AUC-ROC to **0.733**.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange?logo=scikit-learn)
![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.733-brightgreen)
![Recall](https://img.shields.io/badge/Class%201%20Recall-0.52-green)
![Forage](https://img.shields.io/badge/British%20Airways-Forage%20Virtual%20Internship-005EB8)

---

## What Changed vs. the Original Approach

| Issue | Original | This Version |
|-------|----------|-------------|
| SMOTE placement | Applied before split (leakage) | Inside `ImbPipeline` — only on training folds |
| Feature selection | `SelectKBest` on full dataset (leakage) | Removed; all features used with proper regularization |
| Class handling | SMOTE only | SMOTE + `class_weight='balanced'` |
| Evaluation metric | Accuracy (misleading) | AUC-ROC, F1, Recall |
| Cross-validation | Standard KFold | Stratified KFold (preserves class ratio) |
| Class 1 Recall | **0.01** ❌ | **0.52** ✅ |

---

## Results

### Cross-Validation (5-Fold Stratified)

| Metric | Mean | Std |
|--------|:----:|:---:|
| **AUC-ROC** | **0.729** | ±0.006 |
| Recall (class 1) | 0.492 | ±0.013 |
| F1 (class 1) | 0.382 | ±0.007 |
| Precision (class 1) | 0.312 | ±0.006 |
| Accuracy | 0.762 | ±0.004 |

### Test Set Performance

| Metric | Class 0 (Not Completed) | Class 1 (Completed) |
|--------|:-----------------------:|:-------------------:|
| Precision | 0.90 | 0.31 |
| **Recall** | 0.79 | **0.52** |
| F1-Score | 0.85 | 0.39 |
| Support | 8,504 | 1,496 |
| **AUC-ROC** | — | **0.733** |

### Top Feature Importances

| Rank | Feature | Importance | Insight |
|------|---------|:----------:|---------|
| 1 | `booking_origin_Malaysia` | 0.239 | Malaysian bookers have distinct completion patterns |
| 2 | `flight_duration` | 0.109 | Longer routes → higher commitment → more completions |
| 3 | `booking_origin_Australia` | 0.064 | Second-largest origin group |
| 4 | `sales_channel_Internet` | 0.050 | Internet vs Mobile completion difference |
| 5 | `booking_origin_Indonesia` | 0.047 | Regional pattern |
| 6 | `length_of_stay` | 0.040 | Longer trips → more committed bookings |
| 7 | `route_PENTPE` | 0.039 | Penang–Taipei specific behavior |

---

## Feature Engineering

Three new features added beyond the original columns:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `addon_count` | `baggage + seat + meals` | Engagement signal — customers selecting more add-ons are more committed |
| `is_last_minute` | `purchase_lead ≤ 7` | Last-minute bookers may have different dropout rates |
| `is_weekend_flight` | `flight_day ≥ 6` | Weekend vs weekday booking behavior |

---

## Pipeline Architecture

```
Input (50,000 bookings × 14 features)
    ↓
Feature Engineering
    → flight_day ordinal encoding
    → addon_count, is_last_minute, is_weekend_flight
    → rare route/origin grouping (< 50/100 occurrences → 'Other')
    → one-hot encoding (sales_channel, trip_type, route, booking_origin)
    ↓
Stratified 80/20 Split (stratify=y preserves 15% class 1 in both sets)
    ↓
ImbPipeline (applied ONLY to training folds):
    [1] SMOTE(k_neighbors=5)              → oversample minority class
    [2] RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            class_weight='balanced',      → further imbalance correction
            criterion='entropy'
        )
    ↓
Evaluation: AUC-ROC, Precision-Recall, F1, Confusion Matrix
```

---

## Why AUC-ROC, Not Accuracy

With 85% of bookings not completed, a model that always predicts "not completed" achieves **85% accuracy** while being completely useless for the business goal of identifying customers likely to convert.

**AUC-ROC = 0.733** means the model has a 73.3% chance of correctly ranking a completed booking above a non-completed booking — a meaningful performance signal independent of class distribution.

**Recall = 0.52** means the model catches 52% of actual completions — a 51× improvement over the original model's recall of 0.01.

---

## Key Business Insights

**1. Booking origin is the dominant signal.** Malaysia (23.9% importance) and Australia (6.4%) together explain ~30% of model decisions. BA's APAC-heavy booking base means regional marketing and UX differences have outsized impact on conversion.

**2. Flight duration predicts commitment** (10.9% importance). Customers booking longer routes (>8 hrs) are more likely to complete — they've invested more consideration into the trip.

**3. Internet vs. Mobile gap.** Internet bookings complete at a higher rate. Mobile may have checkout friction or be used more for browsing than committing.

**4. Add-on selection is an early commitment signal.** Customers who select extra baggage, preferred seats, or meals show slightly higher completion rates — these add-ons require deliberate choices that signal purchase intent.

**5. Purchase lead time is highly skewed** (mean 85 days, max 867 days). Last-minute bookers (≤7 days) behave differently — they're likely to convert or not at all.

---

## Dataset

- **Source:** British Airways customer booking data (Forage Virtual Internship)
- **Records:** 50,000 bookings
- **Target:** `booking_complete` (0=not completed, 1=completed)
- **Class ratio:** 85% : 15% (imbalanced)
- **No missing values**
- **Train/Test:** 80/20 stratified split

---

## Project Structure

```
British-Airways-Customer-Booking-Prediction/
│
├── ba_booking_prediction.ipynb     # Rebuilt notebook — full pipeline, no leakage
├── customer_booking.csv            # Dataset
├── key_insights_task2_BA.pdf       # Original task summary
├── requirements.txt                # Dependencies
└── README.md
```

---

## How to Run

```bash
git clone https://github.com/Bufatima-Nk/British-Airways-Customer-Booking-Prediction
cd British-Airways-Customer-Booking-Prediction
pip install -r requirements.txt
jupyter notebook ba_booking_prediction.ipynb
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| ML | scikit-learn (RandomForestClassifier, StratifiedKFold, cross_validate) |
| Imbalance | imbalanced-learn (SMOTE, ImbPipeline) |
| Data | pandas, NumPy |
| Visualization | Matplotlib, Seaborn, ConfusionMatrixDisplay, RocCurveDisplay |

---

## Limitations & Next Steps

- **Threshold tuning:** Default threshold = 0.5. Lowering to 0.3 would increase recall further at cost of precision — worthwhile if catching more completions matters more than false positives.
- **Try LightGBM** with `scale_pos_weight=(85/15)` — likely faster convergence and similar or better AUC.
- **Missing features:** Price, route competition, customer history/loyalty tier, device type. These would likely be the strongest predictors and are absent from this dataset.
- **Booking origin investigation:** The dominance of Malaysia/Australia may reflect dataset composition bias rather than generalizable booking behavior. Worth investigating with BA stakeholders.

---

## Author

**Bufatima N.K.**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-bufatima--n--k-blue?logo=linkedin)](https://linkedin.com/in/bufatima-n-k)
[![GitHub](https://img.shields.io/badge/GitHub-Bufatima--Nk-black?logo=github)](https://github.com/Bufatima-Nk)
