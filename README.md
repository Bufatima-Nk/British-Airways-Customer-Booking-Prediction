# British Airways Customer Booking Prediction

> Binary classification model predicting whether a customer will complete a flight booking, trained on **50,000 BA booking records**. Built a leak-free `ImbPipeline` with SMOTE oversampling, `class_weight='balanced'`, and stratified cross-validation — achieving **AUC-ROC: 0.733** and **class 1 recall: 0.52**.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange?logo=scikit-learn)
![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.733-brightgreen)
![Recall](https://img.shields.io/badge/Class%201%20Recall-0.52-green)
![Forage](https://img.shields.io/badge/British%20Airways-Forage%20Virtual%20Internship-005EB8)

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
| 5 | `booking_origin_Indonesia` | 0.047 | Regional booking pattern |
| 6 | `length_of_stay` | 0.040 | Longer trips → more committed bookings |
| 7 | `route_PENTPE` | 0.039 | Penang–Taipei route specific behavior |

---

## Dataset

- **Source:** British Airways customer booking data (Forage Virtual Internship)
- **Records:** 50,000 bookings
- **Target:** `booking_complete` (0 = not completed, 1 = completed)
- **Class ratio:** 85% not completed / 15% completed (imbalanced)
- **No missing values**
- **Train/Test:** 80% / 20% stratified split

### Features

| Feature | Type | Description |
|---------|------|-------------|
| `num_passengers` | int | Number of passengers travelling |
| `sales_channel` | categorical | Internet vs. Mobile |
| `trip_type` | categorical | Round Trip / One Way / Circle Trip |
| `purchase_lead` | int | Days between booking and travel date |
| `length_of_stay` | int | Nights at destination |
| `flight_hour` | int | Departure hour (0–23) |
| `flight_day` | ordinal | Day of week (Mon=1 → Sun=7) |
| `route` | categorical | Origin–destination code |
| `booking_origin` | categorical | Country of booking |
| `wants_extra_baggage` | binary | Add-on preference |
| `wants_preferred_seat` | binary | Add-on preference |
| `wants_in_flight_meals` | binary | Add-on preference |
| `flight_duration` | float | Flight duration in hours |

---

## Feature Engineering

Three new features created beyond the original columns:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `addon_count` | `baggage + seat + meals` | Engagement signal — more add-ons = more committed customer |
| `is_last_minute` | `purchase_lead ≤ 7` | Last-minute bookers behave differently from planners |
| `is_weekend_flight` | `flight_day ≥ 6` | Weekend vs weekday booking behavior differs |

Also applied:
- Rare route grouping: 799 → 217 categories (threshold: < 50 occurrences → "Other")
- Rare origin grouping: 104 → 20 categories (threshold: < 100 → "Other")
- One-hot encoding for all categorical columns

---

## Pipeline Architecture

```
Input (50,000 bookings × 14 features)
    ↓
Feature Engineering
    → Ordinal encode flight_day
    → Create addon_count, is_last_minute, is_weekend_flight
    → Group rare routes and origins
    → One-hot encode categoricals
    ↓
Stratified 80/20 Split (preserves 15% class 1 in both sets)
    ↓
ImbPipeline (SMOTE applied only inside training folds — no leakage):
    [1] SMOTE(k_neighbors=5)
    [2] RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            class_weight='balanced',
            criterion='entropy'
        )
    ↓
Evaluation: AUC-ROC · Precision-Recall · F1 · Confusion Matrix
```

---

## Why AUC-ROC, Not Accuracy

With 85% of bookings not completed, a model predicting "not completed" for every case would achieve 85% accuracy while being completely useless. **AUC-ROC = 0.733** measures the model's ability to correctly rank a completed booking above a non-completed one — a meaningful metric independent of class distribution.

---

## Key Business Insights

**1. Booking origin is the dominant signal.** Malaysia (23.9% importance) and Australia (6.4%) together explain ~30% of model decisions. BA's APAC-heavy booking base means regional conversion rates differ significantly.

**2. Flight duration predicts commitment** (10.9% importance). Customers booking longer routes are more likely to complete — long-haul travel requires more deliberate decision-making.

**3. Internet vs. Mobile channel gap.** Internet bookings complete at a higher rate than Mobile, suggesting potential UX friction in the mobile checkout experience.

**4. Add-on selection signals purchase intent.** Customers who select extra baggage, preferred seats, or meals show slightly higher completion rates — these choices indicate a more committed buyer.

**5. Purchase lead time matters.** Highly skewed (mean: 85 days, max: 867 days). Last-minute bookers (≤7 days) have different conversion behavior than advance planners.

---

## Project Structure

```
British-Airways-Customer-Booking-Prediction/
│
├── ba_booking_prediction.ipynb     # Full pipeline: EDA → feature engineering → modeling
├── customer_booking.csv            # Dataset (50,000 records)
├── fig1_eda.png                    # EDA visualizations
├── fig2_evaluation.png             # ROC, PR curve, confusion matrix
├── fig3_features.png               # Feature importance chart
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

## Limitations & Future Work

- **Threshold tuning:** Lowering the default threshold from 0.5 → 0.3 would increase recall at the cost of precision — useful if catching more completions is the priority.
- **Try LightGBM** with `scale_pos_weight` — likely faster convergence with similar or better AUC.
- **Missing features:** Price, route competition, customer loyalty tier, and device type would likely be strong predictors but are absent from this dataset.

---

## Certificate

Completed as part of the [British Airways Data Science Virtual Experience](https://www.theforage.com/simulations/british-airways/data-science-yqoz) on Forage.

---

## Author

**Bufatima N.K.**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-bufatima--n--k-blue?logo=linkedin)](https://linkedin.com/in/bufatima-n-k)
[![GitHub](https://img.shields.io/badge/GitHub-Bufatima--Nk-black?logo=github)](https://github.com/Bufatima-Nk)
