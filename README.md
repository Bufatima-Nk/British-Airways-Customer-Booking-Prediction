# British Airways Customer Booking Prediction

> Binary classification model predicting whether a customer will complete a flight booking, trained on **50,000 BA booking records** with 14 features. Applied SMOTE oversampling, z-score outlier removal, SelectKBest feature selection, and Random Forest with GridSearchCV — achieving **85.2% test accuracy** with key finding that booking origin (Australia, Malaysia) dominates predictions.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange?logo=scikit-learn)
![Dataset](https://img.shields.io/badge/Dataset-50%2C000%20bookings-purple)
![Forage](https://img.shields.io/badge/British%20Airways-Forage%20Virtual%20Internship-005EB8)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## ⚠️ Honest Model Assessment

This model achieves **85.2% accuracy but has near-zero recall for completed bookings (class 1)**. Understanding why is the most important analytical takeaway from this project.

| Metric | Class 0 (Not Completed) | Class 1 (Completed) |
|--------|:-----------------------:|:-------------------:|
| Precision | 0.85 | 0.50 |
| **Recall** | **1.00** | **0.01** |
| F1-Score | 0.92 | 0.02 |
| Support | 8,520 | 1,480 |

**What this means:** The model correctly identifies 100% of non-completions but catches only 1% of actual completed bookings. In practice it predicts "not complete" for almost every case — which is correct 85% of the time due to class imbalance (85% of bookings are not completed). The 85.2% accuracy number is misleading without this context.

**Why this matters:** In a real BA deployment, missing 99% of actual completions (false negatives) would be catastrophic — you'd never identify customers likely to convert. The correct metric for this problem is **recall on class 1** or **AUC-ROC**, not accuracy.

**Root cause:** Despite applying SMOTE to the training set, SMOTE was applied *after* SelectKBest feature selection on the full dataset, causing a subtle preprocessing pipeline issue. Additionally, the class imbalance (85:15) combined with a small positive class makes recall recovery difficult without threshold tuning.

**What would fix this:**
- Lower the classification threshold (default 0.5 → try 0.2–0.3) to increase class 1 recall
- Use `class_weight='balanced'` in RandomForestClassifier
- Evaluate with AUC-ROC instead of accuracy
- Apply SMOTE *inside* the cross-validation loop to prevent data leakage

This honest assessment is more valuable than hiding the recall issue behind the accuracy headline.

---

## Results

### Model Performance

| Metric | Value |
|--------|-------|
| Baseline accuracy (majority class) | 85.0% |
| Test accuracy | **85.2%** |
| 10-fold CV accuracy | 83.0% |
| GridSearchCV best score | 85.0% |
| Class 1 recall | **0.01** ⚠️ |
| Macro F1 | 0.47 |

### Best Hyperparameters (GridSearchCV)

```
criterion:    entropy
max_depth:    10
max_features: sqrt
n_estimators: 300
```

### Top 10 Feature Importances

| Rank | Feature | Importance | Insight |
|------|---------|:----------:|---------|
| 1 | `booking_origin_Australia` | 0.358 | **Dominant predictor** — 35.8% of model decisions |
| 2 | `booking_origin_Malaysia` | 0.285 | Second dominant — 28.5% |
| 3 | `flight_duration` | 0.170 | Longer flights → different completion behavior |
| 4 | `wants_extra_baggage` | 0.052 | Engagement signal |
| 5 | `route_PENTPE` | 0.044 | Penang–Taipei route specific behavior |
| 6 | `booking_origin_Indonesia` | 0.036 | Regional pattern |
| 7 | `booking_origin_Singapore` | 0.024 | Regional pattern |
| 8 | `route_ICNPEN` | 0.022 | Incheon–Penang route |
| 9 | `route_JHBKTM` | 0.006 | — |
| 10 | `route_KTMPEN` | 0.003 | — |

**Critical finding:** Two features (`booking_origin_Australia` + `booking_origin_Malaysia`) account for **64.3% of total feature importance**. The model is primarily a booking-origin classifier, not a true behavioral predictor. This suggests the dataset has strong geographic confounding — Australian and Malaysian bookers have very different completion rates from other origins, likely reflecting dataset composition rather than generalizable booking behavior.

---

## Dataset

- **Source:** British Airways customer booking data (via Forage Virtual Internship)
- **Records:** 50,000 bookings
- **Target:** `booking_complete` (0 = not completed, 1 = completed)
- **Class distribution:** 85.0% not completed / **15.0% completed** (imbalanced)
- **No missing values**

### Features

| Feature | Type | Description |
|---------|------|-------------|
| `num_passengers` | int | Number of passengers |
| `sales_channel` | categorical | Internet vs. Mobile |
| `trip_type` | categorical | Round Trip / One Way / Circle Trip |
| `purchase_lead` | int | Days between booking and travel (0–867) |
| `length_of_stay` | int | Nights at destination (0–778) |
| `flight_hour` | int | Departure hour (0–23) |
| `flight_day` | ordinal | Day of week (Mon=1 → Sun=7) |
| `route` | categorical | Origin–destination code (799 unique → grouped) |
| `booking_origin` | categorical | Country of booking (104 → grouped) |
| `wants_extra_baggage` | binary | Add-on preference |
| `wants_preferred_seat` | binary | Add-on preference |
| `wants_in_flight_meals` | binary | Add-on preference |
| `flight_duration` | float | Hours (4.67–9.50) |

---

## Approach

### 1. EDA & Preprocessing
- Mapped `flight_day` (Mon–Sun) → ordinal integers (1–7)
- Grouped rare routes (< 50 occurrences → "Other"): 799 → 217 categories
- Grouped rare booking origins (< 100 occurrences → "Other"): 104 → 20 categories
- One-hot encoded all categorical columns (`sales_channel`, `trip_type`, `route`, `booking_origin`)

### 2. Class Imbalance Handling
- Target split: 85.0% not completed vs. 15.0% completed
- Applied **SMOTE** (`sampling_strategy='auto'`) to training set to oversample minority class
- Z-score outlier removal on training set (threshold: |z| < 3)

### 3. Feature Selection
- `SelectKBest(f_classif, k=10)` — selected top 10 features by ANOVA F-statistic
- Applied to full dataset before splitting (note: this introduces mild leakage — see limitations)

### 4. Model Training
- `RandomForestClassifier(criterion='entropy')`
- 10-fold cross-validation: mean accuracy = 83.0%
- `GridSearchCV` over 18 parameter combinations

### 5. Evaluation
- Test accuracy: 85.2%
- Classification report with precision/recall/F1 per class
- Confusion matrix visualization
- Feature importance bar chart

---

## Project Structure

```
British-Airways-Customer-Booking-Prediction/
│
├── ba_task_2.ipynb              # Full pipeline: EDA → preprocessing → modeling → evaluation
├── customer_booking.csv         # Dataset (50,000 booking records)
├── key_insights_task2_BA.pdf    # Key insights summary
├── requirements.txt             # Dependencies
└── README.md
```

---

## How to Run

```bash
git clone https://github.com/Bufatima-Nk/British-Airways-Customer-Booking-Prediction
cd British-Airways-Customer-Booking-Prediction
pip install -r requirements.txt
jupyter notebook ba_task_2.ipynb
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| ML Model | scikit-learn (RandomForestClassifier, GridSearchCV, cross_validate) |
| Imbalance | imbalanced-learn (SMOTE) |
| Feature Selection | scikit-learn SelectKBest (f_classif) |
| Outlier Removal | scipy zscore |
| Data | pandas, NumPy |
| Visualization | Matplotlib, Seaborn, ConfusionMatrixDisplay |

---

## Limitations & What I Would Do Differently

| Issue | Fix |
|-------|-----|
| Class 1 recall = 0.01 | Tune classification threshold to 0.2–0.3; use `class_weight='balanced'` |
| SelectKBest applied before split | Move feature selection inside cross-validation pipeline to prevent leakage |
| Accuracy as primary metric | Switch to AUC-ROC and F1 for imbalanced classification |
| SMOTE outside CV loop | Apply SMOTE inside each fold to prevent oversampling leakage |
| Booking origin dominance | Investigate whether origin is a proxy for dataset bias, not true signal |

---

## Certificate

Completed as part of the [British Airways Data Science Virtual Experience](https://www.theforage.com/simulations/british-airways/data-science-yqoz) on Forage.

---

## Author

**Bufatima N.K.**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-bufatima--n--k-blue?logo=linkedin)](https://linkedin.com/in/bufatima-n-k)
[![GitHub](https://img.shields.io/badge/GitHub-Bufatima--Nk-black?logo=github)](https://github.com/Bufatima-Nk)
