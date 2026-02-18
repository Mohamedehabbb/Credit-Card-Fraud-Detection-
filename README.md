# Credit-Card-Fraud-Detection-
End- to-End  Data science project
# 💳 Credit Card Fraud Detection — End-to-End Data science Project

## 📌 Project Overview

This project presents a complete, production-oriented fraud detection pipeline built on a highly imbalanced credit card transactions dataset. The objective is to accurately identify fraudulent transactions while minimizing financial loss and operational investigation costs.

The notebook walks through every stage of a real-world machine learning workflow:

* Deep exploratory data analysis (EDA)
* Severe class imbalance handling strategies
* Feature engineering & preprocessing
* Baseline and advanced model training
* Precision-Recall focused evaluation (AUPRC)
* Threshold optimization based on financial cost
* Model explainability using SHAP
* Business recommendations and deployment plan

This project is designed not only to achieve strong predictive performance but also to align model decisions with business impact.

---

# 📂 Dataset Description

The dataset contains anonymized credit card transactions.

### Key Characteristics:

* Features V1–V28 are PCA-transformed components (original variables hidden for privacy)
* "Amount" represents transaction value
* "Time" represents seconds elapsed between transactions
* "Class" is the target variable:

  * 0 → Normal transaction
  * 1 → Fraudulent transaction

### ⚠️ Major Challenge

The dataset is **extremely imbalanced**, with fraud cases representing a very small percentage of all transactions.

This makes accuracy and ROC-AUC misleading metrics. Therefore, the project prioritizes:

* Recall (detecting fraud)
* Precision (avoiding false alarms)
* AUPRC (Area Under Precision-Recall Curve)

---

# 🔎 STEP 1 — Data Loading & Deep Understanding

* Loaded and inspected dataset structure
* Verified data types and missing values
* Measured fraud ratio
* Identified severe class imbalance

Early understanding confirmed that imbalance handling would be critical to success.

---

# 📊 STEP 2 — Deep EDA & Fraud Behavior Analysis

Because features are PCA-transformed and not interpretable by name, analysis focused on behavior patterns.

### Key Analyses:

## 1️⃣ Transaction Amount Analysis

* Distribution visualization
* Fraud vs Normal comparison
* Detection of skewness and outliers

**Insight:** Fraud transactions often occur at medium amounts to avoid suspicion.

## 2️⃣ Time-Based Behavior

* Fraud occurrence over time
* Pattern consistency analysis

## 3️⃣ Correlation Analysis

* Full heatmap
* Correlation with target
* Identification of highly predictive components

## 4️⃣ Fraud Feature Distribution Deep Dive

* Feature-level separation
* Outlier behavior in fraud transactions

### ✅ EDA Outcome

* Identified informative components
* Confirmed separability in certain PCA features
* Validated need for precision-recall evaluation

---

# ⚙️ STEP 3 — Feature Engineering & Imbalance Strategy Design

## 🔧 Feature Engineering

* Scaling applied to Amount and Time
* Optional log transformation to handle skewness
* Proper train-test split strategy

---

## ⚖️ Imbalance Handling Strategies

Instead of relying on a single technique, four strategies were implemented and compared:

### Strategy 1 — Class Weight (Corporate-Safe Approach)

* Assign higher weight to fraud class
* Stable and production-friendly

### Strategy 2 — SMOTE (Oversampling)

* Synthetic fraud generation
* Improves recall
* Risk of overfitting if not carefully validated

### Strategy 3 — Undersampling

* Reduce majority class
* Faster training
* Loss of information

### Strategy 4 — Scale_Pos_Weight (Boosting-Optimized)

* Used in XGBoost/LightGBM
* Mathematically adjusts gradient updates
* Often superior to SMOTE in fraud problems

---

# 🤖 STEP 4 — Baseline Modeling (Logistic Regression)

Logistic Regression was used as a corporate baseline.

### Evaluation Layers:

1. Classification Report
2. Confusion Matrix
3. Precision-Recall Curve (AUPRC)

### Why AUPRC?

In highly imbalanced datasets:

* Accuracy is misleading
* ROC-AUC may appear strong even with weak fraud detection
* AUPRC directly measures fraud detection quality

---

# 🚀 STEP 5 — Advanced Models (Gradient Boosting)

Goal:

* Improve recall without collapsing precision
* Increase AUPRC significantly
* Handle imbalance more intelligently

## Model 1 — XGBoost

* Used scale_pos_weight
* Tuned for fraud recall
* Strong performance on tabular data

## Model 2 — LightGBM

* Faster boosting alternative
* Competitive performance

### Result:

XGBoost delivered the best balance between recall and precision.

---

# 💰 STEP 6 — Threshold Optimization & Financial Cost Analysis

Default classification threshold (0.5) is not optimal for fraud detection.

Instead, threshold was optimized based on financial cost assumptions:

* False Negative (missed fraud) = $50
* False Positive (false investigation) = $5

### Process:

* Generated probability predictions
* Evaluated multiple thresholds
* Calculated financial loss at each threshold
* Selected optimal threshold minimizing total cost

### Business Impact:

This step transforms the model from a technical solution into a financially optimized system.

---

# 🔍 STEP 7 — Explainability with SHAP

Explainability is essential in financial systems.

Using SHAP:

* Identified top predictive PCA components
* Analyzed contribution of features to fraud predictions
* Verified model behavior consistency

This ensures transparency and regulatory readiness.

---

# 🏆 STEP 8 — Final Model Comparison & Business Recommendation

After comparing models:

| Model               | Strength                       | Weakness              |
| ------------------- | ------------------------------ | --------------------- |
| Logistic Regression | Stable baseline                | Lower recall          |
| Random Forest       | Balanced                       | Slightly less optimal |
| XGBoost             | Best recall + strong precision | Slightly more complex |

### ✅ Final Selection: XGBoost

Reasons:

* Highest AUPRC
* Strong fraud recall
* Controlled false positives
* Best financial performance after threshold tuning

---

# 🏭 Production Deployment Plan

## Recommended Pipeline:

1. Input transaction
2. Apply scaling
3. Generate probability using XGBoost
4. Apply optimized threshold
5. Flag suspicious transaction

## Monitoring Strategy:

* Track fraud rate drift
* Monitor precision & recall weekly
* Re-tune threshold if business costs change
* Periodic retraining with fresh data

---

# 📈 Executive Summary

This project demonstrates a full, real-world fraud detection system built with business alignment in mind.

Key achievements:

✔ Deep imbalance analysis
✔ Multiple strategy comparison
✔ Precision-Recall focused evaluation
✔ Financial threshold optimization
✔ Explainable AI integration
✔ Production-ready design

The selected XGBoost model successfully detects the majority of fraudulent transactions while maintaining controlled false positives.

This ensures reduced financial loss, improved operational efficiency, and scalable deployment readiness.

---

# 🛠 Tech Stack

* Python
* Pandas & NumPy
* Scikit-learn
* XGBoost
* LightGBM
* Matplotlib & Seaborn
* SHAP

---

# 📎 How to Use

1. Clone repository
2. Install dependencies
3. Run notebook sequentially
4. Review threshold optimization section carefully
5. Deploy selected model with optimized threshold

---

If you are building fraud detection systems, this project provides a complete blueprint from data exploration to business-ready deployment.

