# 🛡️ Fraud Detection for E-Commerce and Banking Transactions

> **Organization**: Adey Innovations Inc.
> **Challenge**: Improved detection of fraud cases for e-commerce and bank transactions
> **Duration**: 2 Weeks | July 2025
> **Author**: Teshager Admasu

---

## 🚀 Overview

This project focuses on developing machine learning models to detect fraud in both e-commerce and credit card transactions. It tackles the challenge of extreme class imbalance, emphasizes real-time fraud detection, and uses explainable AI (XAI) techniques for model transparency. The models aim to strike a balance between minimizing false positives (protecting user experience) and false negatives (preventing fraud loss).

---

## 🧾 Objectives

- Clean and merge complex transaction datasets
- Engineer features to detect fraudulent patterns
- Handle class imbalance via advanced techniques
- Build, compare, and evaluate fraud detection models
- Explain predictions using SHAP

---

## 🗂️ Dataset Descriptions

### 📦 Fraud_Data.csv
E-commerce transaction records
- `user_id`, `signup_time`, `purchase_time`, `device_id`, `source`, `browser`, `purchase_value`, `ip_address`, `class`
- **Target**: `class` (1 = fraud, 0 = legit)

### 🌍 IpAddress_to_Country.csv
Maps IP ranges to countries
- `lower_bound_ip_address`, `upper_bound_ip_address`, `country`

### 💳 creditcard.csv
Credit card transactions (PCA-transformed)
- `Time`, `V1-V28`, `Amount`, `Class`
- **Target**: `Class` (1 = fraud, 0 = legit)

---

## 🧰 Tech Stack

- **Language**: Python 3.10+
- **Libraries**: pandas, scikit-learn, imbalanced-learn, SHAP, XGBoost, matplotlib, seaborn
- **Environment**: Jupyter / VS Code / Google Colab

---

## 🧪 Project Structure

```
fraud-detection-ecommerce-banking/
├── data/                   # Raw datasets
├── notebooks/              # EDA, modeling, SHAP explainability
├── models/                 # Trained model binaries
├── reports/                # Visuals and final report PDFs
├── utils/                  # Helper scripts (e.g., IP conversion, preprocessing)
├── README.md
├── requirements.txt
└── main.py                 # Main training/evaluation script
```

---

## ⚙️ Setup Instructions

```bash
# Clone the repository
git clone https://github.com/yourusername/fraud-detection-ecommerce-banking.git
cd fraud-detection-ecommerce-banking

# Create and activate environment (optional)
python -m venv venv
source venv/bin/activate  # Or use conda

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python main.py
```

---

## 📊 Key Results

| Model              | Dataset        | F1 Score | AUC-PR | Comment       |
|-------------------|----------------|----------|--------|---------------|
| Logistic Regression | E-commerce    | 0.65     | 0.71   | Baseline      |
| XGBoost             | E-commerce    | 0.84     | 0.90   | Best model    |
| Random Forest       | Credit Card   | 0.86     | 0.92   | Best model    |

> 🚨 Used **SMOTE** and **Random Undersampling** to address class imbalance.
> 🧠 **SHAP** used for feature importance explanation.

---

## 🧠 SHAP Explainability

- **Top Features (E-commerce)**: Time since signup, device_id, purchase_value
- **Top Features (Banking)**: V14, V10, Amount
- Used **SHAP Summary Plot** and **Force Plot** for global & local explanations

---

## 📈 Visuals

- 📌 EDA heatmaps and fraud distribution
- 🧮 Correlation matrices
- 🌍 IP → Country mapping (Geo Analysis)
- 📉 Precision-Recall and ROC Curves
- 📊 SHAP Visualizations

---

## 📎 Deliverables

- ✅ Cleaned datasets
- ✅ EDA & Feature Engineering
- ✅ Trained models (LogReg, XGBoost/Random Forest)
- ✅ Final Report & GitHub Repo
- ✅ SHAP explainability visuals

---

## 🛡️ License

MIT License

---

## 👩🏽‍💻 Author

**Teshager Admasu**
[GitHub](https://github.com/yourusername) • [LinkedIn](https://linkedin.com/in/yourprofile)

---

## ⭐ Acknowledgements

- [Kaggle Datasets](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [Scikit-learn](https://scikit-learn.org)
- [SHAP](https://github.com/slundberg/shap)

---
