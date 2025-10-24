# Credit Card Fraud Detection Using Machine Learning

## **Project Overview**

This project detects fraudulent credit card transactions using machine learning. It handles extreme class imbalance with **SMOTE**, scales features for model stability, and evaluates models with **ROC-AUC** and **Precision-Recall metrics**. The pipeline covers **data preprocessing, model training, evaluation, and prediction** for real-time fraud detection.

---

## **Dataset**

* Source: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* Contains **284,807 transactions** with **30 features**:

  * `Time`, `V1`–`V28` (PCA features), `Amount`, and `Class` (target).
* **Class distribution:**

  * Legitimate transactions: 284,315 → 99.83%
  * Fraudulent transactions: 492 → 0.17%

> **Note:** Dataset is highly imbalanced, requiring oversampling techniques like SMOTE.

---

## **Project Structure**

```
Credit-Card-Fraud-Detection/
│
├── Credit_Card_Fraud_Detection.ipynb   # Main Jupyter notebook with full EDA & model pipeline
├── README.md                            # Project overview, instructions, results
├── Requirements.txt                     # Python dependencies
├── .gitignore                            # Ignore dataset, models, cache, etc.
```

---

## **Setup Instructions**

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows
```

3. Install dependencies:

```bash
pip install -r Requirements.txt
```

4. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in a folder, e.g., `data/`.

---

## **Usage**

### **Train & Evaluate Models**

* Open `Credit_Card_Fraud_Detection.ipynb` and run the cells to:

  * Preprocess data
  * Handle class imbalance using SMOTE
  * Train **Logistic Regression** and **Random Forest**
  * Evaluate models with **ROC-AUC** and **Precision-Recall curves**

### **Predict New Transactions**

Use the `predict_transaction` function from the notebook:

```python
preds, proba = predict_transaction("models/final_fraud_model.pkl", X_new, threshold=0.5)
print("Predicted classes:", preds)
print("Fraud probabilities:", proba)
```

* Adjust the **threshold** to trade off precision vs recall.

---

## **Model Performance**

| Model               | ROC-AUC | Precision (fraud) | Recall (fraud) | F1-score (fraud) |
| ------------------- | ------- | ----------------- | -------------- | ---------------- |
| Logistic Regression | 0.9707  | 0.0558            | 0.9184         | 0.1053           |
| Random Forest       | 0.9729  | 0.8495            | 0.8061         | 0.8272           |

> **Inference:** Random Forest with SMOTE provides the best balance between detecting fraud and minimizing false positives.

---

## **Key Takeaways**

1. Handling **class imbalance** is critical; accuracy alone is misleading.
2. **ROC-AUC** and **Precision-Recall** curves are more informative metrics for imbalanced datasets.
3. **Threshold tuning** allows a trade-off between detecting fraud and minimizing false alarms.
4. Saved model can be deployed for **real-time fraud detection**.

---

## **Future Enhancements**

* Deploy as a **web app** or **API** (Streamlit/Flask/FastAPI).
* Include **XGBoost or LightGBM** for potentially better performance.
* Monitor model in production using **drift detection** for evolving fraud patterns.

---
