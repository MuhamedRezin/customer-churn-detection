🧠 Customer Churn Detection using Machine Learning
📋 Project Overview

This project predicts whether a telecom customer is likely to churn (discontinue the service).
It uses machine learning models to analyze customer data, identify churn patterns, and help businesses take preventive action.

🎯 Objective

To build a predictive model that classifies customers as Churn or No Churn based on their service usage and account details.

🧩 Dataset

Source: Telco Customer Churn dataset
Rows: 7043
Features: 21
Target: Churn (Yes / No)

Each record represents a customer’s profile — including demographic data, account information, and service subscriptions.

Key Columns:

tenure – Duration of the customer’s subscription

MonthlyCharges – Monthly billing amount

TotalCharges – Lifetime billing amount

InternetService, Contract, PaymentMethod – Service attributes

Churn – Target variable (Yes → 1, No → 0)

⚙️ Data Preprocessing

Missing values handled: Replaced blank TotalCharges with 0.0.

Type conversion: Converted TotalCharges to float.

Label Encoding: Encoded categorical columns using LabelEncoder and saved encoders via pickle.

Balancing Data: Applied SMOTE (Synthetic Minority Oversampling Technique) to fix class imbalance.

Original: 4138 (No) / 1496 (Yes)

After SMOTE: 4138 (No) / 4138 (Yes)

🧠 Model Training

Three models were trained and evaluated using 5-fold cross-validation:

Model	Cross-Validation Accuracy
Decision Tree	0.78
Random Forest	0.84
XGBoost	0.83

The Random Forest Classifier gave the best overall performance.

📊 Model Evaluation

Test Accuracy: 0.78
Confusion Matrix:

[[878 158]
 [154 219]]


Classification Report:

Metric	Class 0 (No Churn)	Class 1 (Churn)
Precision	0.85	0.58
Recall	0.85	0.59
F1-Score	0.85	0.58
💾 Model Export

Trained Random Forest model saved as: customer_churn_model.pkl

Encoders saved as: encoders.pkl

Both can be loaded for future predictions.

🔮 Sample Prediction

Input Example:

{
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 1,
    'PhoneService': 'No',
    'MultipleLines': 'No phone service',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 29.85,
    'TotalCharges': 29.85
}


Prediction: No Churn
Probability: 79% No Churn | 21% Churn

🧩 Tech Stack

Python

Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn, xgboost

Model Persistence: pickle

🚀 How to Run
# Clone repository
git clone <your-repo-url>
cd customer-churn-detection

# Install dependencies
pip install -r requirements.txt

# Run the script
python main.py

📈 Insights

Customers on month-to-month contracts with high monthly charges show higher churn risk.

Longer tenure correlates with customer retention.

SMOTE improved minority class recall.

🔍 Future Scope

Add hyperparameter tuning for each model.

Implement a web UI for live predictions.

Integrate SHAP or LIME for model interpretability.

👨‍💻 Author

Rezin
BCA Student | Data & AI Enthusiast