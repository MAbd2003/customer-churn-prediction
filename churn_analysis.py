# Customer Churn Prediction
# Author: Muhammad Abdullah
# Tools: Python, Pandas, Scikit-learn, Matplotlib, Seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD DATA ──────────────────────────────────────────
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

print("="*50)
print("CUSTOMER CHURN PREDICTION PROJECT")
print("="*50)
print(f"\nDataset Shape: {df.shape}")
print(f"\nFirst 5 Rows:")
print(df.head())
print(f"\nColumn Names:")
print(df.columns.tolist())
print(f"\nMissing Values:")
print(df.isnull().sum())
print(f"\nChurn Distribution:")
print(df['Churn'].value_counts())
print(f"\nChurn Percentage:")
print(df['Churn'].value_counts(normalize=True) * 100)

# ── 2. DATA CLEANING ──────────────────────────────────────
# TotalCharges should be numeric but has spaces
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Convert Churn to 0/1
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

print("\nData Cleaning Complete!")
print(f"TotalCharges dtype: {df['TotalCharges'].dtype}")

# ── 3. VISUALIZATIONS ─────────────────────────────────────
sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Customer Churn Analysis - Muhammad Abdullah', fontsize=18, fontweight='bold')

# Chart 1 - Churn Distribution
df['Churn'].value_counts().plot(kind='bar', ax=axes[0,0], color=['steelblue','coral'])
axes[0,0].set_title('Churn Distribution')
axes[0,0].set_xticklabels(['No Churn', 'Churned'], rotation=0)
axes[0,0].set_xlabel('')

# Chart 2 - Churn by Contract Type
sns.countplot(x='Contract', hue='Churn', data=df, palette='Set2', ax=axes[0,1])
axes[0,1].set_title('Churn by Contract Type')

# Chart 3 - Churn by Internet Service
sns.countplot(x='InternetService', hue='Churn', data=df, palette='Set1', ax=axes[0,2])
axes[0,2].set_title('Churn by Internet Service')

# Chart 4 - Tenure Distribution
sns.histplot(data=df, x='tenure', hue='Churn', bins=30, ax=axes[1,0], palette='Set2')
axes[1,0].set_title('Tenure vs Churn')

# Chart 5 - Monthly Charges vs Churn
sns.boxplot(x='Churn', y='MonthlyCharges', data=df, palette='Set3', ax=axes[1,1])
axes[1,1].set_title('Monthly Charges vs Churn')
axes[1,1].set_xticklabels(['No Churn', 'Churned'])

# Chart 6 - Senior Citizen vs Churn
sns.countplot(x='SeniorCitizen', hue='Churn', data=df, palette='Set1', ax=axes[1,2])
axes[1,2].set_title('Senior Citizen vs Churn')
axes[1,2].set_xticklabels(['Non-Senior', 'Senior'])

plt.tight_layout()
plt.savefig('churn_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nCharts saved!")


# ── 4. MACHINE LEARNING ───────────────────────────────────
# Encode categorical columns
df_ml = df.copy()
df_ml.drop('customerID', axis=1, inplace=True)
df_ml.fillna(df_ml.median(numeric_only=True), inplace=True)

le = LabelEncoder()
for col in df_ml.select_dtypes(include='object').columns:
    df_ml[col] = le.fit_transform(df_ml[col])

# Features and Target
X = df_ml.drop('Churn', axis=1)
y = df_ml['Churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("\n" + "="*50)
print("MACHINE LEARNING MODELS")
print("="*50)
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# Model 1 - Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

print(f"\n📊 Logistic Regression Accuracy: {lr_acc*100:.2f}%")
print(classification_report(y_test, lr_pred))

# Model 2 - Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print(f"\n🌲 Random Forest Accuracy: {rf_acc*100:.2f}%")
print(classification_report(y_test, rf_pred))

# Winner
print("\n" + "="*50)
if rf_acc > lr_acc:
    print(f"✅ WINNER: Random Forest ({rf_acc*100:.2f}%)")
else:
    print(f"✅ WINNER: Logistic Regression ({lr_acc*100:.2f}%)")
print("="*50)

# Feature Importance
feat_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis')
plt.title('Top 10 Features That Predict Churn - Muhammad Abdullah', fontweight='bold')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFeature importance chart saved!")
print("\n✅ PROJECT COMPLETE!")

# ── 5. INTERACTIVE PREDICTION ─────────────────────────────
print("\n" + "="*50)
print("🔮 CUSTOMER CHURN PREDICTOR")
print("="*50)
print("Enter customer details to predict if they will churn:")

# Example customer - you can change these values!
sample_customer = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 2,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'Yes',
    'StreamingMovies': 'Yes',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 85.0,
    'TotalCharges': 170.0
}

# Convert to dataframe
sample_df = pd.DataFrame([sample_customer])

# Encode same as training data
df_encode = df.drop('Churn', axis=1).copy()
df_encode.drop('customerID', axis=1, inplace=True)

le2 = LabelEncoder()
for col in sample_df.select_dtypes(include='object').columns:
    combined = pd.concat([df_encode[col], sample_df[col]])
    le2.fit(combined)
    sample_df[col] = le2.transform(sample_df[col])

# Predict
prediction = lr.predict(sample_df)[0]
probability = lr.predict_proba(sample_df)[0][1] * 100

print(f"\n📋 Customer Profile:")
print(f"   Contract: Month-to-month")
print(f"   Tenure: 2 months (NEW customer)")
print(f"   Monthly Charges: $85 (HIGH)")
print(f"   Internet: Fiber optic")
print(f"\n🎯 Prediction: {'⚠️ WILL CHURN' if prediction == 1 else '✅ WILL STAY'}")
print(f"📊 Churn Probability: {probability:.1f}%")

if prediction == 1:
    print("\n💡 Business Recommendation:")
    print("   → Offer this customer a discount")
    print("   → Switch them to yearly contract")
    print("   → Assign a retention agent")
else:
    print("\n💡 Business Recommendation:")
    print("   → Customer is loyal, upsell premium services")

print("\n" + "="*50)
print("✅ FULL PROJECT COMPLETE - Muhammad Abdullah")
print("="*50)