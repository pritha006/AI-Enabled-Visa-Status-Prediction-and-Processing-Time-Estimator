import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Load dataset
df = pd.read_csv("Visadataset.csv")

# Cleaning
df.fillna(df.mean(numeric_only=True), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)

le = LabelEncoder()
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = le.fit_transform(df[col].astype(str))

df["company_age"] = 2024 - df["yr_of_estab"]
df["employer_strength"] = df["no_of_employees"] / df["company_age"]

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.median(), inplace=True)

# Save cleaned dataset
df.to_csv("visa_dataset_milestone2.csv", index=False)
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(7,5))
sns.histplot(df['prevailing_wage'], bins=40, kde=True)
plt.title("Distribution of Processing Time (Days)")
plt.show()
plt.figure(figsize=(7,5))
sns.countplot(x=df['case_status'])
plt.title("Visa Status Distribution")
plt.show()
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
# ---------------- CLASSIFIER ----------------
X_cls = df.drop(columns=["case_status"])
y_cls = df["case_status"]

clf = RandomForestClassifier(n_estimators=300, random_state=42)
clf.fit(X_cls, y_cls)
joblib.dump(clf, "visa_status_model.pkl")

# ---------------- REGRESSOR ----------------
X_reg = df.drop(columns=["prevailing_wage", "case_status", "case_id"])
y_reg = df["prevailing_wage"]

reg = RandomForestRegressor(n_estimators=200, random_state=42)
reg.fit(X_reg, y_reg)
joblib.dump(reg, "processing_time_model.pkl")

print("✅ Models trained & saved successfully")