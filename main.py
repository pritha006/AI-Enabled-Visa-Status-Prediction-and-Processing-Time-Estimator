import pandas as pd
df = pd.read_csv("Visadataset.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())
df.fillna(df.mean(numeric_only=True), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])
df['prevailing_wage'] = df['prevailing_wage'] / 24

df.to_csv("cleaned_visa_dataset.csv", index=False)
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
df["company_age"] = 2024 - df["yr_of_estab"]
df["employer_strength"] = df["no_of_employees"] / df["company_age"]
wage_conversion = {
    "Hour": 2080,
    "Week": 52,
    "Month": 12,
    "Bi-Week": 26,
    "Year": 1
}
df["wage_per_year"] = df["prevailing_wage"] * df["unit_of_wage"].map(wage_conversion)
df["high_wage_flag"] = (df["wage_per_year"] > df["wage_per_year"].median()).astype(int)
# Convert to lowercase text first
df["education_of_employee"] = df["education_of_employee"].astype(str)

# Identify STEM degrees by keyword match
# Create an ordered numeric feature for education level
df["education_level"] = df["education_of_employee"].map({
    "High School": 0,
    "Bachelor's": 1,
    "Master's": 2
})

# If there are any unexpected values, fill them with -1
df["education_level"] = df["education_level"].fillna(-1)

# Identify all categorical (object/string) columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Initialize LabelEncoder
le = LabelEncoder()

# Apply label encoding to each categorical column
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

df.to_csv("visa_dataset_milestone2.csv", index=False)
print("Milestone 2 data saved!")






