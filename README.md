# AI-Enabled-Visa-Status-Prediction-and-Processing-Time-Estimator

# Live App

# https://ai-enabled-visa-status-prediction-and-processing-time-estimato.streamlit.app/

# Objectives

### Predict visa approval status (Classification)

### Estimate visa processing time (Regression)

### Assist applicants using AI-driven insights

# Tech Stack

### Python

### Pandas, NumPy

### Matplotlib, Seaborn

### Scikit-learn

### VS Code
# Architecture 
## Step 1: Data Collection

### Historical visa application data is collected from publicly available or synthetic sources. The dataset contains applicant education, employment details, wage information, and visa case status.

## Step 2: Data Ingestion

### The collected dataset is loaded into the system using Python libraries such as Pandas for further processing and analysis.

## Step 3: Data Preprocessing

### The raw dataset is cleaned by:

### Handling missing values

### Correcting data types

### Removing inconsistencies

### Encoding categorical features into numerical values

### This ensures the dataset is machine-learning ready.

## Step 4: Feature Engineering

### Important features are refined and structured to improve analytical quality, including:

### Education level categorization

### Employment type indicators

### Wage-related attributes

### Region-based features

## Step 5: Exploratory Data Analysis (EDA)

### Statistical analysis and visualizations are performed to:

### Understand data distribution

### Identify trends in visa approvals

### Analyze correlations between features and processing time

## Step 6: Processed Data Storage

### The final cleaned and engineered dataset is saved as a CSV file, which will be used for model training in future milestones.
# Features
## 1️ Structured Data Preparation

### The system prepares a clean and well-structured visa dataset by removing inconsistencies and standardizing formats for further analysis.

## 2️ Missing Value Handling

### Missing or incomplete values in the dataset are automatically identified and handled to ensure data quality and reliability.

## 3️ Categorical Feature Encoding

### Text-based features such as education level, region of employment, and job type are converted into numerical form for machine learning compatibility.

## 4️ Feature Engineering

### Important attributes are refined to improve analytical performance, including:

### Education level categorization

### Employment and experience indicators

### Wage-related attributes

## 5️ Exploratory Data Analysis (EDA)

### The system performs detailed data analysis to understand:

### Visa approval and rejection trends

### Distribution of processing times

### Relationships between features

## 6️ Correlation Analysis

### Statistical correlation analysis is used to identify relationships between input features and visa processing outcomes.

## 7️ Visualization Support

### Visual representations such as bar charts and correlation plots are generated to simplify data interpretation.

## 8️ Processed Dataset Export

### The final transformed dataset is saved as a CSV file, ready for machine learning model training in subsequent milestones.

# Problem Statement

### Many visa applicants face uncertainty regarding visa approval outcomes and processing durations. Manual evaluation of applications is time-consuming and prone to inconsistencies. This project leverages Machine Learning techniques to predict visa approval status and estimate processing time, helping applicants and organizations make informed decisions.

# Machine Learning Models Used

## Classification Model:

### Logistic Regression / Random Forest Classifier

### Used to predict visa approval status (Approved / Denied)

## Regression Model:

### Linear Regression / Random Forest Regressor

### Used to estimate visa processing time (in days)

# Model Training Pipeline
### Data preprocessing
### Feature scaling and encoding
### Train-test split
### Model training and validation
### Model persistence using .pkl files

# Application Workflow 

### 1 User inputs visa application details
### 2 Data is preprocessed in real-time
### 3 Trained ML models generate predictions
### 4️ Results are displayed via Streamlit UI
### 5️ Visual insights support decision-making

# User Interface (UI)
### Built using Streamlit with:

### Interactive input fields

### Real-time prediction output

### Clean and responsive design

# Deployment
### Platform: Streamlit Cloud
### Repository: GitHub
### Deployment Type: Cloud-based web application
### Live URL:

## https://ai-enabled-visa-status-prediction-and-processing-time-estimato.streamlit.app/

