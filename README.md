# Credit-Risk-Assessment-Finance-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from google.colab import files
uploaded = files.upload()

df = pd.read_csv("loan_data.csv")

# Preprocess data
df = df.dropna() # Remove missing values

# Convert relevant columns to numeric, coercing errors
df['credit_score'] = pd.to_numeric(df['credit_score'], errors='coerce')
df['person_age'] = pd.to_numeric(df['person_age'], errors='coerce')
df['person_income'] = pd.to_numeric(df['person_income'], errors='coerce')
df['loan_amnt'] = pd.to_numeric(df['loan_amnt'], errors='coerce')
df['loan_percent_income'] = pd.to_numeric(df['loan_percent_income'], errors='coerce')
df['cb_person_cred_hist_length'] = pd.to_numeric(df['cb_person_cred_hist_length'], errors='coerce')

# Drop rows with NaN values that resulted from coercion
df = df.dropna()

# Handle categorical features (one-hot encoding)
categorical_features = ['person_gender', 'person_education', 'person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file']
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)


# Split data into training and testing sets
X = df.drop(['loan_status'], axis=1)
y = df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Use model for prediction
# Create a new applicant DataFrame with dummy variables matching the training data
# Note: You would need to provide values for all relevant features, including the dummy variables
# For simplicity, this example uses some basic features. You will need to adjust this
# based on the actual features used in your training data after one-hot encoding.
# A more robust approach would involve creating a preprocessor pipeline.
new_applicant_data = {
    'person_age': [30],
    'person_income': [50000],
    'credit_score': [700],
    'loan_amnt': [10000],
    'loan_percent_income': [0.2],
    'cb_person_cred_hist_length': [5],
    'person_emp_exp': [5], # Example value
    'person_gender_male': [1], # Example: 1 for male, 0 for female
    'person_education_Master': [0], # Example: Adjust based on education
    'person_education_High School': [0], # Example: Adjust based on education
    'person_education_PhD': [0], # Example: Adjust based on education
    'person_home_ownership_OWN': [0], # Example: Adjust based on home ownership
    'person_home_ownership_RENT': [1], # Example: Adjust based on home ownership
    'person_home_ownership_OTHER': [0], # Example: Adjust based on home ownership
    'loan_intent_MEDICAL': [0], # Example: Adjust based on loan intent
    'loan_intent_VENTURE': [0], # Example: Adjust based on loan intent
    'loan_intent_PERSONAL': [1], # Example: Adjust based on loan intent
    'loan_intent_EDUCATION': [0], # Example: Adjust based on loan intent
    'loan_intent_HOMEIMPROVEMENT': [0], # Example: Adjust based on loan intent
    'previous_loan_defaults_on_file_Yes': [0] # Example: 1 for Yes, 0 for No
}
new_applicant = pd.DataFrame(new_applicant_data)

# Ensure the new applicant DataFrame has the same columns in the same order as the training data
# This is a crucial step for consistent predictions
missing_cols = set(X_train.columns) - set(new_applicant.columns)
for c in missing_cols:
    new_applicant[c] = 0
new_applicant = new_applicant[X_train.columns]


new_applicant_pred = model.predict(new_applicant)
print(f'Predicted Loan Status: {new_applicant_pred}')
