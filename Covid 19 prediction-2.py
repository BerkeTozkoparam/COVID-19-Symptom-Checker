import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# Load and preprocess the dataset
df = pd.read_csv("Cleaned-Data.csv")
df = df.dropna()  # Remove rows with missing values
df = df.drop("Country", axis=1)  # Drop the 'Country' column as it is not needed

# Create the target variable:
# 1 if any severity symptom is present, 0 otherwise
df["target"] = (df[["Severity_Mild", "Severity_Moderate", "Severity_Severe"]].sum(axis=1) > 0).astype(int)

# Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Store the model’s expected input columns
model_columns = X_train.columns.tolist()

# Streamlit app title
st.title("COVID-19 Symptom Checker")

# Define symptom checkboxes
symptoms = [
    "Fever", "Cough", "Fatigue", "Loss of Taste",
    "Loss of Smell", "Shortness of Breath", "Sore Throat"
]

# Define age groups
ages = ["Age_0-9", "Age_10-19", "Age_20-24", "Age_25-59", "Age_60+"]

# Collect user inputs for symptoms
input_data = {}
for symptom in symptoms:
    input_data[symptom] = int(st.checkbox(symptom))

# User selects their age group
selected_age = st.selectbox("Select your age group:", ages)
for age in ages:
    input_data[age] = 1 if age == selected_age else 0

# Fill missing features with 0 (features not provided by user)
for col in model_columns:
    if col not in input_data:
        input_data[col] = 0

# Create DataFrame with columns ordered as model expects
sample_df = pd.DataFrame([input_data])[model_columns]

# When the user clicks the Predict button
if st.button("Predict"):
    prediction = model.predict(sample_df)[0]
    if prediction == 1:
        st.error("⚠️ Symptoms suggest a possible COVID-19 infection.")
    else:
        st.success("✅ No significant COVID-19 symptoms detected.")
