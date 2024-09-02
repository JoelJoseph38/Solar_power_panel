import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pickle

# Load data
df = pd.read_excel("C:/Users/kmith/Desktop/DATA SCIENCE/PROJECT 2/mine/GPVS-Faults.xlsx")

# Check for duplicates and missing values
print(df.duplicated().sum())
print(df.isnull().sum())

# Drop near-zero variance columns and time column
variance = df.var()
near_zero_var_features = variance[variance < 0.01]
df = df.drop(columns=near_zero_var_features.index)
df = df.drop(columns='Time')

df.columns

# Splitting features and target variable
X = df.drop(columns='Defective/Non Defective ')
y = df['Defective/Non Defective ']

# Define a pipeline with MinMaxScaler, PCA, and RandomForestClassifier
pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('pca', PCA(n_components=0.95)),
    ('classifier', RandomForestClassifier())
])

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Evaluate and serialize the model
accuracy = pipeline.score(X_test, y_test)
print(f"Random Forest Accuracy: {accuracy}")

# Dump the model into a .pkl file
with open('Random_Forest_model.pkl', 'wb') as file:
    pickle.dump(pipeline, file)

# Load the model from disk for verification
with open('Random_Forest_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

print(loaded_model)
