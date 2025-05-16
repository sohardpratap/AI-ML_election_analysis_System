import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Define the path to the processed dataset
CLEANED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../datasets/cleaned")

# Load the processed winner data
df = pd.read_csv(os.path.join(CLEANED_DIR, "processed_winner_data.csv"))

# Show the first few rows to understand the data
print(df.head())

# Handle missing values (if any)
df = df.dropna()

# Define features (X) and target (y)
X = df[['state', 'year', 'vote_percentage']]
y = df['winner']

# Split the data into training and testing sets (Stratified to handle class imbalance)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Define a preprocessor for one-hot encoding of categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('state_year', OneHotEncoder(), ['state', 'year']),  # One-hot encode 'state' and 'year'
        ('vote_percentage', 'passthrough', ['vote_percentage'])  # Leave vote_percentage as is
    ])

# Initialize the RandomForestClassifier
rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model for future use
import joblib
joblib.dump(rf_model, os.path.join(CLEANED_DIR, "winning_prediction_model.pkl"))
