import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_model():
    """
    Loads the trained KNN model, scaler, and symptom column names.
    """
    model = joblib.load("knn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    symptom_columns = joblib.load("symptom_columns.pkl")
    return model, scaler, symptom_columns

def preprocess_symptoms(symptoms, symptom_columns):
    """
    Converts user input symptoms into a binary vector for prediction.
    """
    symptoms = [s.lower().strip() for s in symptoms]
    vector = [1 if col.lower() in symptoms else 0 for col in symptom_columns]
    return vector

def evaluate_model():
    """
    Evaluates the model using the original dataset.
    Returns accuracy, precision, recall, and F1-score.
    """
    try:
        # Load data
        data = pd.read_csv("more_extended_disease_symptoms.csv")
        y = data['disease']
        X = data.drop(columns=['disease'])
        X = X.select_dtypes(include=['int64', 'float64']).fillna(0)

        # Load model and scaler
        model = joblib.load("knn_model.pkl")
        scaler = joblib.load("scaler.pkl")

        # Predict and evaluate
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)

        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)

        return accuracy, precision, recall, f1
    except Exception as e:
        print("Error during model evaluation:", e)
        return 0.0, 0.0, 0.0, 0.0
