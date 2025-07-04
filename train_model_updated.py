import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# -----------------------------
# Load and Preprocess Dataset
# -----------------------------
data = pd.read_csv('more_extended_disease_symptoms.csv')

# Drop non-feature columns to get feature set
X = data.drop(columns=['disease', 'medication', 'nutrition', 'advice'])
y = data['disease']

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Train KNN Model
# -----------------------------
model = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='manhattan')
model.fit(X_scaled, y)

# -----------------------------
# Model Evaluation
# -----------------------------
y_pred = model.predict(X_scaled)

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, average='macro', zero_division=0)
recall = recall_score(y, y_pred, average='macro', zero_division=0)
f1 = f1_score(y, y_pred, average='macro', zero_division=0)

print("Model Evaluation Metrics:")
print(f"Accuracy  : {accuracy:.2f}")
print(f"Precision : {precision:.2f}")
print(f"Recall    : {recall:.2f}")
print(f"F1 Score  : {f1:.2f}")

# -----------------------------
# Save Model, Scaler & Columns
# -----------------------------
joblib.dump(model, 'knn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(list(X.columns), 'symptom_columns.pkl')

print("Model, scaler, and symptom list saved successfully.")
