import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv('more_extended_disease_symptoms.csv')

# Ensure 'disease' column exists
if 'disease' not in data.columns:
    raise ValueError("Dataset must contain a 'disease' column.")

# Separate features and labels
y = data['disease']

# Select only numeric columns as features
X = data.drop(columns=['disease'])
X = X.select_dtypes(include=['int64', 'float64'])

# Handle missing values
X = X.fillna(0)

# Save the column names for later use in model.py
symptom_columns = X.columns.tolist()

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KNN model
model = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='manhattan')
model.fit(X_scaled, y)

# Evaluate model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained successfully with accuracy: {accuracy * 100:.2f}%")

# Save model, scaler, and symptom columns
joblib.dump(model, 'knn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(symptom_columns, 'symptom_columns.pkl')
