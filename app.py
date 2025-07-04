from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import pandas as pd
import numpy as np
from flask_cors import CORS
from model import evaluate_model, preprocess_symptoms, load_model

app = Flask(__name__)
app.secret_key = "your_secret_key"
CORS(app)

# ------------------ Load model and disease info ------------------
model, scaler, symptom_columns = load_model()
accuracy, precision, recall, F1_Score = evaluate_model()

# Load disease information
info_file = "more_extended_disease_symptoms.csv"
disease_data = pd.read_csv(info_file)
disease_data['disease'] = disease_data['disease'].str.strip().str.lower()
disease_info = disease_data.drop_duplicates(subset='disease')[['disease', 'medication', 'advice', 'nutrition']]
disease_info.set_index('disease', inplace=True)

# ------------------ ROUTES ------------------

# Login Page (GET & POST)
@app.route('/', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        gender = request.form.get('gender')
        age = request.form.get('age')

        # Store user info in session
        session['user'] = name
        session['email'] = email
        return redirect(url_for('symptom_checker', name=name, email=email, gender=gender, age=age))

    return render_template('login.html')

# Main Symptom Checker Page
@app.route('/symptom-checker', methods=['GET'])
def symptom_checker():
    if 'user' not in session:
        return redirect(url_for('login_page'))
    return render_template('index1.html')

# Predict Disease (POST)
@app.route('/predict-disease', methods=['POST'])
def predict_disease():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        data = request.get_json()
        symptoms = data.get("symptoms", [])

        # Validate symptom input
        if not symptoms:
            return jsonify({"error": "No symptoms provided"}), 400

        # Clean and validate against known symptom columns
        cleaned_symptoms = [s.strip().lower() for s in symptoms if isinstance(s, str)]
        valid_symptoms = [s for s in cleaned_symptoms if s in symptom_columns]

        if not valid_symptoms:
            return jsonify({
                "error": "No valid symptoms provided.",
                "message": f"Please enter at least one valid symptom like: {', '.join(symptom_columns[:10])}..."
            }), 400

        # Create input vector
        input_vector = preprocess_symptoms(valid_symptoms, symptom_columns)
        input_df = pd.DataFrame([input_vector], columns=symptom_columns)
        input_scaled = scaler.transform(input_df)

        # Predict top 2 diseases
        probabilities = model.predict_proba(input_scaled)[0]
        top_indices = np.argsort(probabilities)[-2:][::-1]
        top_diseases = model.classes_[top_indices]
        top_probs = probabilities[top_indices]

        # Gather disease info
        predicted_diseases = []
        medication_list = []
        advice_list = []
        nutrition_list = []

        for disease in top_diseases:
            disease_key = disease.strip().lower()
            if disease_key in disease_info.index:
                row = disease_info.loc[disease_key]
                predicted_diseases.append(disease)
                medication_list.append(row['medication'] if pd.notna(row['medication']) else "No medication listed")
                advice_list.append(row['advice'] if pd.notna(row['advice']) else "No advice available")
                nutrition_list.append(row['nutrition'] if pd.notna(row['nutrition']) else "No nutrition info")
            else:
                predicted_diseases.append(disease)
                medication_list.append("No info")
                advice_list.append("No info")
                nutrition_list.append("No info")

        return jsonify({
            "diseases": predicted_diseases,
            "probabilities": top_probs.tolist(),
            "medications": medication_list,
            "advice": advice_list,
            "nutrition": nutrition_list
        })

    except Exception as e:
        print("❌ Error in /predict-disease:", e)
        return jsonify({"error": "Internal server error"}), 500

# Logout Route
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login_page'))

# ------------------ RUN APP ------------------
if __name__ == '__main__':
    print("🚀 Server running at: http://127.0.0.1:5000")
    app.run(debug=True)
