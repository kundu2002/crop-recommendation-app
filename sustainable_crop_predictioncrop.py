import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

def prepare_crop_prediction_model(dataset_path):
    # 🟢 Load dataset from GitHub raw URL
    df = pd.read_csv(dataset_path)

    # 🛠 Feature Engineering (Adding New Features)
    df["N_P_Ratio"] = df["N_Effect"] / (df["P_Effect"] + 0.1)
    df["P_K_Ratio"] = df["P_Effect"] / (df["K_Effect"] + 0.1)
    df["N_pH_Product"] = df["N_Effect"] * df["pH_Effect"]

    # 🎯 Features & Target
    X = df.drop(columns=["Crop"])
    y = df["Crop"]

    # 🏷 Encode the target variable
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # ✂️ Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # 🟢 Apply SMOTE for class balancing
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # ⚖️ Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_smote)
    X_test_scaled = scaler.transform(X_test)

    # 🌳 Train Random Forest Classifier
    rf_model = RandomForestClassifier(
        n_estimators=300, 
        max_depth=None, 
        min_samples_split=2, 
        min_samples_leaf=1, 
        bootstrap=True, 
        random_state=42, 
        class_weight="balanced"
    )
    rf_model.fit(X_train_scaled, y_train_smote)

    # Model Evaluation
    y_pred_rf = rf_model.predict(X_test_scaled)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    print(f"🌳 Random Forest Accuracy: {rf_accuracy:.2f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred_rf, target_names=encoder.classes_))

    return rf_model, scaler, encoder

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# GitHub raw dataset URL (replace with your actual raw GitHub URL)
DATASET_PATH = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/filtered_crop_effects_dataset.csv"

# Prepare Model
MODEL, SCALER, ENCODER = prepare_crop_prediction_model(DATASET_PATH)

@app.route('/predict', methods=['POST'])
def predict_crops():
    try:
        # Get input data
        data = request.json
        n_effect = float(data.get('n_effect', 0))
        p_effect = float(data.get('p_effect', 0))
        k_effect = float(data.get('k_effect', 0))
        ph_effect = float(data.get('ph_effect', 0))

        # Feature Engineering for Input
        n_p_ratio = n_effect / (p_effect + 0.1)
        p_k_ratio = p_effect / (k_effect + 0.1)
        n_ph_product = n_effect * ph_effect

        # Create input array
        input_data = np.array([[n_effect, p_effect, k_effect, ph_effect, n_p_ratio, p_k_ratio, n_ph_product]])

        # Scale input
        input_scaled = SCALER.transform(input_data)

        # Predict crop probabilities
        crop_probs = MODEL.predict_proba(input_scaled)[0]

        # Get top 2 crops and their probabilities
        top_2_indices = np.argsort(crop_probs)[-2:][::-1]
        top_2_crops = ENCODER.inverse_transform(top_2_indices)
        top_2_percentages = crop_probs[top_2_indices] * 100

        # Prepare response
        results = [
            {"crop": crop, "suitability": round(percent, 2)} 
            for crop, percent in zip(top_2_crops, top_2_percentages)
        ]

        return jsonify({"predictions": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Crop Recommendation API is running"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)