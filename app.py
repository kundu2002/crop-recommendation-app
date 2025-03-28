import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import json
import os

app = Flask(__name__)
CORS(app)

# Global variables for model and scaler
MODEL = None
SCALER = None
ENCODER = None

# Dataset path
DATASET_PATH = "https://raw.githubusercontent.com/kundu2002/crop-recommendation-app/refs/heads/main/filtered_crop_effects_dataset.csv"

def load_model():
    global MODEL, SCALER, ENCODER
    
    try:
        # Load the dataset
        df = pd.read_csv(DATASET_PATH)
        
        # Prepare features
        X = df[['N', 'P', 'K', 'ph', 'N_P_Ratio', 'P_K_Ratio', 'N_PH_Product']]
        y = df['Crop']
        
        # Initialize and fit scaler
        SCALER = StandardScaler()
        X_scaled = SCALER.fit_transform(X)
        
        # Initialize and train model
        MODEL = RandomForestClassifier(n_estimators=100, random_state=42)
        MODEL.fit(X_scaled, y)
        
        # Create label encoder mapping
        unique_crops = sorted(y.unique())
        ENCODER = {crop: idx for idx, crop in enumerate(unique_crops)}
        ENCODER = {v: k for k, v in ENCODER.items()}  # Reverse mapping
        
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

@app.route('/predict', methods=['POST'])
def predict_crops():
    try:
        # Get input data
        data = request.json
        print("Received input data:", data)  # Debug log
        
        # Get adjustment effects from the crop adjustment section
        n_effect = float(data.get('N_Effect', 0))  # Changed from 'N' to 'N_Effect'
        p_effect = float(data.get('P_Effect', 0))  # Changed from 'P' to 'P_Effect'
        k_effect = float(data.get('K_Effect', 0))  # Changed from 'K' to 'K_Effect'
        ph_effect = float(data.get('PH_Effect', 0))  # Changed from 'ph' to 'PH_Effect'

        print(f"Adjustment effects - N: {n_effect}, P: {p_effect}, K: {k_effect}, pH: {ph_effect}")  # Debug log

        # Feature Engineering for Input
        n_p_ratio = n_effect / (p_effect + 0.1)
        p_k_ratio = p_effect / (k_effect + 0.1)
        n_ph_product = n_effect * ph_effect

        # Create input array
        input_data = np.array([[n_effect, p_effect, k_effect, ph_effect, n_p_ratio, p_k_ratio, n_ph_product]])
        print("Input array shape:", input_data.shape)  # Debug log

        # Scale input
        input_scaled = SCALER.transform(input_data)
        print("Scaled input shape:", input_scaled.shape)  # Debug log

        # Predict crop probabilities
        crop_probs = MODEL.predict_proba(input_scaled)[0]
        print("Raw probabilities:", crop_probs)  # Debug log

        # Get top 2 crops and their probabilities
        top_2_indices = np.argsort(crop_probs)[-2:][::-1]
        top_2_crops = [ENCODER[idx] for idx in top_2_indices]
        top_2_percentages = crop_probs[top_2_indices] * 100

        print(f"Top 2 crops: {top_2_crops}, Percentages: {top_2_percentages}")  # Debug log

        # Prepare response
        results = [
            {"crop": crop, "suitability": round(percent, 2)} 
            for crop, percent in zip(top_2_crops, top_2_percentages)
        ]

        return jsonify({"predictions": results})

    except Exception as e:
        print(f"Error in predict_crops: {str(e)}")  # Debug log
        return jsonify({"error": str(e)}), 400

@app.route('/test', methods=['GET'])
def test_prediction():
    """Test route for manual verification - not visible in the main app"""
    return '''
    <html>
        <head>
            <title>Crop Prediction Test</title>
            <style>
                body { font-family: Arial; padding: 20px; }
                .input-group { margin: 10px 0; }
                input { padding: 5px; margin: 5px; }
                button { padding: 10px; background: #4CAF50; color: white; border: none; cursor: pointer; }
                #result { margin-top: 20px; padding: 10px; border: 1px solid #ccc; }
            </style>
        </head>
        <body>
            <h2>Crop Prediction Test (Adjustment Effects)</h2>
            <div class="input-group">
                <label>N_Effect (Nitrogen Adjustment):</label>
                <input type="number" id="N_Effect" value="20" step="0.1">
                <small>Positive for increase, negative for decrease</small>
            </div>
            <div class="input-group">
                <label>P_Effect (Phosphorus Adjustment):</label>
                <input type="number" id="P_Effect" value="-10" step="0.1">
                <small>Positive for increase, negative for decrease</small>
            </div>
            <div class="input-group">
                <label>K_Effect (Potassium Adjustment):</label>
                <input type="number" id="K_Effect" value="15" step="0.1">
                <small>Positive for increase, negative for decrease</small>
            </div>
            <div class="input-group">
                <label>PH_Effect (pH Adjustment):</label>
                <input type="number" id="PH_Effect" value="-0.5" step="0.1">
                <small>Positive for increase, negative for decrease</small>
            </div>
            <button onclick="testPrediction()">Test Prediction</button>
            <div id="result"></div>

            <script>
                async function testPrediction() {
                    const data = {
                        N_Effect: parseFloat(document.getElementById('N_Effect').value),
                        P_Effect: parseFloat(document.getElementById('P_Effect').value),
                        K_Effect: parseFloat(document.getElementById('K_Effect').value),
                        PH_Effect: parseFloat(document.getElementById('PH_Effect').value)
                    };

                    try {
                        const response = await fetch('/predict', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify(data)
                        });
                        const result = await response.json();
                        document.getElementById('result').innerHTML = 
                            '<pre>' + JSON.stringify(result, null, 2) + '</pre>';
                    } catch (error) {
                        document.getElementById('result').innerHTML = 
                            'Error: ' + error.message;
                    }
                }
            </script>
        </body>
    </html>
    '''

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Crop Recommendation API is running"})

if __name__ == '__main__':
    # Load model before starting the server
    if load_model():
        app.run(host='0.0.0.0', port=10000)
    else:
        print("Failed to load model. Server not started.")
