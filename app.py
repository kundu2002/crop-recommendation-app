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

# Dataset path - using local file
DATASET_PATH = "assets/data/filtered_crop_effects_dataset.csv"

def load_model():
    global MODEL, SCALER, ENCODER
    
    try:
        print("Starting model loading process...")
        print(f"Attempting to load dataset from: {DATASET_PATH}")
        
        # Check if file exists
        if not os.path.exists(DATASET_PATH):
            print(f"Error: Dataset file not found at {DATASET_PATH}")
            return False
            
        # Load the dataset
        df = pd.read_csv(DATASET_PATH)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        print("Columns in dataset:", df.columns.tolist())
        
        # Prepare features
        required_columns = ['N_Effect', 'P_Effect', 'K_Effect', 'pH_Effect']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print("Missing required columns:", missing_columns)
            return False
            
        X = df[required_columns]
        y = df['Crop']
        
        print("Features prepared successfully")
        
        # Initialize and fit scaler
        SCALER = StandardScaler()
        X_scaled = SCALER.fit_transform(X)
        print("Scaler fitted successfully")
        
        # Initialize and train model
        MODEL = RandomForestClassifier(n_estimators=100, random_state=42)
        MODEL.fit(X_scaled, y)
        print("Model trained successfully")
        
        # Create label encoder mapping
        unique_crops = sorted(y.unique())
        ENCODER = {crop: idx for idx, crop in enumerate(unique_crops)}
        ENCODER = {v: k for k, v in ENCODER.items()}  # Reverse mapping
        print(f"Label encoder created with {len(ENCODER)} crops")
        
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())
        return False

@app.route('/predict', methods=['POST'])
def predict_crops():
    try:
        # Get input data
        data = request.json
        print("Received input data:", data)  # Debug log
        
        # Get adjustment effects from the crop adjustment section
        n_effect = float(data.get('N_Effect', 0))
        p_effect = float(data.get('P_Effect', 0))
        k_effect = float(data.get('K_Effect', 0))
        ph_effect = float(data.get('PH_Effect', 0))

        print(f"Adjustment effects - N: {n_effect}, P: {p_effect}, K: {k_effect}, pH: {ph_effect}")  # Debug log

        # Create input array (no need for additional feature engineering since we're using effects directly)
        input_data = np.array([[n_effect, p_effect, k_effect, ph_effect]])
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
