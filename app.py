import numpy as np
import pandas as pd
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import os
import traceback
import sys

app = Flask(__name__)
CORS(app)

# Global variables for model and scaler
MODEL = None
SCALER = None
ENCODER = None

# Dataset path - using GitHub raw URL
DATASET_PATH = "https://raw.githubusercontent.com/kundu2002/crop-recommendation-app/main/filtered_crop_effects_dataset.csv"

def load_model():
    global MODEL, SCALER, ENCODER
    
    try:
        print("Starting model loading process...")
        print(f"Current Working Directory: {os.getcwd()}")
        print(f"Python Version: {sys.version}")
        print(f"Numpy Version: {np.__version__}")
        print(f"Pandas Version: {pd.__version__}")
        
        # Attempt to download dataset
        try:
            response = requests.get(DATASET_PATH, timeout=30)
            
            # Comprehensive logging of download attempt
            print(f"Request Status Code: {response.status_code}")
            print(f"Response Headers: {response.headers}")
            
            if response.status_code != 200:
                print(f"CRITICAL: Failed to download dataset. Status code: {response.status_code}")
                print(f"Response Content: {response.text}")
                return False
            
            # Save content to a temporary file
            temp_file = "filtered_crop_effects_dataset.csv"
            with open(temp_file, 'wb') as f:
                f.write(response.content)
            
            # Log file details
            print(f"Temp file created: {temp_file}")
            print(f"Temp file size: {os.path.getsize(temp_file)} bytes")
            
            # Load the dataset
            df = pd.read_csv(temp_file)
            
        except Exception as download_error:
            print(f"CRITICAL: Dataset Download Error: {download_error}")
            print(traceback.format_exc())
            return False
        
        # Debugging: Print dataset details
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        print("Columns in dataset:", df.columns.tolist())
        
        # Prepare features
        required_columns = ['N_Effect', 'P_Effect', 'K_Effect', 'pH_Effect']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print("CRITICAL: Missing required columns:", missing_columns)
            print("Actual columns available:", df.columns.tolist())
            return False
            
        X = df[required_columns]
        y = df['Crop']
        
        # Add more validation
        print(f"Features shape: {X.shape}")
        print(f"Unique crop count: {len(y.unique())}")
        print(f"Unique crops: {y.unique()}")
        
        # Initialize and fit scaler
        SCALER = StandardScaler()
        X_scaled = SCALER.fit_transform(X)
        
        # Train model
        MODEL = RandomForestClassifier(n_estimators=100, random_state=42)
        MODEL.fit(X_scaled, y)
        
        # Create label encoder mapping
        unique_crops = sorted(y.unique())
        ENCODER = {idx: crop for idx, crop in enumerate(unique_crops)}
        
        print(f"Model training complete. Unique crops: {unique_crops}")
        print(f"Label encoder created with {len(ENCODER)} crops")
        
        return True
    
    except Exception as e:
        print("CRITICAL: Unexpected error in model loading:")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {str(e)}")
        print("Full Traceback:")
        traceback.print_exc()
        return False

# Rest of the code remains the same as in the previous implementation

@app.route('/', methods=['GET'])
def health_check():
    try:
        return jsonify({
            "status": "healthy",
            "message": "Crop Recommendation API is running",
            "model_loaded": MODEL is not None,
            "scaler_loaded": SCALER is not None,
            "encoder_loaded": ENCODER is not None,
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "sklearn_version": "1.2.2"  # Hardcoded version to avoid potential import issues
        })
    except Exception as e:
        print(f"Error in health check: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    # Load model before starting the server
    if load_model():
        # Get port from environment variable or use 10000 as default
        port = int(os.environ.get('PORT', 10000))
        print(f"Starting server on port {port}")
        app.run(host='0.0.0.0', port=port)
    else:
        print("Failed to load model. Server not started.")
