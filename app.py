import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import os
import urllib.request
import traceback
import requests

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
        print(f"Attempting to load dataset from: {DATASET_PATH}")
        
        # Enhanced URL download with more detailed error handling
        try:
            # Use requests library for more robust download
            response = requests.get(DATASET_PATH, timeout=30)
            
            # Check if request was successful
            if response.status_code != 200:
                print(f"HTTP Error: {response.status_code}")
                print(f"Response Content: {response.text}")
                return False
            
            # Save content to a temporary file
            temp_file = "temp_dataset.csv"
            with open(temp_file, 'wb') as f:
                f.write(response.content)
            
            # Load the dataset
            df = pd.read_csv(temp_file)
            
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
        except requests.exceptions.RequestException as req_error:
            print(f"Network Request Error: {req_error}")
            return False
        except Exception as download_error:
            print(f"Dataset Download Error: {download_error}")
            
            # Fallback: Try urllib method
            try:
                temp_file = "temp_dataset.csv"
                urllib.request.urlretrieve(DATASET_PATH, temp_file)
                df = pd.read_csv(temp_file)
                
                # Clean up temp file
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as urllib_error:
                print(f"Urllib Download Error: {urllib_error}")
                return False
        
        # Debugging: Print dataset details
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        print("Columns in dataset:", df.columns.tolist())
        
        # Prepare features
        required_columns = ['N_Effect', 'P_Effect', 'K_Effect', 'pH_Effect']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print("Missing required columns:", missing_columns)
            print("Actual columns available:", df.columns.tolist())
            return False
            
        X = df[required_columns]
        y = df['Crop']
        
        # Add more validation
        print(f"Features shape: {X.shape}")
        print(f"Unique crop count: {len(y.unique())}")
        
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
        print("Unexpected error in model loading:")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {str(e)}")
        print("Full Traceback:")
        traceback.print_exc()
        return False

@app.route('/predict', methods=['POST'])
def predict_crops():
    try:
        # Get input data
        data = request.json
        print("Received input data:", data)  # Debug log
        
        # Check if model is loaded
        if MODEL is None or SCALER is None:
            return jsonify({"error": "Model not loaded"}), 500
        
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
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 400

@app.route('/test', methods=['GET'])
def test_prediction():
    """Test route for manual verification - not visible in the main app"""
    return '''
    <html>
        <head>
            <title>Crop Prediction Test</title>
            <style>
                body { font-family: Arial; padding: 20px; background-color: #f5f5f5; }
                .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                h2 { color: #388e3c; text-align: center; margin-bottom: 20px; }
                .input-group { margin: 15px 0; }
                label { display: block; margin-bottom: 5px; font-weight: bold; color: #333; }
                input { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 16px; }
                small { display: block; margin-top: 5px; color: #777; }
                button { width: 100%; padding: 12px; background: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; margin-top: 20px; transition: background 0.3s; }
                button:hover { background: #2e7d32; }
                #result { margin-top: 30px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; background: #f9f9f9; }
                pre { white-space: pre-wrap; word-break: break-all; }
                .result-header { font-weight: bold; margin-bottom: 10px; color: #333; }
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Sustainable Crop Prediction Test</h2>
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
            </div>

            <script>
                async function testPrediction() {
                    const data = {
                        N_Effect: parseFloat(document.getElementById('N_Effect').value),
                        P_Effect: parseFloat(document.getElementById('P_Effect').value),
                        K_Effect: parseFloat(document.getElementById('K_Effect').value),
                        PH_Effect: parseFloat(document.getElementById('PH_Effect').value)
                    };

                    document.getElementById('result').innerHTML = 'Loading...';

                    try {
                        const response = await fetch('/predict', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify(data)
                        });
                        const result = await response.json();
                        
                        if (result.error) {
                            document.getElementById('result').innerHTML = 
                                '<div class="result-header">Error:</div><pre>' + result.error + '</pre>';
                        } else {
                            let resultHtml = '<div class="result-header">Prediction Results:</div>';
                            
                            if (result.predictions && result.predictions.length) {
                                resultHtml += '<ul>';
                                result.predictions.forEach(pred => {
                                    resultHtml += `<li><strong>${pred.crop}</strong>: ${pred.suitability}% suitable</li>`;
                                });
                                resultHtml += '</ul>';
                            } else {
                                resultHtml += '<p>No predictions available</p>';
                            }
                            
                            resultHtml += '<pre>' + JSON.stringify(result, null, 2) + '</pre>';
                            document.getElementById('result').innerHTML = resultHtml;
                        }
                    } catch (error) {
                        document.getElementById('result').innerHTML = 
                            '<div class="result-header">Error:</div><pre>' + error.message + '</pre>';
                    }
                }
            </script>
        </body>
    </html>
    '''

@app.route('/', methods=['GET'])
def health_check():
    try:
        return jsonify({
            "status": "healthy",
            "message": "Crop Recommendation API is running",
            "model_loaded": MODEL is not None,
            "scaler_loaded": SCALER is not None
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
