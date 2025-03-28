import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

def prepare_crop_prediction_model():
    # 🟢 Load dataset
    df = pd.read_csv("for_deployment/filtered_crop_effects_dataset.csv")

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
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # ⚖️ Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
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
    rf_model.fit(X_train_scaled, y_train)

    # Model Evaluation
    y_pred_rf = rf_model.predict(X_test_scaled)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    print(f"🌳 Random Forest Accuracy: {rf_accuracy:.2f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred_rf, target_names=encoder.classes_))

    return rf_model, scaler, encoder

def predict_top_crops(model, scaler, encoder, n_effect, p_effect, k_effect, ph_effect):
    """Predict top 2 crops with suitability"""
    # Feature Engineering for Input
    n_p_ratio = n_effect / (p_effect + 0.1)
    p_k_ratio = p_effect / (k_effect + 0.1)
    n_ph_product = n_effect * ph_effect

    # Create input array
    input_data = np.array([[n_effect, p_effect, k_effect, ph_effect, n_p_ratio, p_k_ratio, n_ph_product]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict crop probabilities
    crop_probs = model.predict_proba(input_scaled)[0]

    # Get top 2 crops and their probabilities
    top_2_indices = np.argsort(crop_probs)[-2:][::-1]
    top_2_crops = encoder.inverse_transform(top_2_indices)
    top_2_percentages = crop_probs[top_2_indices] * 100

    # Display results
    results = [(crop, round(percent, 2)) for crop, percent in zip(top_2_crops, top_2_percentages)]

    return results

def user_input_loop(model, scaler, encoder):
    """Interactive user input loop for crop prediction"""
    print("🌾 Crop Prediction Tool 🌾")
    print("Enter soil effect values (type 'exit' to quit)")
    
    while True:
        try:
            print("\nEnter soil effect values:")
            n_effect = float(input("Nitrogen Effect (-100 to 100): "))
            p_effect = float(input("Phosphorus Effect (-100 to 100): "))
            k_effect = float(input("Potassium Effect (-100 to 100): "))
            ph_effect = float(input("pH Effect (-100 to 100): "))

            top_crops = predict_top_crops(model, scaler, encoder, n_effect, p_effect, k_effect, ph_effect)
            
            print("\n🌾 Top 2 Recommended Crops with Suitability:")
            for crop, percentage in top_crops:
                print(f"✅ {crop}: {percentage}%")
        
        except ValueError as e:
            print("Invalid input. Please enter numeric values.")
            
        exit_choice = input("\nDo you want to continue? (press Enter to continue, type 'exit' to quit): ")
        if exit_choice.lower() == 'exit':
            break

def main():
    # Prepare the model
    model, scaler, encoder = prepare_crop_prediction_model()
    
    # Start user input loop
    user_input_loop(model, scaler, encoder)

if __name__ == "__main__":
    main()