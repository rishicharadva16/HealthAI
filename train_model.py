import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def train():
    print("Loading data...")
    df = pd.read_csv("data/Diseases_and_Symptoms_data.csv")
    
    # Prepare X (features) and y (target)
    X = df.drop("diseases", axis=1)
    y = df["diseases"]
    
    # Encoder
    print("Encoding labels...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    # Save Encoder
    joblib.dump(le, "label_encoder.pkl")
    print("Label Encoder saved.")
    
    best_model = Pipeline([
        ("classifier", MultinomialNB(alpha=1.0)),
    ])

    print("Training multinomial_nb_pipeline...")
    best_model.fit(X_train, y_train)
    best_name = "multinomial_nb_pipeline"
    y_pred = best_model.predict(X_test)
    best_accuracy = accuracy_score(y_test, y_pred)
    best_f1_score = f1_score(y_test, y_pred, average="weighted")
    best_matrix = confusion_matrix(y_test, y_pred)

    print(f"{best_name} accuracy: {best_accuracy:.4f}")
    print(f"{best_name} f1_score: {best_f1_score:.4f}")
    print(f"{best_name} confusion matrix:\n{best_matrix}")
    print(f"\nBest model: {best_name} (weighted F1={best_f1_score:.4f})")

    # Save model with compression to keep the artifact small.
    joblib.dump(best_model, "disease_model.pkl", compress=3)
    print("Model saved successfully.")
    print(f"Compressed model size: {Path('disease_model.pkl').stat().st_size} bytes")

    # Save test metrics for later reference.
    metrics_path = Path("model_metrics.json")
    matrix_path = Path("confusion_matrix.csv")

    pd.DataFrame(best_matrix).to_csv(matrix_path, index=False, header=False)
    pd.Series({
        "best_model": best_name,
        "accuracy": float(best_accuracy),
        "f1_score": float(best_f1_score),
    }).to_json(metrics_path, indent=2)

    print(f"\nSaved metrics to {metrics_path}")
    print(f"Saved confusion matrix to {matrix_path}")

    # Test Prediction
    print("\nVerifying model...")
    test_symptom = "pain_chest" # Try to find a valid column
    if test_symptom not in X.columns:
        test_symptom = X.columns[0]

    print(f"Testing with symptom: {test_symptom}")
    input_vector = pd.DataFrame([np.zeros(len(X.columns))], columns=X.columns)
    idx = X.columns.get_loc(test_symptom)
    input_vector.iloc[0, idx] = 1

    pred = best_model.predict(input_vector)
    pred_name = le.inverse_transform(pred)[0]
    print(f"Prediction: {pred_name}")

if __name__ == "__main__":
    train()
