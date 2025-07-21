import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

DATASET_DIR = "datasets"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df.dropna(inplace=True)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=['object']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )

    return train_test_split(X, y, test_size=0.2, random_state=42), preprocessor

def train_and_save_model(key, filename):
    print(f"Training model for: {key}")
    filepath = os.path.join(DATASET_DIR, filename)
    (X_train, X_test, y_train, y_test), preprocessor = load_and_prepare_data(filepath)

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"{key} Accuracy: {accuracy:.2f}")

    model_path = os.path.join(MODEL_DIR, f"{key}_model.pkl")
    joblib.dump(model_pipeline, model_path)
    print(f"Saved model to: {model_path}")

def train_all_models():
    dataset_files = {
        "blood_sugar": "bloodsugar.csv",
        "blood_pressure": "bp.csv",
        "diabetes": "diabetes.csv",
        "fever": "fever.csv",
        "heart": "heart.csv",
        "readmission": "readmission.csv"
    }

    for key, filename in dataset_files.items():
        train_and_save_model(key, filename)

if __name__ == "__main__":
    train_all_models()
