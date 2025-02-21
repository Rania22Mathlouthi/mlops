import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

from fastapi import FastAPI
import joblib
import mlflow

app = FastAPI()


def load_data(file):
    merged_data = pd.read_csv(file)
    return merged_data


def process_data(data, target_column):
    # Encode categorical variables
    label_encoder = LabelEncoder()
    categorical_columns = data.select_dtypes(include=["object", "category"]).columns
    for col in categorical_columns:
        data[col] = label_encoder.fit_transform(data[col])

    features = data.drop(target_column, axis=1)
    target = data[target_column]
    return features, target


def prepare_data(filepath, target_column, test_size=0.2, random_state=42):
    """Loads, cleans, and splits the data into training and testing sets."""
    data = pd.read_csv(filepath)

    X, y = process_data(data, target_column)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Return resampled data
    return X_train_res, X_test, y_train_res, y_test


def train_model(model, X_train, y_train):
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_param("model_type", type(model).__name__)

        if hasattr(model, "n_estimators"):
            mlflow.log_param("n_estimators", model.n_estimators)

        # Train the model
        model.fit(X_train, y_train)

        # Log model's tuning parameters (from image)
        mlflow.log_param("layers", "some_value")  # Replace with actual parameter
        mlflow.log_param("alpha", "some_value")  # Replace with actual parameter

        # Log the trained model
        mlflow.sklearn.log_model(model, "model")

    return model


def evaluate_model(model, X_test, y_test):
    with mlflow.start_run():
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Log model's metrics (from the image)
        mlflow.log_metric("mse", 0)  # Replace 0 with actual MSE if applicable
        mlflow.log_artifact("plot", "some_plot_path")  # Replace with actual plot file
        mlflow.sklearn.log_model(model, "model")  # Log the model

        print(f"Model accuracy: {accuracy:.4f}")

    return accuracy, report, conf_matrix  # Keep all three returns


def load_model(filename):
    """Loads a trained model from a file."""
    return joblib.load(filename)


def predict(model, features):
    """Predict the target variable using the trained model."""
    # Reshape features if it's a single sample (1D array)
    features = np.array(features).reshape(1, -1)  # Reshape for a single sample
    return model.predict(features)


def deploy(model, model_path):

    # Save the model to the specified path
    joblib.dump(model, model_path)
    print(f"Model deployed and saved to: {model_path}")
