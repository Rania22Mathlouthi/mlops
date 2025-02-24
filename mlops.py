"""
MLOps pipeline for model training, evaluation, and deployment using FastAPI.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from fastapi import FastAPI
import joblib
import mlflow

# Initialize FastAPI app
app = FastAPI()

# MLflow configuration
mlflow.set_tracking_uri("http://127.0.0.1:5000")


def load_data(file):
    """
    Loads a CSV file into a pandas DataFrame.

    Args:
        file (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: DataFrame containing the data.
    """
    return pd.read_csv(file)


def process_data(data, target_column):
    """
    Processes the data by encoding categorical variables and separating features and target.

    Args:
        data (pandas.DataFrame): Input data.
        target_column (str): Name of the target column.

    Returns:
        tuple: Features and target data (X, y).
    """
    label_encoder = LabelEncoder()
    categorical_columns = data.select_dtypes(include=["object", "category"]).columns
    for col in categorical_columns:
        if col != target_column:
            data[col] = label_encoder.fit_transform(data[col])
    features = data.drop(target_column, axis=1)
    target = data[target_column]
    return features, target


def prepare_data(filepath, target_column, test_size=0.2, random_state=42):
    """
    Loads, cleans, and splits the data into training and testing sets.

    Args:
        filepath (str): Path to the dataset.
        target_column (str): Name of the target column.
        test_size (float, optional): Proportion of test data. Default is 0.2.
        random_state (int, optional): Random seed. Default is 42.

    Returns:
        tuple: Resampled training features, testing features,
        resampled training target, testing target.
    """
    data = load_data(filepath)
    x, y = process_data(data, target_column)

    # Convert y to numpy array
    y = y.values

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train.values)  # Convert DataFrame to numpy array
    x_test = scaler.transform(x_test.values)  # Convert DataFrame to numpy array

    smote = SMOTE(random_state=42)
    x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

    return x_train_res, x_test, y_train_res, y_test


def train_model(model, x_train, y_train):
    """
    Trains the model, logs parameters, and registers the model in MLflow Model Registry.

    Args:
        model: The model to be trained.
        x_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target.

    Returns:
        model: The trained model.
    """

    # Train the model
    model.fit(x_train, y_train)

    return model


def evaluate_model(model, x_test, y_test):
    """
    Evaluates the trained model and logs the results.
    """
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    print("\nRapport de classification :")
    print(report)
    print("\nMatrice de confusion :")
    print(conf_matrix)
    return accuracy, report, conf_matrix


def load_model(filename):
    """
    Loads a trained model from a file.

    Args:
        filename (str): Path to the model file.

    Returns:
        model: Loaded model.
    """
    return joblib.load(filename)  # Changed from dump() to load()


def deploy(model, model_path):
    """
    Deploys the model to MLflow.

    Args:
        model: Trained model.
        model_path (str): Path to save the model in MLflow.
    """
    # Log the model with MLflow (recommended)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=model_path,
        registered_model_name="MyModel",  # Optional: register in Model Registry
    )

    # Alternative: Save model to local path
    # mlflow.sklearn.save_model(model, model_path)

    print(f"Model deployed to MLflow at: {model_path}")


def predict(model, features):
    """
    Predicts the target variable using the trained model.

    Args:
        model: The trained model.
        features (numpy.ndarray): Input features for prediction.

    Returns:
        numpy.ndarray: Predicted values.
    """
    features = np.array(features).reshape(1, -1)
    return model.predict(features)
