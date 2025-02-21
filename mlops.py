import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

from fastapi import FastAPI
import joblib
import mlflow

app = FastAPI()

"""
MLOps pipeline for model training, evaluation, and deployment with FastAPI.
This script contains functions to load data, preprocess it, train and evaluate models, 
and deploy them using joblib and MLflow.
"""


def load_data(file):
    """Loads a CSV file into a pandas DataFrame."""
    return pd.read_csv(file)


def process_data(data, target_column):
    """
    Encodes categorical variables and separates features and target variables.

    Args:
        data (pandas.DataFrame): Input data.
        target_column (str): Name of the target column.

    Returns:
        tuple: Features and target data (X, y).
    """
    label_encoder = LabelEncoder()
    categorical_columns = data.select_dtypes(include=["object", "category"]).columns
    for col in categorical_columns:
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
        test_size (float, optional): Proportion of the data for testing. Default is 0.2.
        random_state (int, optional): Random seed. Default is 42.

    Returns:
        tuple: Resampled training features, testing features, resampled training target, testing target.
    """
    data = pd.read_csv(filepath)
    x, y = process_data(data, target_column)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    smote = SMOTE(random_state=42)
    x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

    return x_train_res, x_test, y_train_res, y_test


def train_model(model, x_train, y_train):
    """
    Trains the model and logs the parameters using MLflow.

    Args:
        model: The model to be trained.
        x_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target.

    Returns:
        model: The trained model.
    """
    with mlflow.start_run():
        mlflow.log_param("model_type", type(model).__name__)
        if hasattr(model, "n_estimators"):
            mlflow.log_param("n_estimators", model.n_estimators)

        model.fit(x_train, y_train)

        mlflow.log_param("layers", "some_value")  # Replace with actual parameter
        mlflow.log_param("alpha", "some_value")  # Replace with actual parameter

        mlflow.sklearn.log_model(model, "model")

    return model


def evaluate_model(model, x_test, y_test):
    """
    Evaluates the trained model using various metrics and logs the results.

    Args:
        model: The trained model.
        x_test (numpy.ndarray): Test features.
        y_test (numpy.ndarray): Test target.

    Returns:
        tuple: Accuracy, classification report, confusion matrix.
    """
    with mlflow.start_run():
        y_pred = model.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)

        mlflow.log_metric("mse", 0)  # Replace 0 with actual MSE if applicable
        mlflow.log_artifact("plot", "some_plot_path")  # Replace with actual plot file
        mlflow.sklearn.log_model(model, "model")

        print(f"Model accuracy: {accuracy:.4f}")

    return accuracy, report, conf_matrix


def load_model(filename):
    """Loads a trained model from a file."""
    return joblib.load(filename)


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


def deploy(_model, _model_path):
    """Placeholder function for deploying the model. Implementation needed."""
    pass
