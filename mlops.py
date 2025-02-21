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

# Module docstring
""" 
MLOps pipeline for model training, evaluation, and deployment with FastAPI.
This script contains functions to load data, preprocess it, train and evaluate models, 
and deploy them using joblib and MLflow.
"""

def load_data(file):
    """
    Loads a CSV file into a pandas DataFrame.

    Args:
        file (str): Path to the CSV file.
    
    Returns:
        pandas.DataFrame: DataFrame containing the data.
    """
    merged_data = pd.read_csv(file)
    return merged_data

def process_data(data, target_column):
    """
    Processes the data by encoding categorical variables and separating
    features and target variables.

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
        test_size (float, optional): Proportion of the data to be used as the test set. Default is 0.2.
        random_state (int, optional): Random seed. Default is 42.

    Returns:
        tuple: Resampled training features, testing features, resampled training target, testing target.
    """
    data = pd.read_csv(filepath)

    x, y = process_data(data, target_column)

    # Split into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # Scale features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

    # Return resampled data
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
        # Log hyperparameters
        mlflow.log_param("model_type", type(model).__name__)

        if hasattr(model, "n_estimators"):
            mlflow.log_param("n_estimators", model.n_estimators)

        # Train the model
        model.fit(x_train, y_train)

        # Log model's tuning parameters (from image)
        mlflow.log_param("layers", "some_value")  # Replace with actual parameter
        mlflow.log_param("alpha", "some_value")  # Replace with actual parameter

        # Log the trained model
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
    """
    Loads a trained model from a file.

    Args:
        filename (str): Path to the model file.

    Returns:
        model: Loaded model.
    """
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
    features = np.array(features).reshape(1, -1)  # Reshape for a single sample
    return model.predict(features)

def deploy(model, model_path):
    """
    Deploys the trained model by saving it to the specified path.