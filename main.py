"""
Machine Learning Pipeline for Classification Tasks

This module provides a CLI and API interface for training, evaluating, and deploying ML models.
"""

# Standard library imports
import argparse
import sys
import os

# Third-party imports
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import mlflow
from fastapi import FastAPI
import uvicorn

# Local application imports
from mlops import prepare_data, train_model, evaluate_model, deploy, load_model

# Initialize FastAPI app
app = FastAPI()

# MLflow configuration
mlflow.set_tracking_uri("http://127.0.0.1:5000")


@app.post("/predict")
def predict(features: list):
    """Predict the target variable using the trained model.

    Args:
        features (list): Input features for prediction.

    Returns:
        dict: Prediction results in JSON format.
    """
    model = load_model("model.pkl")  # Replace with actual model path
    predictions = model.predict(features)
    return {"predictions": predictions.tolist()}


def main():
    """Main entry point for the ML pipeline CLI.

    Handles command-line arguments for data preparation, model training,
    evaluation, and deployment.
    """
    parser = argparse.ArgumentParser(
        description="Machine Learning Classification Pipeline"
    )
    parser.add_argument("--data", required=True, help="Path to the CSV dataset file.")
    parser.add_argument("--target", required=True, help="Name of the target column.")
    parser.add_argument("--model", required=True, help="Path to save/load the model.")
    parser.add_argument(
        "--action",
        required=True,
        choices=["train", "evaluate", "deploy"],
        help="Action to perform: train, evaluate, or deploy.",
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="Test dataset proportion."
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()

    # Data preparation
    print("Preparing data...")
    x_train, x_test, y_train, y_test = prepare_data(
        filepath=args.data,
        target_column=args.target,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    # Reuse the same MLflow run ID if it exists
    run_id = None
    if os.path.exists("run_id.txt"):
        with open("run_id.txt", "r", encoding="utf-8") as f:
            run_id = f.read().strip()

    # Start or continue the MLflow run
    with mlflow.start_run(run_id=run_id) as run:
        # Save the run ID to a file (only during training)
        if args.action == "train":
            with open("run_id.txt", "w", encoding="utf-8") as f:
                f.write(run.info.run_id)

        if args.action == "train":
            print("Training model...")
            model = AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1, class_weight="balanced"),
                n_estimators=50,
            )
            model = train_model(model, x_train, y_train)
            print(f"Saving model to: {args.model}")

            # Log model parameters
            mlflow.log_param("model_type", type(model).__name__)
            if hasattr(model, "n_estimators"):
                mlflow.log_param("n_estimators", model.n_estimators)

            # Log the trained model
            mlflow.sklearn.log_model(model, "model")

            # Register the model in MLflow Model Registry
            model_uri = f"runs:/{run.info.run_id}/model"
            mlflow.register_model(model_uri, "MyModel")

        elif args.action == "evaluate":
            print(f"Loading model from: {args.model}")
            model = load_model(args.model)
            print("Evaluating model...")
            accuracy, report, _ = evaluate_model(model, x_test, y_test)
            print(f"Accuracy: {accuracy * 100:.2f}%")
            print("\nClassification Report:")
            print(report)

            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", report["weighted avg"]["precision"])
            mlflow.log_metric("recall", report["weighted avg"]["recall"])
            mlflow.log_metric("f1_score", report["weighted avg"]["f1-score"])

            # Log classification report and confusion matrix as artifacts
            mlflow.log_dict(report, "classification_report.json")
            mlflow.log_text(str(_), "confusion_matrix.txt")

        elif args.action == "deploy":
            print(f"Deploying model to MLflow: {args.model}")
            model = load_model(args.model)
            deploy(model, args.model)


if __name__ == "__main__":
    if any(arg.startswith("--action") for arg in sys.argv):
        main()
    else:
        print("Starting FastAPI server...")
        uvicorn.run(app, host="127.0.0.1", port=8000)
