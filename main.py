"""
Machine Learning Pipeline for Classification Tasks

This module provides a CLI and API interface for training, evaluating, and deploying ML models.
"""

# Standard library imports
import argparse

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
    model = load_model("path_to_your_model")  # Remplace par le chemin réel du modèle
    predictions = model.predict([features])  # Mettre entre crochets pour le bon format
    return {"predictions": predictions.tolist()}


def main():
    """Main entry point for the ML pipeline CLI.

    Handles command-line arguments for data preparation, model training,
    evaluation, and deployment.
    """
    parser = argparse.ArgumentParser(
        description="Pipeline de classification Machine Learning"
    )
    parser.add_argument(
        "--data", required=True, help="Chemin vers le fichier de données CSV."
    )
    parser.add_argument("--target", required=True, help="Nom de la colonne cible.")
    parser.add_argument(
        "--model", required=True, help="Chemin pour sauvegarder ou charger le modèle."
    )
    parser.add_argument(
        "--action",
        required=True,
        choices=["train", "evaluate", "deploy"],
        help="Action à effectuer (train/evaluate/deploy)",
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="Taille du jeu de test."
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Graine aléatoire pour la reproductibilité.",
    )

    args = parser.parse_args()

    # Data preparation
    print("Préparation des données...")
    x_train, x_test, y_train, y_test = prepare_data(
        filepath=args.data,
        target_column=args.target,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    if args.action == "train":
        print("Entraînement du modèle...")
        model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, class_weight="balanced"),
            n_estimators=50,
        )
        model = train_model(model, x_train, y_train)
        print(f"Sauvegarde du modèle dans : {args.model}")
        deploy(args.model)

    elif args.action == "evaluate":
        print(f"Chargement du modèle depuis : {args.model}")
        model = load_model(args.model)  # Charger le modèle avant évaluation

        print("Évaluation du modèle...")
        accuracy, report, _ = evaluate_model(model, x_test, y_test)
        print(f"Précision : {accuracy * 100:.2f}%")
        print("\nRapport de classification :")
        print(report)

    elif args.action == "deploy":
        print(f"Chargement du modèle depuis : {args.model}")
        model = load_model(args.model)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
