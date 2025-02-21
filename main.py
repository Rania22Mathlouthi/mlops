from fastapi import FastAPI
import argparse  # Standard
import uvicorn

app = FastAPI()


from sklearn.ensemble import AdaBoostClassifier  # Bibliothèques tierces
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from mlops import (
    prepare_data,
    train_model,
    evaluate_model,
    deploy,
    load_model,
)  # Modules internes

import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

with mlflow.start_run() as run:
    model_uri = f"runs:/{run.info.run_id}/model"
    mlflow.register_model(model_uri, "MyModel")


@app.post("/predict")
def predict(features: list):
    """Predict the target variable using the trained model."""
    model = load_model("path_to_your_model")  # Load your model here
    predictions = model.predict(features)
    return {"predictions": predictions.tolist()}


def main():

    # Configuration des arguments CLI
    parser = argparse.ArgumentParser(
        description="Pipeline de classification Machine Learning"
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Chemin vers le fichier de données CSV.",
    )
    parser.add_argument("--target", required=True, help="Churn")
    parser.add_argument("--model", required=True, help="mlops.py")
    parser.add_argument(
        "--action",
        required=True,
        choices=["train", "evaluate", "deploy"],
        help="Action à effectuer (train/evaluate/deploy)",
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Graine aléatoire pour la reproductibilité",
    )
    args = parser.parse_args()

    # Chargement et préparation des données
    print("Préparation des données...")
    x_train, x_test, y_train, y_test = prepare_data(
        filepath=args.data,
        target_column=args.target,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    if args.action == "train":
        # Entraînement du modèle
        print("Entraînement du modèle...")
        model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, class_weight="balanced"),
            n_estimators=50,
        )
        model = train_model(model, x_train, y_train)

        # Sauvegarde du modèle
        print(f"Sauvegarde du modèle dans : {args.model}")
        deploy(model, args.model)

    elif args.action == "evaluate":
        # Évaluation du modèle
        print("Évaluation du modèle...")
        accuracy, report, conf_matrix = evaluate_model(model, x_test, y_test)

        print(f"Précision : {accuracy * 100:.2f}%")
        print("\nRapport de classification :")
        print(report)
    elif args.action == "deploy":
        # Chargement du modèle
        print(f"Chargement du modèle depuis : {args.model}")
        model = load_model(args.model)


if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)
