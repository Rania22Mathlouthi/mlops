# Declaration des variables
PYTHON=python3
ENV_NAME=mlops_env
REQUIREMENTS=requirements.txt
MAIN_SCRIPT=main.py
DATA_FILE=/mnt/c/Users/RANIA/Downloads/merged_data.csv
TARGET_COLUMN=Churn
MODEL_FILE=model.pkl

# 1. Configuration de l'environnement
setup:
	@echo "Creation de l'environnement virtuel et installation des dependances..."
	@$(PYTHON) -m venv $(ENV_NAME)
	@. $(ENV_NAME)/bin/activate && pip install -r $(REQUIREMENTS)

# 2. Qualite du code, formattage automatique du code, securite du code, etc.
quality:
	@echo "Vérification de la qualité du code, du formatage, de la sécurité, etc."
	@. $(ENV_NAME)/bin/activate && make black
	@. $(ENV_NAME)/bin/activate && make pylint
	@. $(ENV_NAME)/bin/activate && make mypy
	@. $(ENV_NAME)/bin/activate && make bandit

black:
	@echo "Formatage du code avec Black..."
	@. $(ENV_NAME)/bin/activate && black mlops.py main.py

pylint:
	@echo "Analyse statique du code avec Pylint..."
	@. $(ENV_NAME)/bin/activate && pylint mlops.py main.py

mypy:
	@echo "Vérification des types avec MyPy..."
	@. $(ENV_NAME)/bin/activate && mypy --ignore-missing-imports mlops.py main.py

bandit:
	@echo "Analyse de sécurité avec Bandit..."
	@. $(ENV_NAME)/bin/activate && bandit -r mlops.py main.py

# 3. Preparation des donnees
data:
	@echo "Preparation des donnees..."
	@. $(ENV_NAME)/bin/activate && $(PYTHON) mlops.py --data "$(DATA_FILE)" --target $(TARGET_COLUMN) --action prepare

# 4. Entrainement du modele
train:
	@echo "Entrainement du modele..."
	@. $(ENV_NAME)/bin/activate && $(PYTHON) mlops.py --data "$(DATA_FILE)" --target $(TARGET_COLUMN) --model $(MODEL_FILE) --action train
	@echo "Évaluation du modèle..."
	@. $(ENV_NAME)/bin/activate && $(PYTHON) mlops.py --data $(DATA_FILE) --target $(TARGET_COLUMN) --action evaluate --model $(MODEL_FILE)
# 5. Tests unitaires
test:
	@echo "Execution des tests..."
	@. $(ENV_NAME)/bin/activate && pytest test.py

# 6. Deployment
deploy:
	@echo "Déploiement du modèle..."
	@. $(ENV_NAME)/bin/activate && $(PYTHON) $(MAIN_SCRIPT) --data "$(DATA_FILE)" --target $(TARGET_COLUMN) --model $(MODEL_FILE) --action deploy

# 7. Demarrage du serveur Jupyter Notebook
.PHONY: notebook
notebook:
	@echo "Demarrage de Jupyter Notebook..."
	@. $(ENV_NAME)/bin/activate && jupyter notebook

# 8. Nettoyage des fichiers de modele
clean:
	@echo "Nettoyage des fichiers de modele..."
	@rm -f $(MODEL_FILE)

# 9. Supprimer l'environnement virtuel
clean-env:
	@echo "Suppression de l'environnement virtuel..."
	@rm -rf $(ENV_NAME)

#10. fastapi
run_api:
	uvicorn app:app --reload --host 127.0.0.1 --port 8000

# Run all steps
all: setup quality data train test

.PHONY: setup quality black pylint mypy bandit data train test deploy clean clean-env notebook
