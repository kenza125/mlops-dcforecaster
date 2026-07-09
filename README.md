DCForecaster — Prévision de la demande de recharge EV

Pipeline MLOps complet autour d'un modèle de prévision de la demande de recharge de véhicules électriques (arrivées, départs, énergie consommée) à une résolution de 15 minutes.

Le modèle repose sur une architecture à deux étages (classification + régression) implémentée avec XGBoost. Ce dépôt ne se limite pas au modèle : il couvre l'ensemble du cycle de vie MLOps — versioning, service d'inférence, tests automatisés, CI/CD, et monitoring (usage + dérive des données).

Fonctionnalités


Prévision multi-pas de l'occupation de bornes de recharge (nombre d'arrivées, de départs, énergie moyenne, durée moyenne) sur un horizon configurable.
API REST (FastAPI) exposant le modèle via un endpoint /predict, avec upload de CSV.
Versioning et tracking des runs et des modèles avec MLflow (paramètres, métriques, Model Registry, alias de production).
Tests automatisés de l'API (modèle mocké, pas besoin des artefacts réels pour lancer les tests).
CI/CD avec GitHub Actions : tests à chaque push, puis construction de l'image Docker.
Monitoring d'usage de l'API à partir des logs de prédiction.
Monitoring de dérive des données avec Evidently AI, sur les variables d'entrée du modèle.


Structure du projet

mlops-dcforecaster/
├── predictor.py              # Logique d'inférence du modèle (implémentation privée)
├── models/dc_v2/              # Artefacts du modèle entraîné
├── api.py                    # API FastAPI (POST /predict)
├── predict.py                 # Prédiction en ligne de commande
├── track.py                   # Tracking et versioning MLflow
├── analyze_logs.py             # Analyse des logs d'utilisation de l'API
├── monitor_drift.py             # Détection de dérive des données (Evidently)
├── test_api.py                 # Tests automatisés de l'API
├── Dockerfile
├── requirements.txt
└── .github/workflows/ci.yml    # Pipeline CI/CD


Note sur predictor.py : ce fichier contient l'implémentation du modèle de prévision et n'est pas inclus dans ce dépôt public. L'interface qu'il expose (forecast(reference_time, history, horizon_slots) -> dict) est en revanche documentée ci-dessous, pour permettre de comprendre et d'utiliser le reste du pipeline.



Architecture du modèle (aperçu)

Le modèle combine, pour chaque créneau à prévoir :


un classifieur estimant la probabilité qu'un événement ait lieu (arrivée, départ) ;
un régresseur estimant la magnitude de cet événement s'il a lieu.


La prévision finale résulte de la combinaison des deux étages. La prévision sur un horizon de plusieurs créneaux est calculée de façon autorégressive : chaque créneau prédit est réinjecté comme donnée pour prédire le créneau suivant.

Installation

bashgit clone https://github.com/kenza125/mlops-dcforecaster.git
cd mlops-dcforecaster
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt

Utilisation

Lancer l'API

bashuvicorn api:app --reload

Puis envoyer une requête :

bashcurl -X POST "http://127.0.0.1:8000/predict?horizon_slots=4" \
  -F "file=@history.csv"

Versionner un modèle avec MLflow

bashpython track.py --csv history.csv --horizon 4
mlflow ui

Lancer les tests

bashpytest test_api.py

Analyser l'usage de l'API

bashpython analyze_logs.py

Détecter une dérive des données

bashpython monitor_drift.py --csv history.csv --window 96

Stack technique

FastAPI · Uvicorn · XGBoost · MLflow · Evidently AI · pandas · pytest · Docker · GitHub Actions
