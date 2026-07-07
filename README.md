# DCForecaster MLOps Prototype

Prototype MLOps pour déployer et tester le modèle **DCForecaster** (XGBoost 2 étages :
classification arrivée/départ + régression count/énergie/durée), sur le même schéma que
[mlops-prototype](https://github.com/kenza125/mlops-prototype), mais avec un modèle
propriétaire au lieu d'un modèle Hugging Face pré-entraîné.

## Structure

```
mlops-dcforecaster/
├── predictor.py            <- À copier depuis ton package modèle (non inclus ici)
├── models/dc_v2/            <- À copier: les 6 fichiers .json + feature_cols.json
├── api.py                  <- API FastAPI (POST /predict avec upload CSV)
├── predict.py               <- Prédiction en ligne de commande
├── track.py                 <- Tracking MLflow (params + métriques + artifacts modèle)
├── analyze_logs.py           <- Analyse du fichier predictions_log.csv
├── test_api.py               <- Tests (modèle mocké, pas besoin des vrais fichiers)
├── Dockerfile
├── requirements.txt
└── .github/workflows/ci.yml  <- Tests + build Docker automatique
```

## Mise en place

1. Copie `predictor.py` et `models/dc_v2/*.json` (y compris `feature_cols.json`) dans ce dossier.
2. Installe les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Lancer l'API

```bash
uvicorn api:app --reload
```

Puis teste avec un CSV d'historique (colonnes: timestamp en index, `arrival_count`,
`departure_count`, `avg_energy_kWh`) :

```bash
curl -X POST "http://localhost:8000/predict?horizon_slots=4" \
  -F "file=@history.csv"
```

Chaque appel est loggé dans `predictions_log.csv`.

## Prédiction en ligne de commande

```bash
python predict.py --csv history.csv --horizon 4
```

## Tracking MLflow

```bash
python track.py --csv history.csv --horizon 4
mlflow ui   # http://localhost:5000
```

## Analyse des logs

```bash
python analyze_logs.py
```

## Tests

```bash
pytest test_api.py -v
```

## Docker

```bash
docker build -t dcforecaster-api .
docker run -p 8000:8000 dcforecaster-api
```
