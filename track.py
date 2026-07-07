"""
Tracking MLflow pour DCForecaster.

Version simplifiée : un seul RMSE, calculé sur les derniers `horizon_slots`
créneaux de l'historique.

Principe :
    - On garde les derniers `horizon_slots` créneaux comme "vérité terrain"
      (valeurs réellement observées, déjà dans le CSV).
    - On entraîne/prédit avec tout le reste de l'historique (tout sauf
      ces derniers créneaux) -> pas de fuite de données.
    - On compare prédiction vs réel avec sqrt(mean((y_pred-y_true)^2)).
    - On enregistre le modèle comme un vrai "MLflow Model" (MLmodel +
      signature + flavor pyfunc) dans le Model Registry sous
      REGISTERED_MODEL_NAME.

Usage:
    python track.py --csv history.csv --horizon 4
"""

import argparse
import os

import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature
import numpy as np
import pandas as pd

from predictor import DCForecaster

MODEL_DIR = os.path.join("models", "dc_v2")
REGISTERED_MODEL_NAME = "DCForecaster_v2"
SIGS = ["arrival_count", "departure_count", "avg_energy_kWh"]


# ---------------------------------------------------------------------------
# Wrapper pyfunc : permet d'enregistrer TOUT le pipeline (2 clf + 4 reg +
# feature_cols.json) comme un seul modèle MLflow déployable, plutôt que 6
# modèles séparés sans lien entre eux dans le registry.
# ---------------------------------------------------------------------------
class DCForecasterModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import predictor as predictor_module
        from pathlib import Path

        # Redirige le module vers les .json embarqués dans l'artefact MLflow
        # (au lieu du chemin relatif codé en dur dans predictor.py)
        predictor_module.MODELS_DIR = Path(context.artifacts["models_dir"])
        self.forecaster = predictor_module.DCForecaster()

    def predict(self, context, model_input: pd.DataFrame, params=None):
        params = params or {}
        horizon_slots = int(params.get("horizon_slots", 4))
        reference_time = pd.Timestamp(
            params.get("reference_time", model_input.index.max())
        )
        return self.forecaster.forecast(
            reference_time=reference_time,
            history=model_input,
            horizon_slots=horizon_slots,
        )


def _rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_rmse(forecaster: DCForecaster, history: pd.DataFrame, horizon_slots: int):
    """
    Calcule le RMSE sur une seule fenêtre : les derniers `horizon_slots`
    créneaux de `history` servent de vérité terrain, le modèle prédit
    cette fenêtre en ne voyant que les données antérieures.
    """
    resolution = pd.Timedelta(minutes=15)

    if len(history) <= horizon_slots:
        raise ValueError(
            f"Pas assez de données : il faut plus de {horizon_slots} créneaux "
            f"dans l'historique (trouvé : {len(history)})."
        )

    # Le début de la fenêtre de test = horizon_slots créneaux avant la fin
    ref_time = history.index.max().ceil("15min") - horizon_slots * resolution

    # Tout ce qui précède ref_time sert à entraîner/prédire
    train_buffer = history[history.index < ref_time]

    if train_buffer.empty:
        raise ValueError(
            "Aucune donnée disponible avant la fenêtre de test. "
            "Fournis un historique plus long ou réduis --horizon."
        )

    result = forecaster.forecast(
        reference_time=ref_time, history=train_buffer, horizon_slots=horizon_slots
    )
    sample_output = result  # servira d'exemple de sortie pour la signature MLflow

    preds = {c: [] for c in SIGS}
    trues = {c: [] for c in SIGS}

    for slot in result["forecastData"]:
        slot_ts = pd.Timestamp(slot["timeSlotStart"])
        if slot_ts not in history.index:
            continue
        actual = history.loc[slot_ts]
        preds["arrival_count"].append(slot["expectedArrivalCount"])
        trues["arrival_count"].append(actual["arrival_count"])
        preds["departure_count"].append(slot["expectedDepartureCount"])
        trues["departure_count"].append(actual["departure_count"])
        preds["avg_energy_kWh"].append(slot["expectedEnergyKwh"])
        trues["avg_energy_kWh"].append(actual["avg_energy_kWh"])

    if not preds["arrival_count"]:
        raise ValueError(
            "Aucun créneau prédit ne correspond à l'historique réel. "
            "Vérifie l'alignement des timestamps."
        )

    rmse = {c: _rmse(trues[c], preds[c]) for c in SIGS}
    return rmse, len(preds["arrival_count"]), sample_output


def run_tracking(csv_path: str, horizon_slots: int):
    history = pd.read_csv(csv_path, parse_dates=[0], index_col=0)
    forecaster = DCForecaster()

    rmse, n_slots, sample_output = compute_rmse(forecaster, history, horizon_slots)

    with mlflow.start_run() as run:
        mlflow.log_param("model_name", REGISTERED_MODEL_NAME)
        mlflow.log_param("csv_input", os.path.basename(csv_path))
        mlflow.log_param("horizon_slots", horizon_slots)
        mlflow.log_param("n_rows_history", len(history))
        mlflow.log_param("n_test_slots", n_slots)

        mlflow.log_metric("rmse_arrival_count", rmse["arrival_count"])
        mlflow.log_metric("rmse_departure_count", rmse["departure_count"])
        mlflow.log_metric("rmse_energy_kwh", rmse["avg_energy_kWh"])

        # On calcule la signature nous-mêmes (entrée + sortie), au lieu de
        # laisser MLflow la deviner via les type hints de `predict()`
        # -> évite l'erreur "Expected examples to be list, got DataFrame"
        # des versions récentes de MLflow.
        #
        # sample_output vient d'une vraie prédiction déjà faite pendant le
        # calcul du RMSE (compute_rmse) -> pas besoin d'appeler le modèle
        # une deuxième fois.
        input_example = history.tail(200)
        try:
            model_signature = infer_signature(input_example, sample_output)
        except Exception as exc:
            # Le format de sortie (dict imbriqué "forecastData": [...]) peut
            # ne pas être inférable tel quel par MLflow selon la version.
            # Dans ce cas on garde au moins la signature d'entrée, plutôt
            # que de faire planter tout le run.
            print(
                f"[avertissement] Impossible d'inférer le schéma de sortie "
                f"({exc}). Signature limitée à l'entrée uniquement."
            )
            model_signature = infer_signature(input_example)

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=DCForecasterModel(),
            artifacts={"models_dir": MODEL_DIR},
            code_paths=["predictor.py"],
            signature=model_signature,
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        print(
            f"Run {run.info.run_id} terminée sur {n_slots} créneau(x) de test "
            f"(horizon={horizon_slots} créneaux).\n"
            f"RMSE arrival_count   : {rmse['arrival_count']:.4f}\n"
            f"RMSE departure_count : {rmse['departure_count']:.4f}\n"
            f"RMSE energy_kWh      : {rmse['avg_energy_kWh']:.4f}\n"
            f"Modèle enregistré sous '{REGISTERED_MODEL_NAME}' dans le Model Registry."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tracking MLflow pour DCForecaster")
    parser.add_argument("--csv", required=True, help="Chemin vers le CSV d'historique")
    parser.add_argument("--horizon", type=int, default=4)
    args = parser.parse_args()

    run_tracking(args.csv, args.horizon)