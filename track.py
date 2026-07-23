"""
Tracking MLflow pour DCForecaster.

Version avec métriques d'évaluation par étage (classifieur + régresseur)
et métriques sur le résultat final combiné.

Principe :
    - Stage 1 (classifieurs arrival / departure) : PR-AUC + Brier score,
      évalués sur tous les slots de la fenêtre de test.
    - Stage 2 (régresseurs) : NRMSE (normalisé par std), calculé
      uniquement sur les slots actifs (valeur réelle > 0), sur la
      sortie BRUTE du régresseur (avant multiplication par la proba).
      Signaux couverts : arrival_count, departure_count, avg_energy_kWh,
      avg_duration_mins (classifieur associé = p_arrival, comme dans
      predictor.py::_predict_slot).
    - Résultat final (proba x régresseur, clip pour la durée) : NRMSE
      sur tous les slots de la fenêtre de test, normalisé par std.
    - Enregistrement du pipeline complet comme MLflow Model.
    - Chaque nouveau modèle enregistré reçoit automatiquement
      l'alias MLflow @production.

Usage:
    python track.py --csv history.csv --horizon 4
"""

import argparse
import os
import warnings

import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from predictor import DCForecaster, MAX_DURATION_MINS


MODEL_DIR = os.path.join("models", "dc_v2")
REGISTERED_MODEL_NAME = "DCForecaster_v2"

# Colonnes utilisées comme historique par le forecaster (feed-back autorégressif)
SIGS = [
    "arrival_count",
    "departure_count",
    "avg_energy_kWh"
]

# Signal supplémentaire évalué (non ré-injecté dans le buffer autorégressif,
# car predictor.py ne le boucle pas non plus)
DURATION_COL = "avg_duration_mins"


# ---------------------------------------------------------------------------
# Wrapper MLflow PyFunc
# ---------------------------------------------------------------------------
class DCForecasterModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        import predictor as predictor_module
        from pathlib import Path

        predictor_module.MODELS_DIR = Path(
            context.artifacts["models_dir"]
        )

        self.forecaster = predictor_module.DCForecaster()

    def predict(self, context, model_input: pd.DataFrame, params=None):

        params = params or {}

        horizon_slots = int(
            params.get("horizon_slots", 4)
        )

        reference_time = pd.Timestamp(
            params.get(
                "reference_time",
                model_input.index.max()
            )
        )

        return self.forecaster.forecast(
            reference_time=reference_time,
            history=model_input,
            horizon_slots=horizon_slots,
        )


# ---------------------------------------------------------------------------
# Fonctions de métriques
# ---------------------------------------------------------------------------
def _classifier_metrics(y_true_active, y_prob):
    """
    PR-AUC et Brier score pour un classifieur de stage 1.

    y_true_active : labels binaires (1 si le slot est actif, réel > 0)
    y_prob        : probabilités prédites par le classifieur
    """

    y_true_active = np.asarray(y_true_active, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    brier = float(
        brier_score_loss(y_true_active, y_prob)
    )

    if len(np.unique(y_true_active)) < 2:
        warnings.warn(
            "Une seule classe présente dans la fenêtre de test : "
            "PR-AUC non calculable (NaN)."
        )
        pr_auc = float("nan")

    else:
        pr_auc = float(
            average_precision_score(y_true_active, y_prob)
        )

    return pr_auc, brier


def _nrmse(y_true, y_pred):
    """NRMSE = RMSE / std(y_true). Retourne NaN si std == 0 ou trop peu de points."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) < 2:
        return float("nan")

    rmse = np.sqrt(
        np.mean((y_true - y_pred) ** 2)
    )

    std = y_true.std()

    if std == 0:
        warnings.warn(
            "std(y_true) == 0 sur cette fenêtre : NRMSE non calculable (NaN)."
        )
        return float("nan")

    return float(rmse / std)


def _regressor_nrmse_active(actual, raw_pred):
    """NRMSE du régresseur seul (sortie brute), restreint aux slots actifs (actual > 0)."""

    actual = np.asarray(actual, dtype=float)
    raw_pred = np.asarray(raw_pred, dtype=float)

    mask = actual > 0

    if mask.sum() < 2:
        warnings.warn(
            "Moins de 2 slots actifs dans la fenêtre de test : "
            "NRMSE régresseur non calculable (NaN)."
        )
        return float("nan")

    return _nrmse(actual[mask], raw_pred[mask])


# ---------------------------------------------------------------------------
# Évaluation du modèle sur la fenêtre de test
# ---------------------------------------------------------------------------
def evaluate_model(
        forecaster: DCForecaster,
        history: pd.DataFrame,
        horizon_slots: int
):
    """
    Reproduit manuellement le rollout autorégressif de forecaster.forecast()
    afin de récupérer, pour chaque slot de la fenêtre de test :
      - les probabilités des 2 classifieurs (p_arrival, p_departure)
      - les sorties BRUTES des régresseurs (avant multiplication par la proba)
      - la prédiction finale combinée (proba x régresseur)
    Le buffer autorégressif ne contient que les 3 signaux SIGS, exactement
    comme dans predictor.py::forecast (avg_duration_mins n'est jamais
    ré-injecté en lag, il n'est prédit qu'à titre indicatif).
    """

    resolution = pd.Timedelta(minutes=15)

    if len(history) <= horizon_slots:
        raise ValueError(
            f"Pas assez de données : {len(history)} lignes disponibles."
        )

    ref_time = (
        history.index.max().ceil("15min")
        -
        horizon_slots * resolution
    )

    buffer = history[SIGS][history.index < ref_time].copy()

    if buffer.empty:
        raise ValueError(
            "Aucune donnée avant la fenêtre de test."
        )

    has_duration = DURATION_COL in history.columns

    records = []

    for step in range(horizon_slots):

        slot_start = ref_time + step * resolution

        row = forecaster._build_row(slot_start, buffer)

        p_arr = float(
            forecaster._clf_arrival.predict_proba(row)[:, 1][0]
        )

        p_dep = float(
            forecaster._clf_departure.predict_proba(row)[:, 1][0]
        )

        raw_arrival = max(
            float(forecaster._regs["arrival_count"].predict(row)[0]), 0.0
        )

        raw_departure = max(
            float(forecaster._regs["departure_count"].predict(row)[0]), 0.0
        )

        raw_energy = max(
            float(forecaster._regs["avg_energy_kWh"].predict(row)[0]), 0.0
        )

        raw_duration = max(
            float(forecaster._regs["avg_duration_mins"].predict(row)[0]), 0.0
        )

        final_arrival = p_arr * raw_arrival
        final_departure = p_dep * raw_departure
        final_energy = p_arr * raw_energy
        final_duration = min(p_arr * raw_duration, MAX_DURATION_MINS)

        # feed-back autorégressif, identique à forecaster.forecast()
        # (avg_duration_mins n'est PAS dans le buffer, comme dans predictor.py)
        buffer.loc[slot_start] = [
            final_arrival,
            final_departure,
            final_energy,
        ]

        if slot_start not in history.index:
            continue

        actual = history.loc[slot_start]

        record = {
            "timestamp":        slot_start,
            "p_arrival":        p_arr,
            "p_departure":      p_dep,
            "raw_arrival":      raw_arrival,
            "raw_departure":    raw_departure,
            "raw_energy":       raw_energy,
            "raw_duration":     raw_duration,
            "final_arrival":    final_arrival,
            "final_departure":  final_departure,
            "final_energy":     final_energy,
            "final_duration":   final_duration,
            "actual_arrival":   float(actual["arrival_count"]),
            "actual_departure": float(actual["departure_count"]),
            "actual_energy":    float(actual["avg_energy_kWh"]),
        }

        record["actual_duration"] = (
            float(actual[DURATION_COL]) if has_duration else float("nan")
        )

        records.append(record)

    if not records:
        raise ValueError(
            "Aucun timestamp correspondant trouvé."
        )

    df = pd.DataFrame(records).set_index("timestamp")

    # ---------------- Stage 1 : classifieurs ----------------
    pr_auc_arrival, brier_arrival = _classifier_metrics(
        df["actual_arrival"] > 0,
        df["p_arrival"]
    )

    pr_auc_departure, brier_departure = _classifier_metrics(
        df["actual_departure"] > 0,
        df["p_departure"]
    )

    # ---------------- Stage 2 : régresseurs (slots actifs) ----------------
    nrmse_reg_arrival = _regressor_nrmse_active(
        df["actual_arrival"], df["raw_arrival"]
    )

    nrmse_reg_departure = _regressor_nrmse_active(
        df["actual_departure"], df["raw_departure"]
    )

    nrmse_reg_energy = _regressor_nrmse_active(
        df["actual_energy"], df["raw_energy"]
    )

    # ---------------- Résultat final combiné (tous les slots) ----------------
    nrmse_final_arrival = _nrmse(
        df["actual_arrival"], df["final_arrival"]
    )

    nrmse_final_departure = _nrmse(
        df["actual_departure"], df["final_departure"]
    )

    nrmse_final_energy = _nrmse(
        df["actual_energy"], df["final_energy"]
    )

    metrics = {
        "pr_auc_arrival":        pr_auc_arrival,
        "brier_arrival":         brier_arrival,
        "pr_auc_departure":      pr_auc_departure,
        "brier_departure":       brier_departure,
        "nrmse_reg_arrival":     nrmse_reg_arrival,
        "nrmse_reg_departure":   nrmse_reg_departure,
        "nrmse_reg_energy":      nrmse_reg_energy,
        "nrmse_final_arrival":   nrmse_final_arrival,
        "nrmse_final_departure": nrmse_final_departure,
        "nrmse_final_energy":    nrmse_final_energy,
    }

    n_active = {
        "n_active_arrival":   int((df["actual_arrival"] > 0).sum()),
        "n_active_departure": int((df["actual_departure"] > 0).sum()),
        "n_active_energy":    int((df["actual_energy"] > 0).sum()),
    }

    if has_duration:
        nrmse_reg_duration = _regressor_nrmse_active(
            df["actual_duration"], df["raw_duration"]
        )

        nrmse_final_duration = _nrmse(
            df["actual_duration"], df["final_duration"]
        )

        metrics["nrmse_reg_duration"] = nrmse_reg_duration
        metrics["nrmse_final_duration"] = nrmse_final_duration

        n_active["n_active_duration"] = int(
            (df["actual_duration"] > 0).sum()
        )

    else:
        warnings.warn(
            f"Colonne '{DURATION_COL}' absente du CSV : "
            "métriques de durée non calculées."
        )

    return metrics, n_active, len(df), ref_time


def _log_metric_safe(name, value):
    """Évite de logger des NaN (MLflow les accepte mais ça pollue les dashboards)."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        print(f"[WARNING] Métrique '{name}' = NaN, non loggée.")
        return

    mlflow.log_metric(name, value)


def run_tracking(
        csv_path: str,
        horizon_slots: int
):

    history = pd.read_csv(
        csv_path,
        parse_dates=[0],
        index_col=0
    )

    if history.index.duplicated().any():
        raise ValueError(
            "Timestamps dupliqués détectés dans le CSV. "
            "Vérifie qu'il ne mélange pas plusieurs zones/chargerTypes."
        )

    forecaster = DCForecaster()

    metrics, n_active, n_slots, ref_time = evaluate_model(
        forecaster,
        history,
        horizon_slots
    )

    # échantillon pour la signature MLflow (rollout standard, non instrumenté)
    sample_input = history[history.index < ref_time].tail(200)

    sample_output = forecaster.forecast(
        reference_time=ref_time,
        history=history[history.index < ref_time],
        horizon_slots=horizon_slots,
    )

    with mlflow.start_run() as run:

        mlflow.log_param("model_name", REGISTERED_MODEL_NAME)
        mlflow.log_param("csv_input", os.path.basename(csv_path))
        mlflow.log_param("horizon_slots", horizon_slots)
        mlflow.log_param("n_rows_history", len(history))
        mlflow.log_param("n_test_slots", n_slots)

        for name, value in n_active.items():
            mlflow.log_param(name, value)

        for name, value in metrics.items():
            _log_metric_safe(name, value)

        try:
            model_signature = infer_signature(
                sample_input,
                sample_output
            )

        except Exception as exc:

            print(
                "[WARNING] Signature sortie impossible :",
                exc
            )

            model_signature = infer_signature(sample_input)

        # ---------------------------------------------------------------
        # Enregistrement MLflow Model Registry
        # ---------------------------------------------------------------
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=DCForecasterModel(),
            artifacts={
                "models_dir": MODEL_DIR
            },
            code_paths=[
                "predictor.py"
            ],
            signature=model_signature,
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        # ---------------------------------------------------------------
        # Attribution automatique de l'alias @production
        # ---------------------------------------------------------------
        client = MlflowClient()

        latest_version = model_info.registered_model_version

        try:
            client.delete_registered_model_alias(
                REGISTERED_MODEL_NAME,
                "production"
            )

        except Exception:
            pass

        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME,
            alias="production",
            version=latest_version
        )

        print("\n==============================")
        print("MLflow Tracking terminé")
        print("==============================")
        print(f"Run ID : {run.info.run_id}")
        print(f"Version MLflow : {latest_version}")
        print("Alias actif : @production")
        print(f"Slots évalués : {n_slots}")
        print("--- Stage 1 (classifieurs) ---")
        print(f"PR-AUC arrival   : {metrics['pr_auc_arrival']:.4f}   Brier arrival   : {metrics['brier_arrival']:.4f}")
        print(f"PR-AUC departure : {metrics['pr_auc_departure']:.4f}   Brier departure : {metrics['brier_departure']:.4f}")
        print("--- Stage 2 (régresseurs, slots actifs) ---")
        print(f"NRMSE reg arrival   : {metrics['nrmse_reg_arrival']:.4f}")
        print(f"NRMSE reg departure : {metrics['nrmse_reg_departure']:.4f}")
        print(f"NRMSE reg energy    : {metrics['nrmse_reg_energy']:.4f}")
        if "nrmse_reg_duration" in metrics:
            print(f"NRMSE reg duration  : {metrics['nrmse_reg_duration']:.4f}")
        print("--- Résultat final (tous les slots) ---")
        print(f"NRMSE final arrival   : {metrics['nrmse_final_arrival']:.4f}")
        print(f"NRMSE final departure : {metrics['nrmse_final_departure']:.4f}")
        print(f"NRMSE final energy    : {metrics['nrmse_final_energy']:.4f}")
        if "nrmse_final_duration" in metrics:
            print(f"NRMSE final duration  : {metrics['nrmse_final_duration']:.4f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Tracking MLflow DCForecaster"
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="Chemin CSV historique"
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=4
    )

    args = parser.parse_args()

    run_tracking(
        args.csv,
        args.horizon
    )