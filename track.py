"""
Tracking MLflow pour DCForecaster.

Version avec gestion des aliases MLflow.

Principe :
    - Calcul du RMSE sur les derniers `horizon_slots` créneaux.
    - Enregistrement du pipeline complet comme MLflow Model.
    - Chaque nouveau modèle enregistré reçoit automatiquement
      l'alias MLflow @production.
    - Les anciennes versions restent disponibles dans le Registry.

Usage:
    python track.py --csv history.csv --horizon 4
"""

import argparse
import os

import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

import numpy as np
import pandas as pd

from predictor import DCForecaster


MODEL_DIR = os.path.join("models", "dc_v2")
REGISTERED_MODEL_NAME = "DCForecaster_v2"
SIGS = [
    "arrival_count",
    "departure_count",
    "avg_energy_kWh"
]


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



def _rmse(y_true, y_pred):

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return float(
        np.sqrt(
            np.mean(
                (y_true - y_pred) ** 2
            )
        )
    )



def compute_rmse(
        forecaster: DCForecaster,
        history: pd.DataFrame,
        horizon_slots: int
):

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


    train_buffer = history[
        history.index < ref_time
    ]


    if train_buffer.empty:
        raise ValueError(
            "Aucune donnée avant la fenêtre de test."
        )


    result = forecaster.forecast(
        reference_time=ref_time,
        history=train_buffer,
        horizon_slots=horizon_slots
    )


    sample_output = result


    preds = {
        c: []
        for c in SIGS
    }

    trues = {
        c: []
        for c in SIGS
    }



    for slot in result["forecastData"]:

        slot_ts = pd.Timestamp(
            slot["timeSlotStart"]
        )


        if slot_ts not in history.index:
            continue


        actual = history.loc[slot_ts]


        preds["arrival_count"].append(
            slot["expectedArrivalCount"]
        )

        trues["arrival_count"].append(
            actual["arrival_count"]
        )


        preds["departure_count"].append(
            slot["expectedDepartureCount"]
        )

        trues["departure_count"].append(
            actual["departure_count"]
        )


        preds["avg_energy_kWh"].append(
            slot["expectedEnergyKwh"]
        )

        trues["avg_energy_kWh"].append(
            actual["avg_energy_kWh"]
        )



    if not preds["arrival_count"]:
        raise ValueError(
            "Aucun timestamp correspondant trouvé."
        )


    rmse = {
        c: _rmse(
            trues[c],
            preds[c]
        )
        for c in SIGS
    }


    return (
        rmse,
        len(preds["arrival_count"]),
        sample_output
    )



def run_tracking(
        csv_path: str,
        horizon_slots: int
):


    history = pd.read_csv(
        csv_path,
        parse_dates=[0],
        index_col=0
    )


    forecaster = DCForecaster()


    rmse, n_slots, sample_output = compute_rmse(
        forecaster,
        history,
        horizon_slots
    )



    with mlflow.start_run() as run:


        mlflow.log_param(
            "model_name",
            REGISTERED_MODEL_NAME
        )


        mlflow.log_param(
            "csv_input",
            os.path.basename(csv_path)
        )


        mlflow.log_param(
            "horizon_slots",
            horizon_slots
        )


        mlflow.log_param(
            "n_rows_history",
            len(history)
        )


        mlflow.log_param(
            "n_test_slots",
            n_slots
        )



        mlflow.log_metric(
            "rmse_arrival_count",
            rmse["arrival_count"]
        )


        mlflow.log_metric(
            "rmse_departure_count",
            rmse["departure_count"]
        )


        mlflow.log_metric(
            "rmse_energy_kwh",
            rmse["avg_energy_kWh"]
        )



        input_example = history.tail(200)



        try:

            model_signature = infer_signature(
                input_example,
                sample_output
            )


        except Exception as exc:

            print(
                "[WARNING] Signature sortie impossible :",
                exc
            )

            model_signature = infer_signature(
                input_example
            )



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


        versions = client.search_model_versions(
            f"name='{REGISTERED_MODEL_NAME}'"
        )


        latest_version = max(
            int(v.version)
            for v in versions
        )


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

        print(
            f"Run ID : {run.info.run_id}"
        )

        print(
            f"Version MLflow : {latest_version}"
        )

        print(
            "Alias actif : @production"
        )

        print(
            f"RMSE arrival_count   : {rmse['arrival_count']:.4f}"
        )

        print(
            f"RMSE departure_count : {rmse['departure_count']:.4f}"
        )

        print(
            f"RMSE energy_kWh      : {rmse['avg_energy_kWh']:.4f}"
        )




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