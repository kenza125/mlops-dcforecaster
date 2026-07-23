"""
Drift_report.py — Détection et analyse de data drift pour DCForecaster avec Evidently.

Compatible evidently==0.4.17 (ancienne API : evidently.report.Report,
evidently.metric_preset.DataDriftPreset, evidently.pipeline.column_mapping.ColumnMapping).

Compare une période de référence (comportement connu/validé) à une période
courante (récente) sur les colonnes d'entrée du modèle, afin de détecter
un changement de distribution qui pourrait dégrader les prédictions.

Structure CSV attendue :
    timestamp, zone, chargerType, arrival_count, avg_energy_kWh,
    avg_duration_mins, departure_count

Usage:
    python Drift_report.py --csv history.csv --current-days 7
    python Drift_report.py --csv history.csv --current-days 7 --by-segment
    python Drift_report.py --csv history.csv --current-days 7 --log-mlflow
"""

import argparse
import os
from datetime import timedelta

import pandas as pd
import mlflow

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping


NUMERICAL_COLS = [
    "arrival_count",
    "departure_count",
    "avg_energy_kWh",
    "avg_duration_mins",
]
CATEGORICAL_COLS = ["zone", "chargerType"]

OUTPUT_DIR = "Drift_reports"
DRIFT_SHARE_THRESHOLD = 0.5  # alerte si >= 50% des colonnes ont drifté


# ---------------------------------------------------------------------------
# Chargement et découpage temporel
# ---------------------------------------------------------------------------
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    missing = set(NUMERICAL_COLS + CATEGORICAL_COLS + ["timestamp"]) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV : {missing}")

    return df


def split_reference_current(df: pd.DataFrame, current_days: int):
    """
    current   = les `current_days` derniers jours
    reference = tout ce qui précède
    """
    cutoff = df["timestamp"].max() - timedelta(days=current_days)

    reference = df[df["timestamp"] < cutoff].copy()
    current = df[df["timestamp"] >= cutoff].copy()

    if reference.empty:
        raise ValueError(
            "Aucune donnée de référence disponible avant la période courante. "
            "Vérifie --current-days ou la profondeur de ton historique."
        )
    if current.empty:
        raise ValueError("Aucune donnée dans la période courante.")

    return reference, current


# ---------------------------------------------------------------------------
# Construction du rapport Evidently (API 0.4.17)
# ---------------------------------------------------------------------------
def build_column_mapping() -> ColumnMapping:
    column_mapping = ColumnMapping()
    column_mapping.numerical_features = NUMERICAL_COLS
    column_mapping.categorical_features = CATEGORICAL_COLS
    return column_mapping


def run_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> Report:
    column_mapping = build_column_mapping()

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_df[NUMERICAL_COLS + CATEGORICAL_COLS],
        current_data=current_df[NUMERICAL_COLS + CATEGORICAL_COLS],
        column_mapping=column_mapping,
    )
    return report


# ---------------------------------------------------------------------------
# Extraction et interprétation des résultats (structure as_dict() de 0.4.17)
# ---------------------------------------------------------------------------
def extract_drift_summary(report: Report) -> dict:
    result = report.as_dict()

    # metrics[0] = DatasetDriftMetric (résumé global)
    # metrics[1] = DataDriftTable (détail par colonne)
    dataset_drift_result = result["metrics"][0]["result"]

    summary = {
        "dataset_drift": dataset_drift_result.get("dataset_drift"),
        "n_drifted_columns": dataset_drift_result.get("number_of_drifted_columns"),
        "n_total_columns": dataset_drift_result.get("number_of_columns"),
        "per_column": {},
    }

    drift_table_result = result["metrics"][1]["result"]
    drift_by_columns = drift_table_result.get("drift_by_columns", {})

    for col, col_result in drift_by_columns.items():
        summary["per_column"][col] = {
            "drift_score": col_result.get("drift_score"),
            "drift_detected": col_result.get("drift_detected"),
            "stattest": col_result.get("stattest_name"),
        }

    if summary["n_total_columns"]:
        summary["drift_share"] = (
            summary["n_drifted_columns"] / summary["n_total_columns"]
        )
    else:
        summary["drift_share"] = 0.0

    return summary


def print_summary(label: str, summary: dict):
    print(f"\n=== Drift Report : {label} ===")
    print(f"Colonnes driftées : {summary['n_drifted_columns']} / {summary['n_total_columns']}")
    print(f"Part de colonnes driftées : {summary['drift_share']:.1%}")
    print(f"Dataset drift détecté : {summary['dataset_drift']}")
    print("--- Détail par colonne ---")
    for col, info in summary["per_column"].items():
        flag = "DRIFT" if info["drift_detected"] else "OK"
        score = info["drift_score"]
        score_str = f"{score:.4f}" if score is not None else "N/A"
        stattest = info["stattest"] or "N/A"
        print(f"  {col:25s} score={score_str:>8s}  test={stattest:10s}  {flag}")

    if summary["drift_share"] >= DRIFT_SHARE_THRESHOLD:
        print(
            f"\n[ALERTE] {summary['drift_share']:.1%} des colonnes ont drifté "
            f"(seuil : {DRIFT_SHARE_THRESHOLD:.0%}). Le modèle opère peut-être "
            "hors de son environnement d'entraînement."
        )


# ---------------------------------------------------------------------------
# Logging MLflow (optionnel, cohérent avec track.py)
# ---------------------------------------------------------------------------
def log_summary_to_mlflow(label: str, summary: dict, ref_len: int, cur_len: int):
    with mlflow.start_run(run_name=f"drift_monitor_{label}"):
        mlflow.log_param("segment", label)
        mlflow.log_param("n_reference_rows", ref_len)
        mlflow.log_param("n_current_rows", cur_len)

        mlflow.log_metric("n_drifted_columns", summary["n_drifted_columns"])
        mlflow.log_metric("drift_share", summary["drift_share"])
        mlflow.log_metric("dataset_drift", int(bool(summary["dataset_drift"])))

        for col, info in summary["per_column"].items():
            if info["drift_score"] is not None:
                mlflow.log_metric(f"drift_score_{col}", info["drift_score"])
            mlflow.log_metric(f"drift_detected_{col}", int(bool(info["drift_detected"])))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def analyze_segment(label: str, reference_df: pd.DataFrame, current_df: pd.DataFrame,
                     output_dir: str, log_mlflow: bool):
    report = run_drift_report(reference_df, current_df)
    summary = extract_drift_summary(report)

    print_summary(label, summary)

    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, f"drift_{label}.html")
    report.save_html(html_path)
    print(f"Rapport HTML sauvegardé : {html_path}")

    if log_mlflow:
        log_summary_to_mlflow(label, summary, len(reference_df), len(current_df))

    return summary


def run_monitoring(csv_path: str, current_days: int, by_segment: bool,
                    output_dir: str, log_mlflow: bool):
    df = load_data(csv_path)

    if not by_segment:
        reference, current = split_reference_current(df, current_days)
        analyze_segment("global", reference, current, output_dir, log_mlflow)
        return

    segments = df.groupby(["zone", "chargerType"])

    for (zone, charger_type), segment_df in segments:
        label = f"{zone}_{charger_type}".replace(" ", "_")

        try:
            reference, current = split_reference_current(segment_df, current_days)
        except ValueError as exc:
            print(f"[SKIP] Segment {label} : {exc}")
            continue

        analyze_segment(label, reference, current, output_dir, log_mlflow)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitoring de drift DCForecaster (Evidently)")
    parser.add_argument("--csv", required=True, help="Chemin vers le CSV d'historique")
    parser.add_argument("--current-days", type=int, default=7,
                         help="Nombre de jours considérés comme 'période courante'")
    parser.add_argument("--by-segment", action="store_true",
                         help="Analyser le drift séparément par zone x chargerType")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                         help="Dossier de sortie pour les rapports HTML")
    parser.add_argument("--log-mlflow", action="store_true",
                         help="Logger les résultats dans MLflow")

    args = parser.parse_args()

    run_monitoring(
        csv_path=args.csv,
        current_days=args.current_days,
        by_segment=args.by_segment,
        output_dir=args.output_dir,
        log_mlflow=args.log_mlflow,
    )