import argparse
import pandas as pd
import numpy as np

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


NUMERIC_COLUMNS = [
    "arrival_count",
    "departure_count",
    "avg_energy_kWh",
    "avg_duration_mins",
]

OUTPUT_HTML = "drift_report.html"



def load_history(csv_path: str) -> pd.DataFrame:

    df = pd.read_csv(
        csv_path,
        parse_dates=["timestamp"]
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    return df



def split_reference_current(df: pd.DataFrame, window: int):

    if len(df) <= window:
        raise ValueError(
            f"Pas assez de données : {len(df)} lignes disponibles"
        )

    reference = df.iloc[:-window]
    current = df.iloc[-window:]

    return reference, current



def calculate_rmse(reference, current, column):

    # prendre une fenêtre de même taille
    reference = reference.tail(len(current))


    reference_values = pd.to_numeric(
        reference[column],
        errors="coerce"
    )

    current_values = pd.to_numeric(
        current[column],
        errors="coerce"
    )


    comparison = pd.DataFrame(
        {
            "reference": reference_values.values,
            "current": current_values.values
        }
    )


    comparison = comparison.dropna()


    if comparison.empty:
        return 0.0


    rmse = np.sqrt(
        np.mean(
            (
                comparison["current"]
                -
                comparison["reference"]
            ) ** 2
        )
    )


    return float(rmse)



def compute_rmse_metrics(reference, current):

    return {

        "rmse_arrival_count":
            calculate_rmse(
                reference,
                current,
                "arrival_count"
            ),


        "rmse_departure_count":
            calculate_rmse(
                reference,
                current,
                "departure_count"
            ),


        "rmse_energy_kwh":
            calculate_rmse(
                reference,
                current,
                "avg_energy_kWh"
            ),


        "rmse_duration_mins":
            calculate_rmse(
                reference,
                current,
                "avg_duration_mins"
            )
    }



def append_rmse_to_html(html_file, rmse_values):

    with open(
        html_file,
        "r",
        encoding="utf-8"
    ) as file:

        html = file.read()



    rmse_html = """

    <div style="
        margin:40px;
        padding:20px;
        font-family:Arial;
        border-top:2px solid #000;
    ">

    <h2>RMSE Metrics</h2>

    <table border="1"
           cellpadding="10"
           cellspacing="0">

    <tr>
        <th>Metric</th>
        <th>Value</th>
    </tr>

    """



    for metric, value in rmse_values.items():

        rmse_html += f"""

        <tr>
            <td>{metric}</td>
            <td>{value:.6f}</td>
        </tr>

        """



    rmse_html += """

    </table>

    </div>

    """



    # Ajout directement à la fin du HTML Evidently
    html += rmse_html



    with open(
        html_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(html)



def run_drift_check(csv_path: str, window: int):


    df = load_history(csv_path)



    reference_data, current_data = split_reference_current(
        df,
        window
    )


    print(
        f"Reference : {len(reference_data)} lignes"
    )

    print(
        f"Current : {len(current_data)} lignes"
    )



    # ==========================
    # RMSE
    # ==========================

    rmse_metrics = compute_rmse_metrics(
        reference_data,
        current_data
    )


    print("\nRMSE Metrics:")

    for name, value in rmse_metrics.items():

        print(
            f"{name} = {value}"
        )



    # ==========================
    # Evidently Data Drift
    # ==========================

    report = Report(
        metrics=[
            DataDriftPreset(
                columns=NUMERIC_COLUMNS
            )
        ]
    )



    report.run(

        reference_data=
            reference_data[NUMERIC_COLUMNS],


        current_data=
            current_data[NUMERIC_COLUMNS]

    )



    report.save_html(
        OUTPUT_HTML
    )



    # Ajouter RMSE dans le même HTML
    append_rmse_to_html(
        OUTPUT_HTML,
        rmse_metrics
    )


    print(
        f"\nRapport complet généré : {OUTPUT_HTML}"
    )



if __name__ == "__main__":


    parser = argparse.ArgumentParser(
        description=
        "Monitoring drift + RMSE"
    )


    parser.add_argument(
        "--csv",
        default="history.csv",
        help="Chemin du fichier CSV"
    )


    parser.add_argument(
        "--window",
        type=int,
        default=96,
        help="Nombre de créneaux récents"
    )


    args = parser.parse_args()


    run_drift_check(
        args.csv,
        args.window
    )