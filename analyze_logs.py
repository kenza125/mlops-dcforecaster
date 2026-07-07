"""
Analyse le fichier predictions_log.csv généré par l'API (monitoring simple).

Usage:
    python analyze_logs.py
"""

import os

import pandas as pd

LOG_FILE = "predictions_log.csv"


def main():
    if not os.path.exists(LOG_FILE):
        print(f"Aucun fichier de log trouvé ({LOG_FILE}). Fais d'abord quelques appels à /predict.")
        return

    df = pd.read_csv(LOG_FILE)

    print(f"Nombre total de prédictions : {len(df)}")
    print(f"Horizon moyen demandé      : {df['horizon_slots'].mean():.1f} créneaux")
    print(f"Taille moyenne historique  : {df['n_rows_history'].mean():.1f} lignes")
    print(f"Arrivées moyennes prédites : {df['mean_expected_arrival_count'].mean():.3f}")
    print(f"Énergie moyenne prédite    : {df['mean_expected_energy_kwh'].mean():.2f} kWh")
    print()
    print("Dernières prédictions :")
    print(df.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
