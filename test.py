import pandas as pd
from predictor import DCForecaster

# --- 1. Charger le CSV ---
df = pd.read_csv(r"C:\Users\ASUS\Downloads\deploy_minimal\deploy_minimal\inpuut.csv", parse_dates=["timestamp"])

# --- 2. Choisir TA zone ---
ZONE = "PublicParking"          # <-- change ici si tu veux l'autre zone
CHARGER_TYPE = "DC"

df_filtered = df[(df["zone"] == ZONE) & (df["chargerType"] == CHARGER_TYPE)].copy()

if df_filtered.empty:
    raise ValueError(f"Aucune ligne trouvée pour {ZONE} / {CHARGER_TYPE}")

# --- 3. Timestamp en index, trié ---
df_filtered = df_filtered.set_index("timestamp").sort_index()

# --- 4. Forcer fréquence 15 min, combler les trous avec 0.0 ---
history = df_filtered.asfreq("15min", fill_value=0.0)
history.to_csv("history.csv") 
# --- 5. Vérifier la durée disponible ---
nb_jours = (history.index[-1] - history.index[0]).days
print(f"Historique disponible : {nb_jours} jours (minimum requis : 7)\n")

# --- 6. Lancer la prévision ---
ref_time = history.index[-1]
forecaster = DCForecaster()
result = forecaster.forecast(reference_time=ref_time, history=history, horizon_slots=96)

# --- 7. Résultats ---
print(f"Prévision pour {ZONE} ({CHARGER_TYPE}) à partir de {ref_time} :\n")
for slot in result["forecastData"]:
    print(
        f"{slot['timeSlotStart']}  "
        f"arrivals={slot['expectedArrivalCount']:.3f}  "
        f"energy={slot['expectedEnergyKwh']:.1f} kWh  "
        f"p_arrival={slot['_meta']['pArrivalActive']:.3f}"
    )
    