import numpy as np
import pandas as pd
from predictor import DCForecaster

# --- build a dummy history buffer (10 days, one arrival per weekday at 07:00) ---
ref_time = pd.Timestamp.now().floor("15min")
slots = pd.date_range(end=ref_time - pd.Timedelta(minutes=15), periods=10*96, freq="15min")
history = pd.DataFrame(0.0, index=slots, columns=["arrival_count", "departure_count", "avg_energy_kWh"])
for ts in slots:
    if ts.dayofweek < 5 and ts.hour == 7 and ts.minute == 0:
        history.at[ts, "arrival_count"]  = 1.0
        history.at[ts, "avg_energy_kWh"] = 35.0

# --- run a 1-hour forecast ---
forecaster = DCForecaster()
result = forecaster.forecast(reference_time=ref_time, history=history, horizon_slots=4)

for slot in result["forecastData"]:
    print(
        f"{slot['timeSlotStart']}  "
        f"arrivals={slot['expectedArrivalCount']:.3f}  "
        f"energy={slot['expectedEnergyKwh']:.1f} kWh  "
        f"p_arrival={slot['_meta']['pArrivalActive']:.3f}"
    )