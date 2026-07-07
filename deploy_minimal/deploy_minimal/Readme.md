# DC Charging Forecasting Model — Deployment Guide

Two-stage XGBoost model predicting EV charging session arrivals, departures,
energy demand and session duration at 15-minute resolution.

---

## Package contents

```
predictor.py              Inference engine — the only file you interact with
models/
  dc_v2/
    clf_arrival.json      Stage 1 — arrival event classifier
    clf_departure.json    Stage 1 — departure event classifier
    reg_arrival_count.json    Stage 2 — arrival count regressor
    reg_departure_count.json  Stage 2 — departure count regressor
    reg_energy.json           Stage 2 — energy demand regressor
    reg_duration.json         Stage 2 — session duration regressor
    feature_cols.json     Ordered list of the 34 input features
requirements.txt          Python dependencies
README.md                 This file
```

---

## Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

Requires Python 3.8+.

---

## Step 2 — Verify the installation (smoke test)

Run this once after installing to confirm the models load correctly
and produce valid outputs. No real data needed.

```python
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
```

Expected output (values vary with time of day):

```
2026-06-09T07:00:00  arrivals=0.962  energy=25.5 kWh  p_arrival=0.851
2026-06-09T07:15:00  arrivals=0.881  energy=24.1 kWh  p_arrival=0.812
...
```

If the script runs without error and prints non-zero values → installation OK.

---

## Step 3 — Integrate into your system

### What you need to provide

A `history` DataFrame built from your MQTT rolling buffer:

```python
history = pd.DataFrame(
    data  = ...,                          # your stored slot values
    index = pd.DatetimeIndex(...),        # one timestamp per 15-min slot
    columns = [
        "arrival_count",                  # number of arrivals in that slot
        "departure_count",                # number of departures in that slot
        "avg_energy_kWh",                 # average energy per session (kWh)
    ]
)
```

**Rules:**
- Index must be a `pd.DatetimeIndex` at 15-minute frequency
- Cover at least **7 days** before `reference_time` (15 days recommended)
- Slots with no activity must be filled with `0.0` — not `NaN`
- No need to include future slots — only past history up to `reference_time`

### How to call the forecast

```python
from predictor import DCForecaster
import pandas as pd

# instantiate once at application startup
forecaster = DCForecaster()

# call at each 15-min MQTT tick
reference_time = pd.Timestamp.now().floor("15min")
history        = build_history_from_your_buffer(reference_time)

result = forecaster.forecast(
    reference_time = reference_time,
    history        = history,
    horizon_slots  = 4,    # 4 × 15 min = 1 hour ahead
                           # use 8 for 2 hours, 96 for 24 hours
)
```

### Output format

```json
{
  "forecastTimestamp":    1749456000000,
  "forecastTimestampIso": "2026-06-09T07:00:00",
  "resolutionMinutes":    15,
  "horizonSlots":         4,
  "forecastData": [
    {
      "timeSlotStart":           "2026-06-09T07:00:00",
      "timeSlotEnd":             "2026-06-09T07:15:00",
      "stepsAhead":              1,
      "minutesAhead":            15,
      "expectedArrivalCount":    0.9602,
      "expectedDepartureCount":  0.1411,
      "expectedEnergyKwh":       25.556,
      "expectedDurationMinutes": 41.3,
      "_meta": {
        "pArrivalActive":   0.9562,
        "pDepartureActive": 0.1262
      }
    }
  ]
}
```

| Field | Description |
|---|---|
| `expectedArrivalCount` | Fractional expected number of arriving vehicles |
| `expectedDepartureCount` | Fractional expected number of departing vehicles |
| `expectedEnergyKwh` | Expected average energy per arriving session (kWh) |
| `expectedDurationMinutes` | Expected average session duration (minutes, capped at 240) |
| `pArrivalActive` | Raw classifier probability — P(at least one arrival) |
| `pDepartureActive` | Raw classifier probability — P(at least one departure) |

---

## Notes

- `DCForecaster()` loads all 6 models at instantiation — call it **once**
  at startup, not once per forecast tick.
- The forecast uses an autoregressive rollout: each step feeds its own
  predicted values as lag features for the next step. This is expected
  behaviour — accuracy remains stable up to a 2-hour horizon.
- If your MQTT buffer has a gap (missed slot), fill it with `0.0`.
  Do not forward-fill with the previous slot's value.