"""
Tests de l'API DCForecaster.

On mocke DCForecaster pour ne pas dépendre des vrais fichiers modèles
(utile en CI où models/dc_v2/*.json ne sont pas forcément versionnés).
"""

import io
from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient


def make_fake_forecast(*args, **kwargs):
    return {
        "forecastData": [
            {
                "timeSlotStart": "2026-06-09T07:00:00",
                "expectedArrivalCount": 0.96,
                "expectedEnergyKwh": 25.5,
                "_meta": {"pArrivalActive": 0.85},
            },
            {
                "timeSlotStart": "2026-06-09T07:15:00",
                "expectedArrivalCount": 0.88,
                "expectedEnergyKwh": 24.1,
                "_meta": {"pArrivalActive": 0.81},
            },
        ]
    }


def _sample_csv_bytes():
    idx = pd.date_range(end=datetime.utcnow(), periods=20, freq="15min")
    df = pd.DataFrame(
        {
            "arrival_count": [0.0] * 20,
            "departure_count": [0.0] * 20,
            "avg_energy_kWh": [0.0] * 20,
        },
        index=idx,
    )
    buf = io.StringIO()
    df.to_csv(buf)
    return buf.getvalue().encode("utf-8")


@patch("predictor.DCForecaster.forecast", side_effect=make_fake_forecast)
@patch("predictor.DCForecaster.__init__", return_value=None)
def test_root(mock_init, mock_forecast):
    from api import app
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


@patch("predictor.DCForecaster.forecast", side_effect=make_fake_forecast)
@patch("predictor.DCForecaster.__init__", return_value=None)
def test_predict(mock_init, mock_forecast):
    from api import app
    client = TestClient(app)
    csv_bytes = _sample_csv_bytes()
    response = client.post(
        "/predict",
        files={"file": ("history.csv", csv_bytes, "text/csv")},
        params={"horizon_slots": 2},
    )
    assert response.status_code == 200
    body = response.json()
    assert "forecastData" in body
    assert len(body["forecastData"]) == 2


@patch("predictor.DCForecaster.forecast", side_effect=make_fake_forecast)
@patch("predictor.DCForecaster.__init__", return_value=None)
def test_predict_missing_columns(mock_init, mock_forecast):
    from api import app
    client = TestClient(app)
    bad_df = pd.DataFrame({"foo": [1, 2, 3]}, index=pd.date_range("2026-01-01", periods=3, freq="15min"))
    buf = io.StringIO()
    bad_df.to_csv(buf)
    response = client.post(
        "/predict",
        files={"file": ("bad.csv", buf.getvalue().encode("utf-8"), "text/csv")},
    )
    assert response.status_code == 400
