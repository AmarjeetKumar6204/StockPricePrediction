from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
from yfinance import cache as yf_cache
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

from .config import DEFAULT_START_DATE, LOOKBACK_WINDOW, MODEL_PATH, YFINANCE_CACHE_DIR


@dataclass
class PredictionResult:
    ticker: str
    history: pd.DataFrame
    test_data: pd.DataFrame
    next_close_prediction: float
    latest_close: float
    model_input_rows: int
    forecast_prices: list[float] = None   # future n-day predictions
    forecast_dates: list = None           # corresponding future dates


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [col[0] for col in frame.columns]
    return frame


def load_stock_data(ticker: str, start_date: str = DEFAULT_START_DATE) -> pd.DataFrame:
    YFINANCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    yf_cache.set_cache_location(str(YFINANCE_CACHE_DIR))
    data = yf.download(ticker, start=start_date, progress=False, auto_adjust=False)
    data = _normalize_columns(data)

    if data.empty:
        raise ValueError(f"No market data found for ticker '{ticker}'.")

    data = data.reset_index()
    required_columns = {"Date", "Close"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Downloaded data is missing required columns: {missing}.")

    data = data.dropna(subset=["Close"]).copy()
    data["MA100"] = data["Close"].rolling(100).mean()
    data["MA200"] = data["Close"].rolling(200).mean()
    return data


def build_test_sequences(close_prices: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    close_values = close_prices.to_numpy(dtype=np.float32).reshape(-1, 1)
    if len(close_values) <= LOOKBACK_WINDOW:
        raise ValueError(
            f"At least {LOOKBACK_WINDOW + 1} closing prices are required to make predictions."
        )

    split_index = max(int(len(close_values) * 0.70), LOOKBACK_WINDOW)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close = scaler.fit_transform(close_values)

    x_test: list[np.ndarray] = []
    y_test: list[float] = []

    for idx in range(split_index, len(scaled_close)):
        x_test.append(scaled_close[idx - LOOKBACK_WINDOW : idx])
        y_test.append(close_values[idx, 0])

    if not x_test:
        raise ValueError("Not enough test rows were created from the downloaded data.")

    x_test_array = np.asarray(x_test, dtype=np.float32)
    y_test_array = np.asarray(y_test, dtype=np.float32)
    return x_test_array, y_test_array, scaler


def forecast_future(
    history_close: pd.Series,
    scaler: MinMaxScaler,
    model,
    n_days: int,
    last_date,
) -> tuple[list[float], list]:
    """Iteratively predict `n_days` future closing prices."""
    window = history_close.tail(LOOKBACK_WINDOW).to_numpy(dtype=np.float32).reshape(-1, 1)
    scaled_window = scaler.transform(window).flatten().tolist()

    forecast_prices: list[float] = []
    forecast_dates = []

    current_date = pd.Timestamp(last_date)
    for _ in range(n_days):
        seq = np.array(scaled_window[-LOOKBACK_WINDOW:], dtype=np.float32).reshape(1, LOOKBACK_WINDOW, 1)
        pred_scaled = model.predict(seq, verbose=0)
        pred_price = float(scaler.inverse_transform(pred_scaled)[0, 0])
        forecast_prices.append(pred_price)

        # push prediction back into window
        pred_scaled_val = float(pred_scaled[0, 0])
        scaled_window.append(pred_scaled_val)

        # advance to next trading day (skip weekends)
        current_date += pd.offsets.BDay(1)
        forecast_dates.append(current_date.date())

    return forecast_prices, forecast_dates


def predict_stock(ticker: str, model, n_forecast_days: int = 30) -> PredictionResult:
    history = load_stock_data(ticker)
    x_test, y_test, scaler = build_test_sequences(history["Close"])

    predicted_scaled = model.predict(x_test, verbose=0)
    predicted_close = scaler.inverse_transform(predicted_scaled).flatten()

    split_index = len(history) - len(y_test)
    test_data = history.iloc[split_index:].copy()
    test_data["Actual Close"] = y_test
    test_data["Predicted Close"] = predicted_close

    latest_window = history["Close"].tail(LOOKBACK_WINDOW).to_numpy(dtype=np.float32).reshape(-1, 1)
    next_input = scaler.transform(latest_window).reshape(1, LOOKBACK_WINDOW, 1)
    next_scaled = model.predict(next_input, verbose=0)
    next_close = float(scaler.inverse_transform(next_scaled)[0, 0])

    latest_close = float(history["Close"].iloc[-1])
    last_date = history["Date"].iloc[-1]

    forecast_prices, forecast_dates = forecast_future(
        history["Close"], scaler, model, n_forecast_days, last_date
    )

    return PredictionResult(
        ticker=ticker.upper(),
        history=history,
        test_data=test_data,
        next_close_prediction=next_close,
        latest_close=latest_close,
        model_input_rows=len(history),
        forecast_prices=forecast_prices,
        forecast_dates=forecast_dates,
    )


def load_prediction_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}.")

    return load_model(MODEL_PATH, compile=False)
