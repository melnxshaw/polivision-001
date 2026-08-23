"""
Forecasting models + time-ordered backtesting for monthly NBP series.

Models compared in the backtest:
  - Naive (last value carried forward)
  - Seasonal Naive (same month last year)
  - Moving Average (trailing window)
  - Holt-Winters / ETS (trend + additive seasonality)
  - SARIMA (seasonal ARIMA) — statistical, now included in the backtest comparison too
  - XGBoost (gradient boosted trees on lag/rolling features, recursive multi-step)
  - Prophet (additive trend + yearly seasonality model)

Metrics (matching the hackathon brief): MAE, RMSE, WAPE, Bias
"""
import io
import contextlib
import logging
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("statsmodels").setLevel(logging.ERROR)

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    import xgboost as xgb
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False

try:
    from prophet import Prophet
    _HAS_PROPHET = True
except ImportError:
    _HAS_PROPHET = False


# ---------- Metrics ----------

def mae(actual, pred):
    return float(np.mean(np.abs(np.array(actual) - np.array(pred))))

def rmse(actual, pred):
    return float(np.sqrt(np.mean((np.array(actual) - np.array(pred)) ** 2)))

def wape(actual, pred):
    actual = np.array(actual, dtype=float)
    pred = np.array(pred, dtype=float)
    denom = np.sum(np.abs(actual))
    if denom == 0:
        return np.nan
    return float(np.sum(np.abs(actual - pred)) / denom * 100)

def bias(actual, pred):
    actual = np.array(actual, dtype=float)
    pred = np.array(pred, dtype=float)
    denom = np.sum(np.abs(actual))
    if denom == 0:
        return np.nan
    return float(np.sum(pred - actual) / denom * 100)  # % over(+)/under(-) forecast

def all_metrics(actual, pred):
    return {"MAE": mae(actual, pred), "RMSE": rmse(actual, pred),
            "WAPE_%": wape(actual, pred), "Bias_%": bias(actual, pred)}


# ---------- Statistical / baseline models ----------
# Each model fn(train: pd.Series, horizon: int) -> np.array of length horizon

def model_naive(train, horizon):
    return np.repeat(train.iloc[-1], horizon)

def model_seasonal_naive(train, horizon, season=12):
    if len(train) < season:
        return model_naive(train, horizon)
    vals = train.values
    return np.array([vals[-season + (i % season)] for i in range(horizon)])

def model_moving_average(train, horizon, window=3):
    window = min(window, len(train))
    avg = train.iloc[-window:].mean()
    return np.repeat(avg, horizon)

def model_holt_winters(train, horizon, season=12):
    try:
        seasonal = "add" if len(train) >= 2 * season else None
        model = ExponentialSmoothing(
            train, trend="add",
            seasonal=seasonal, seasonal_periods=season if seasonal else None,
            initialization_method="estimated"
        ).fit(optimized=True)
        fc = model.forecast(horizon)
        return np.clip(fc.values, 0, None)
    except Exception:
        return model_seasonal_naive(train, horizon, season)

def _sarima_fit_forecast(train, horizon, season=12, order=(1, 1, 1), seasonal_order=None):
    """Internal: fits SARIMA and returns (mean, lower80, upper80) arrays."""
    so = seasonal_order or (1, 1, 1, season if len(train) >= 2 * season else 0)
    model = SARIMAX(train, order=order, seasonal_order=so,
                     enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)
    fc = fit.get_forecast(horizon)
    mean = fc.predicted_mean.values
    ci = fc.conf_int(alpha=0.2)  # 80% CI
    lower = ci.iloc[:, 0].values
    upper = ci.iloc[:, 1].values
    return np.clip(mean, 0, None), np.clip(lower, 0, None), np.clip(upper, 0, None)

def model_sarima(train, horizon, season=12):
    """Backtest-friendly wrapper: returns just the mean forecast array."""
    try:
        mean, _, _ = _sarima_fit_forecast(train, horizon, season)
        return mean
    except Exception:
        return model_holt_winters(train, horizon, season)

def model_sarima_with_ci(train, horizon, season=12):
    """Used by final_forecast — returns (mean, lower, upper)."""
    try:
        return _sarima_fit_forecast(train, horizon, season)
    except Exception:
        fc = model_holt_winters(train, horizon, season)
        return fc, fc * 0.85, fc * 1.15


# ---------- ML model: XGBoost on lag/rolling features (recursive multi-step) ----------

_XGB_LAGS = [1, 2, 3, 6, 12]

def _build_lag_features(values: np.ndarray, lags=_XGB_LAGS):
    """Builds a supervised (X, y) table from a 1D array using lag features."""
    max_lag = max(lags)
    X, y = [], []
    for t in range(max_lag, len(values)):
        X.append([values[t - lag] for lag in lags])
        y.append(values[t])
    return np.array(X), np.array(y)

def model_xgboost(train, horizon, lags=_XGB_LAGS):
    if not _HAS_XGBOOST:
        return model_seasonal_naive(train, horizon)
    values = train.values.astype(float)
    max_lag = max(lags)
    if len(values) <= max_lag + 5:
        # not enough history for reliable lag features — fall back
        return model_seasonal_naive(train, horizon)
    try:
        X, y = _build_lag_features(values, lags)
        model = xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.08,
                                  subsample=0.9, colsample_bytree=0.9, verbosity=0)
        model.fit(X, y)

        # recursive multi-step forecasting: predicted values feed back in as lags
        history = list(values)
        preds = []
        for _ in range(horizon):
            feat = np.array([[history[-lag] for lag in lags]])
            next_val = float(model.predict(feat)[0])
            preds.append(max(next_val, 0))
            history.append(next_val)
        return np.array(preds)
    except Exception:
        return model_seasonal_naive(train, horizon)


# ---------- ML model: Prophet ----------

def model_prophet(train, horizon):
    if not _HAS_PROPHET:
        return model_seasonal_naive(train, horizon)
    try:
        dfp = pd.DataFrame({"ds": train.index, "y": train.values})
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            m = Prophet(yearly_seasonality=len(train) >= 24, weekly_seasonality=False,
                        daily_seasonality=False)
            m.fit(dfp)
            future = m.make_future_dataframe(periods=horizon, freq="MS")
            fc = m.predict(future)
        preds = fc["yhat"].values[-horizon:]
        return np.clip(preds, 0, None)
    except Exception:
        return model_seasonal_naive(train, horizon)


MODEL_REGISTRY = {
    "Naive": model_naive,
    "Seasonal Naive": model_seasonal_naive,
    "Moving Average (3mo)": model_moving_average,
    "Holt-Winters (ETS)": model_holt_winters,
    "SARIMA": model_sarima,
    "XGBoost": model_xgboost,
    "Prophet": model_prophet,
}


# ---------- Backtesting ----------

def time_ordered_backtest(series: pd.Series, horizon=3, min_train=18, step=1):
    """
    Rolling-origin backtest. Returns a DataFrame of metrics per model,
    averaged across all backtest folds (time-ordered, no shuffling/leakage).
    """
    results = {name: {"MAE": [], "RMSE": [], "WAPE_%": [], "Bias_%": []}
               for name in MODEL_REGISTRY}

    n = len(series)
    fold_starts = list(range(min_train, n - horizon + 1, step))
    if not fold_starts:
        return pd.DataFrame()

    for start in fold_starts:
        train = series.iloc[:start]
        test = series.iloc[start:start + horizon]
        if len(test) < horizon:
            continue
        for name, fn in MODEL_REGISTRY.items():
            try:
                pred = fn(train, horizon)
                m = all_metrics(test.values, pred)
                for k, v in m.items():
                    if not np.isnan(v):
                        results[name][k].append(v)
            except Exception:
                continue

    rows = []
    for name, metrics in results.items():
        row = {"Model": name}
        for k, vals in metrics.items():
            row[k] = round(np.mean(vals), 2) if vals else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("WAPE_%")


def final_forecast(series: pd.Series, horizon=6, season=12):
    """
    Produces the final forward-looking forecast using SARIMA (with CI),
    plus Holt-Winters, XGBoost, and Prophet comparison lines.
    """
    sarima_mean, sarima_lo, sarima_hi = model_sarima_with_ci(series, horizon, season)
    hw = model_holt_winters(series, horizon, season)
    xgb_fc = model_xgboost(series, horizon)
    prophet_fc = model_prophet(series, horizon)

    future_idx = pd.date_range(
        start=series.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"
    )
    out = pd.DataFrame({
        "Month": future_idx.strftime("%Y-%m"),
        "Forecast_SARIMA": sarima_mean,
        "Lower_80": sarima_lo,
        "Upper_80": sarima_hi,
        "Forecast_HoltWinters": hw,
        "Forecast_XGBoost": xgb_fc,
        "Forecast_Prophet": prophet_fc,
    })
    return out
