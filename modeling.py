from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import logging
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)

def _fit_log_linear_and_predict(series: pd.Series, target_year: int) -> Dict[str, float]:
    X = series.index.values.astype(float).reshape(-1, 1)
    y = series.values.astype(float)
    positive_mask = np.isfinite(y) & (y > 0)
    if positive_mask.sum() < 2:
        raise ValueError("Not enough positive data points to fit log-linear model.")
    X_train = X[positive_mask]
    y_train = y[positive_mask]
    y_log = np.log(y_train)
    X_offset = X_train.min()
    X_train_centered = X_train - X_offset
    X_full_centered = X - X_offset
    X_target_centered = np.array([[float(target_year) - X_offset]])
    model = LinearRegression()
    model.fit(X_train_centered, y_log)
    r2_log = model.score(X_train_centered, y_log)
    y_log_curve = model.predict(X_full_centered)
    y_curve = np.exp(y_log_curve)
    predicted_log = float(model.predict(X_target_centered)[0])
    predicted = float(np.exp(predicted_log))
    y_train_pred_log = model.predict(X_train_centered)
    y_train_pred = np.exp(y_train_pred_log)
    r2_orig = float(r2_score(y_train, y_train_pred))
    mae = float(mean_absolute_error(y_train, y_train_pred))
    mse = float(mean_squared_error(y_train, y_train_pred))
    rmse = float(np.sqrt(mse))
    return {
        "predicted": predicted,
        "r2_log": float(r2_log),
        "r2_original": r2_orig,
        "mae": mae,
        "rmse": rmse,
        "y_curve": y_curve,
        "predicted_log": predicted_log,
        "x_full": X.flatten(),
        "x_offset": float(X_offset),
    }

def perform_prediction_and_plot(
    data: pd.DataFrame,
    target_year: Optional[int] = None,
    title: str = "Pollution prediction (log-linear)",
    show: bool = True,
    save_path: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    predictions = {}
    if data.empty:
        logger.warning("Empty data provided to prediction function.")
        return predictions
    if target_year is None:
        target_year = int(data.index.max()) + 1
    plt.figure(figsize=(12, 7))
    any_plotted = False
    for col in data.columns:
        try:
            res = _fit_log_linear_and_predict(data[col].dropna(), target_year)
        except ValueError as e:
            logger.info("Skipping %s: %s", col, e)
            continue
        any_plotted = True
        x_full = res["x_full"]
        y_curve = res["y_curve"]
        pred_val = res["predicted"]
        r2_log = res["r2_log"]
        predictions[col] = {
            "predicted": pred_val,
            "r2_log": r2_log,
            "r2_original": res["r2_original"],
            "mae": res["mae"],
            "rmse": res["rmse"],
        }
        plt.plot(x_full, data[col].values, marker="o", linestyle="-", label=f"Historical {col}")
        plt.plot(x_full, y_curve, linestyle="--", label=f"Fit {col}")
        plt.scatter([target_year], [pred_val], marker="X", s=80, label=f"Predicted {col}: {pred_val:.2f}")
        logger.info("Column %s: predicted for %d = %.3f (R2_log = %.3f, R2_orig = %.3f, MAE=%.3f, RMSE=%.3f)",
                    col, target_year, pred_val, r2_log, res["r2_original"], res["mae"], res["rmse"])
    if not any_plotted:
        logger.warning("No pollution series had enough positive data to model.")
    plt.axhline(0, color="red", linestyle="--", linewidth=1, label="Physical zero")
    plt.xlabel("Year")
    plt.ylabel("Pollution")
    plt.title(f"{title} (target_year={target_year})")
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        logger.info("Saved plot to %s", save_path)
    if show:
        plt.show()
    plt.close()
    return predictions

def save_prediction_report(predictions: Dict[str, Dict[str, float]], output_csv: str) -> None:
    rows = []
    for pollutant, stats in predictions.items():
        rows.append({
            "pollutant": pollutant,
            "predicted": stats.get("predicted"),
            "r2_log": stats.get("r2_log"),
            "r2_original": stats.get("r2_original"),
            "mae": stats.get("mae"),
            "rmse": stats.get("rmse"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    logger.info("Saved prediction report to %s", output_csv)
