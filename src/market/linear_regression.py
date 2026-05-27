("""Scikit-learn linear regression baseline for the LP market solver.""")

from __future__ import annotations

import os
import pickle
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..configuration import PathConfiguration
from .ml_dataset import (
    build_solved_dataframe_from_predictions,
    load_all_market_estimation_names,
    load_market_learning_dataset,
    load_pooled_market_learning_dataset,
    save_json,
)


def _model_output_dir() -> str:
    return os.path.join("models", "linear_regression")


def _prediction_frame(
    dataset_names: list[str], hours: list[int], y_true: pd.DataFrame, y_pred: np.ndarray
) -> pd.DataFrame:
    frame = pd.DataFrame({"dataset_name": dataset_names, "hour": hours})
    for idx, column in enumerate(y_true.columns):
        frame[f"{column}_true"] = y_true.iloc[:, idx].to_numpy()
        frame[f"{column}_pred"] = y_pred[:, idx]
    return frame


def _metrics_by_target(
    y_true: pd.DataFrame, y_pred: np.ndarray
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for idx, column in enumerate(y_true.columns):
        true_values = y_true.iloc[:, idx].to_numpy()
        pred_values = y_pred[:, idx]
        mae = float(mean_absolute_error(true_values, pred_values))
        rmse = float(np.sqrt(mean_squared_error(true_values, pred_values)))
        r2 = float(r2_score(true_values, pred_values))
        metrics[column] = {"mae": mae, "rmse": rmse, "r2": r2}
    return metrics


def _metrics_summary(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {"mae": 0.0, "rmse": 0.0, "r2": 0.0}

    return {
        metric_name: float(
            np.mean([values[metric_name] for values in metrics.values()])
        )
        for metric_name in ("mae", "rmse", "r2")
    }


def _coefficient_frame(
    model: LinearRegression, feature_names: list[str], target_names: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    coefficients = np.atleast_2d(model.coef_)
    intercepts = np.atleast_1d(model.intercept_)

    for target_index, target_name in enumerate(target_names):
        for feature_name, coefficient in zip(feature_names, coefficients[target_index]):
            rows.append(
                {
                    "target": target_name,
                    "feature": feature_name,
                    "coefficient": float(coefficient),
                    "abs_coefficient": float(abs(coefficient)),
                }
            )

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["target", "abs_coefficient"], ascending=[True, False])
    frame.attrs["intercepts"] = {
        target: float(intercepts[idx]) for idx, target in enumerate(target_names)
    }
    return frame


def _export_market_inference_outputs(
    model: Pipeline,
    feature_names: list[str],
    target_names: list[str],
    model_path: str,
    coefficients_path: str,
) -> None:
    for estimation_name in load_all_market_estimation_names():
        try:
            dataset = load_market_learning_dataset(estimation_name)
        except Exception as exc:
            print(f"Skipping {estimation_name}: {exc}")
            continue

        if (
            dataset.feature_names != feature_names
            or dataset.target_names != target_names
        ):
            print(
                f"Skipping {estimation_name}: feature/target schema does not match the trained model."
            )
            continue

        prediction_start = time.perf_counter()
        y_pred = model.predict(dataset.feature_frame[feature_names])
        prediction_time = time.perf_counter() - prediction_start

        y_true = dataset.target_frame[target_names]
        metrics = _metrics_by_target(y_true, y_pred)
        solved_df = build_solved_dataframe_from_predictions(dataset, y_pred)

        solved_dir, solved_path, info_path = PathConfiguration.market_ml_paths(
            estimation_name, "linear_regression"
        )
        os.makedirs(solved_dir, exist_ok=True)
        solved_df.save_dataframe(solved_path)

        info = {
            "model_name": "linear_regression",
            "estimation_name": estimation_name,
            "base_path": dataset.base_path,
            "solved_path": dataset.solved_path,
            "model_path": model_path,
            "coefficients_path": coefficients_path,
            "results_path": solved_path,
            "entity_order": dataset.entity_order,
            "feature_names": feature_names,
            "target_names": target_names,
            "row_count": len(dataset.feature_frame),
            "prediction_count": int(len(y_pred)),
            "inference_time_seconds": float(prediction_time),
            "inference_time_per_sample_seconds": float(
                prediction_time / max(1, len(y_pred))
            ),
            "error_metrics": metrics,
            "error_metrics_summary": _metrics_summary(metrics),
        }
        save_json(info_path, info)


def main(
    test_size: float = 0.2,
    random_state: int = 42,
    update: bool = True,
):
    dataset = load_pooled_market_learning_dataset()
    output_dir = _model_output_dir()
    model_path = os.path.join(output_dir, "model.pkl")
    metadata_path = os.path.join(output_dir, "metadata.json")
    predictions_path = os.path.join(output_dir, "predictions.csv")
    coefficients_path = os.path.join(output_dir, "coefficients.csv")

    os.makedirs(output_dir, exist_ok=True)

    if not update and os.path.exists(model_path) and os.path.exists(metadata_path):
        print(f"Skipping (already exists): {model_path}")
        with open(model_path, "rb") as file:
            model = pickle.load(file)
    else:
        split = dataset.split(test_size=test_size, random_state=random_state)
        X_train = split["X_train"].drop(columns=["hour", "dataset_name"])
        X_test = split["X_test"].drop(columns=["hour", "dataset_name"])
        y_train = split["y_train"].drop(columns=["hour", "dataset_name"])
        y_test = split["y_test"].drop(columns=["hour", "dataset_name"])

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression()),
            ]
        )

        print("Training pooled linear regression baseline")
        model.fit(X_train, y_train)
        prediction_start = time.perf_counter()
        y_pred = model.predict(X_test)
        prediction_time = time.perf_counter() - prediction_start

        metrics = _metrics_by_target(y_test, y_pred)
        predictions = _prediction_frame(
            split["X_test"]["dataset_name"].tolist(),
            split["test_hours"],
            y_test,
            y_pred,
        )
        coefficients = _coefficient_frame(
            model.named_steps["regressor"],
            dataset.feature_names,
            dataset.target_names,
        )

        with open(model_path, "wb") as file:
            pickle.dump(model, file)

        predictions.to_csv(predictions_path, index=False)
        coefficients.to_csv(coefficients_path, index=False)

        metadata = dataset.metadata(
            test_size=test_size,
            random_state=random_state,
            model_name="linear_regression",
        )
        metadata.update(
            {
                "model_path": model_path,
                "predictions_path": predictions_path,
                "coefficients_path": coefficients_path,
                "prediction_count": int(len(y_pred)),
                "inference_time_seconds": float(prediction_time),
                "inference_time_per_sample_seconds": float(
                    prediction_time / max(1, len(y_pred))
                ),
                "error_metrics": metrics,
                "error_metrics_summary": _metrics_summary(metrics),
                "intercepts": coefficients.attrs.get("intercepts", {}),
            }
        )
        save_json(metadata_path, metadata)

    print(f"Saved linear regression model to: {model_path}")
    print(f"Saved predictions to: {predictions_path}")
    print(f"Saved coefficients to: {coefficients_path}")

    _export_market_inference_outputs(
        model,
        dataset.feature_names,
        dataset.target_names,
        model_path,
        coefficients_path,
    )


if __name__ == "__main__":
    main()
