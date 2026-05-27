from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..configuration import PathConfiguration
from ..data import ElectricityMarketSolvedDataframe


class MarketVisualizer:
    """Plots the LP baseline market results for a solved estimation."""

    def __init__(self, estimation_name: str, update: bool = True):
        if estimation_name is None:
            raise ValueError("estimation_name is required")
        self.estimation_name = estimation_name
        self.update = update
        self.results_path = PathConfiguration.market_lp_paths(self.estimation_name)[1]
        self.solved_df = self._load_solved_df()

    def _load_solved_df(self):
        if not os.path.exists(self.results_path):
            print(f"Directory not found: {self.results_path}")
            return None
        try:
            return ElectricityMarketSolvedDataframe(path=self.results_path)
        except Exception as exc:
            print(f"Warning: Could not load {self.results_path}: {exc}")
            return None

    def plot_individual_results(self):
        if self.solved_df is None:
            print("No LP result available to plot.")
            return

        images_dir = os.path.join(
            PathConfiguration.IMAGES_DIR_PATH,
            self.estimation_name,
            "lp",
        )
        print(f"Saving LP baseline plots for {self.estimation_name}")
        self.solved_df.save_all_plots(
            images_dir, update=self.update, show_dashed_lines=True
        )


@dataclass
class ComparisonRecord:
    model_name: str
    display_name: str
    error_metrics: dict[str, float]
    time_seconds: float | None
    time_per_sample_seconds: float | None
    sample_count: int | None
    source: str


class MarketComparisonVisualizer:
    """Builds model-vs-LP comparison plots for errors and timing."""

    def __init__(self):
        self.records = self._load_records()

    @staticmethod
    def _load_json(path: str) -> dict[str, Any] | None:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _normalized_metrics(metadata: dict[str, Any] | None) -> dict[str, float]:
        if not metadata:
            return {"mae": 0.0, "rmse": 0.0, "r2": 1.0}

        summary = metadata.get("error_metrics_summary") or {}
        if summary:
            return {
                "mae": float(summary.get("mae", 0.0)),
                "rmse": float(summary.get("rmse", 0.0)),
                "r2": float(summary.get("r2", 0.0)),
            }

        metrics = metadata.get("error_metrics") or {}
        if not metrics:
            return {"mae": 0.0, "rmse": 0.0, "r2": 1.0}

        values = {"mae": [], "rmse": [], "r2": []}
        for target_metrics in metrics.values():
            for key in values:
                if key in target_metrics:
                    values[key].append(float(target_metrics[key]))

        return {
            key: float(np.mean(items)) if items else (1.0 if key == "r2" else 0.0)
            for key, items in values.items()
        }

    @staticmethod
    def _time_fields(
        metadata: dict[str, Any] | None,
    ) -> tuple[float | None, float | None, int | None]:
        if not metadata:
            return None, None, None

        time_seconds = metadata.get("inference_time_seconds")
        if time_seconds is None:
            time_seconds = metadata.get("prediction_time_seconds")
        if time_seconds is None:
            time_seconds = metadata.get("execution_time_seconds")

        per_sample = metadata.get("inference_time_per_sample_seconds")
        if per_sample is None:
            per_sample = metadata.get("prediction_time_per_sample_seconds")

        sample_count = metadata.get("prediction_count")
        if sample_count is None:
            sample_count = metadata.get("test_sample_count")
        if sample_count is None:
            sample_count = metadata.get("sample_count")

        if per_sample is None and time_seconds is not None and sample_count:
            per_sample = float(time_seconds) / float(sample_count)

        if time_seconds is not None:
            time_seconds = float(time_seconds)

        if per_sample is not None:
            per_sample = float(per_sample)

        if sample_count is not None:
            sample_count = int(sample_count)

        return time_seconds, per_sample, sample_count

    @staticmethod
    def _resolved_time_per_sample(record: ComparisonRecord) -> float | None:
        if record.time_per_sample_seconds is not None:
            return float(record.time_per_sample_seconds)
        if record.time_seconds is not None and record.sample_count:
            return float(record.time_seconds) / float(record.sample_count)
        return None

    def _load_records(self) -> list[ComparisonRecord]:
        records: list[ComparisonRecord] = []

        model_specs = [
            (
                "linear_regression",
                "Linear Regression",
                os.path.join("models", "linear_regression", "metadata.json"),
            ),
            (
                "decision_tree",
                "Decision Tree",
                os.path.join("models", "decision_tree", "metadata.json"),
            ),
            (
                "xgboost",
                "XGBoost",
                os.path.join("models", "xgboost", "metadata.json"),
            ),
            (
                "bilstm",
                "BiLSTM",
                os.path.join("models", "bilstm", "metadata.json"),
            ),
        ]

        for model_name, display_name, metadata_path in model_specs:
            metadata = self._load_json(metadata_path)
            if not metadata:
                continue

            time_seconds, per_sample, sample_count = self._time_fields(metadata)
            records.append(
                ComparisonRecord(
                    model_name=model_name,
                    display_name=display_name,
                    error_metrics=self._normalized_metrics(metadata),
                    time_seconds=time_seconds,
                    time_per_sample_seconds=per_sample,
                    sample_count=sample_count,
                    source=metadata_path,
                )
            )

        lp_info_paths: list[str] = []
        solved_root = PathConfiguration.SOLVED_ESTIMATIONS_DIR_PATH
        if os.path.isdir(solved_root):
            for estimation_name in sorted(os.listdir(solved_root)):
                info_path = PathConfiguration.market_lp_paths(estimation_name)[2]
                if os.path.exists(info_path):
                    lp_info_paths.append(info_path)

        if lp_info_paths:
            lp_times: list[float] = []
            lp_per_sample_times: list[float] = []
            lp_samples: list[int] = []

            for info_path in lp_info_paths:
                info = self._load_json(info_path)
                if not info:
                    continue

                time_seconds = info.get("execution_time_seconds")
                if time_seconds is None:
                    continue

                solved_path = info_path.replace("results_info.json", "results.csv")
                sample_count = None
                if os.path.exists(solved_path):
                    try:
                        sample_count = int(len(pd.read_csv(solved_path)))
                    except Exception:
                        sample_count = None

                lp_times.append(float(time_seconds))
                if sample_count:
                    lp_samples.append(sample_count)
                    lp_per_sample_times.append(
                        float(time_seconds) / float(sample_count)
                    )

            if lp_times:
                records.append(
                    ComparisonRecord(
                        model_name="lp_solver",
                        display_name="LP Solver",
                        error_metrics={"mae": 0.0, "rmse": 0.0, "r2": 1.0},
                        time_seconds=float(np.mean(lp_times)),
                        time_per_sample_seconds=(
                            float(np.mean(lp_per_sample_times))
                            if lp_per_sample_times
                            else None
                        ),
                        sample_count=int(np.sum(lp_samples)) if lp_samples else None,
                        source=os.path.join(
                            solved_root, "*", "market", "lp", "results_info.json"
                        ),
                    )
                )

        return records

    def _output_dir(self) -> str:
        return os.path.join(PathConfiguration.IMAGES_DIR_PATH, "market", "comparisons")

    def _records_by_name(self) -> list[ComparisonRecord]:
        preferred_order = [
            "Linear Regression",
            "Decision Tree",
            "XGBoost",
            "BiLSTM",
            "LP Solver",
        ]
        ordered = [
            record
            for name in preferred_order
            for record in self.records
            if record.display_name == name
        ]
        remaining = [
            record
            for record in self.records
            if record.display_name not in preferred_order
        ]
        return ordered + sorted(remaining, key=lambda record: record.display_name)

    @staticmethod
    def _latex_float(value: float | None, digits: int = 6) -> str:
        if value is None:
            return "--"
        return f"{float(value):.{digits}g}"

    @staticmethod
    def _latex_time_ms(value: float | None) -> str:
        if value is None:
            return "--"
        return f"{float(value) * 1000.0:.6g}"

    def save_comparison_table(self, output_dir: str | None = None):
        if not self.records:
            print("No model metadata found for comparison summary.")
            return

        output_dir = output_dir or self._output_dir()
        os.makedirs(output_dir, exist_ok=True)

        rows: list[dict[str, Any]] = []
        for record in self._records_by_name():
            resolved_per_sample = self._resolved_time_per_sample(record)
            rows.append(
                {
                    "model_name": record.model_name,
                    "display_name": record.display_name,
                    "mae": record.error_metrics.get("mae"),
                    "rmse": record.error_metrics.get("rmse"),
                    "r2": record.error_metrics.get("r2"),
                    "time_seconds": record.time_seconds,
                    "time_per_sample_ms": (
                        resolved_per_sample * 1000.0
                        if resolved_per_sample is not None
                        else None
                    ),
                    "sample_count": record.sample_count,
                }
            )

        table = pd.DataFrame(rows)[
            [
                "display_name",
                "mae",
                "rmse",
                "r2",
                "time_seconds",
                "time_per_sample_ms",
                "sample_count",
            ]
        ].rename(
            columns={
                "display_name": "Model",
                "mae": "MAE",
                "rmse": "RMSE",
                "r2": "R2",
                "time_seconds": "Total Time (s)",
                "time_per_sample_ms": "Time / Sample (ms)",
                "sample_count": "Samples",
            }
        )

        latex_path = os.path.join(output_dir, "comparison_summary.tex")
        latex_table = table.to_latex(
            index=False,
            escape=True,
            na_rep="--",
            column_format="lrrrrrr",
            float_format=lambda value: f"{value:.6g}",
            caption="Market model comparison summary.",
            label="tab:market_comparison_summary",
        )

        with open(latex_path, "w", encoding="utf-8") as file:
            file.write(latex_table)

        csv_path = os.path.join(output_dir, "comparison_summary.csv")
        table.to_csv(csv_path, index=False)
        print(f"Saved comparison table to {latex_path}")

    def plot_error_comparison(self, output_dir: str | None = None):
        self.save_comparison_table(output_dir)

    def plot_time_comparison(self, output_dir: str | None = None):
        self.save_comparison_table(output_dir)

    def save_summary_table(self, output_dir: str | None = None):
        self.save_comparison_table(output_dir)

    def plot_all(self):
        output_dir = self._output_dir()
        self.save_comparison_table(output_dir)


def main(estimation_name: str = None, update: bool = True):
    if estimation_name is None:
        solved_estimations_dir = PathConfiguration.SOLVED_ESTIMATIONS_DIR_PATH
        if not os.path.exists(solved_estimations_dir):
            print(f"Directory not found: {solved_estimations_dir}")
            return

        estimation_names = [
            d
            for d in os.listdir(solved_estimations_dir)
            if os.path.isdir(os.path.join(solved_estimations_dir, d))
        ]

        for est_name in estimation_names:
            main(estimation_name=est_name, update=update)

        comparison_viz = MarketComparisonVisualizer()
        comparison_viz.plot_all()
        return

    viz = MarketVisualizer(estimation_name=estimation_name, update=update)
    print(f"\nGenerating LP baseline market plots for: {estimation_name}")
    viz.plot_individual_results()


if __name__ == "__main__":
    main()
