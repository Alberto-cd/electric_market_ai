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


@dataclass
class ComparisonRecord:
    ppa_folder: str
    price: float | None
    ppa_percentage: float | None
    error_metrics: dict[str, float]
    time_seconds: float | None
    time_per_sample_seconds: float | None
    sample_count: int | None
    source: str


def _normalized_metrics(
    solved_df: pd.DataFrame, reference_df: pd.DataFrame
) -> dict[str, float]:
    merged = solved_df.merge(
        reference_df[["hour", "entity", "taken"]],
        on=["hour", "entity"],
        how="inner",
        suffixes=("_bilevel", "_lp"),
    )

    if merged.empty:
        return {"mae": 0.0, "rmse": 0.0, "r2": 1.0}

    actual = merged["taken_lp"].to_numpy(dtype=float)
    predicted = merged["taken_bilevel"].to_numpy(dtype=float)

    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))

    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    if denominator == 0.0:
        r2 = 1.0 if np.allclose(actual, predicted) else 0.0
    else:
        r2 = float(1.0 - np.sum((actual - predicted) ** 2) / denominator)

    return {"mae": mae, "rmse": rmse, "r2": r2}


class PPAVisualizer:
    """Auto-discovers bilevel PPA results and provides plotting helpers."""

    def __init__(self, estimation_name: str, update: bool = True):
        if estimation_name is None:
            raise ValueError("estimation_name is required")
        self.estimation_name = estimation_name
        self.update = update
        self.base_dir = os.path.join(
            PathConfiguration.SOLVED_ESTIMATIONS_DIR_PATH, self.estimation_name
        )
        self.results = self._load_all_results()

    @staticmethod
    def _load_json(path: str) -> dict[str, Any] | None:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _load_all_results(self) -> list[ComparisonRecord]:
        results: list[ComparisonRecord] = []
        ppa_base = os.path.join(self.base_dir, "bilevel", "blp")
        if not os.path.exists(ppa_base):
            print(f"Directory not found: {ppa_base}")
            return results

        lp_results_path = PathConfiguration.market_lp_paths(self.estimation_name)[1]
        if not os.path.exists(lp_results_path):
            print(f"LP baseline not found: {lp_results_path}")
            return results

        lp_df = pd.read_csv(lp_results_path)

        for ppa_folder in sorted(os.listdir(ppa_base)):
            folder_path = os.path.join(ppa_base, ppa_folder)
            info_path = os.path.join(folder_path, "results_info.json")
            results_csv_path = os.path.join(folder_path, "results.csv")

            if not os.path.isdir(folder_path) or not os.path.exists(info_path):
                continue

            info = self._load_json(info_path) or {}

            try:
                price = float(ppa_folder)
            except Exception:
                price = info.get("ppa_price") or info.get("price") or None

            ppa_percentage = info.get("ppa_percentage")
            if ppa_percentage is not None:
                ppa_percentage = float(ppa_percentage)

            if not os.path.exists(results_csv_path):
                continue

            try:
                solved_df = ElectricityMarketSolvedDataframe(path=results_csv_path)
            except Exception as exc:
                print(f"Warning: Could not load {results_csv_path}: {exc}")
                continue

            metrics = _normalized_metrics(solved_df.df, lp_df)
            sample_count = int(len(solved_df.df))
            time_seconds = info.get("execution_time_seconds")
            if time_seconds is not None:
                time_seconds = float(time_seconds)

            time_per_sample_seconds = None
            if time_seconds is not None and sample_count:
                time_per_sample_seconds = float(time_seconds) / float(sample_count)

            results.append(
                ComparisonRecord(
                    ppa_folder=ppa_folder,
                    price=price,
                    ppa_percentage=ppa_percentage,
                    error_metrics=metrics,
                    time_seconds=time_seconds,
                    time_per_sample_seconds=time_per_sample_seconds,
                    sample_count=sample_count,
                    source=info_path,
                )
            )

        return sorted(results, key=lambda item: (item.price is None, item.price))

    def _output_dir(self) -> str:
        return os.path.join(
            PathConfiguration.IMAGES_DIR_PATH,
            self.estimation_name,
            "bilevel",
            "blp",
            "comparisons",
        )

    def _records_by_price(self) -> list[ComparisonRecord]:
        return sorted(
            self.results, key=lambda record: (record.price is None, record.price)
        )

    def save_comparison_table(self, output_dir: str | None = None):
        if not self.results:
            print("No bilevel results found for comparison summary.")
            return

        output_dir = output_dir or self._output_dir()
        os.makedirs(output_dir, exist_ok=True)

        rows: list[dict[str, Any]] = []
        for record in self._records_by_price():
            rows.append(
                {
                    "ppa_price": record.price,
                    "mae": record.error_metrics.get("mae"),
                    "rmse": record.error_metrics.get("rmse"),
                    "r2": record.error_metrics.get("r2"),
                    "time_seconds": record.time_seconds,
                    "time_per_sample_ms": (
                        record.time_per_sample_seconds * 1000.0
                        if record.time_per_sample_seconds is not None
                        else None
                    ),
                    "sample_count": record.sample_count,
                }
            )

        table = pd.DataFrame(rows)[
            [
                "ppa_price",
                "mae",
                "rmse",
                "r2",
                "time_seconds",
                "time_per_sample_ms",
                "sample_count",
            ]
        ].rename(
            columns={
                "ppa_price": "PPA Price",
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
            caption=f"Bilevel comparison summary for {self.estimation_name}.",
            label="tab:bilevel_comparison_summary",
        )

        with open(latex_path, "w", encoding="utf-8") as file:
            file.write(latex_table)

        csv_path = os.path.join(output_dir, "comparison_summary.csv")
        table.to_csv(csv_path, index=False)
        print(f"Saved comparison table to {latex_path}")

    def plot_individual_results(self):
        for record in self.results:
            solved_path = os.path.join(
                self.base_dir, "bilevel", "blp", record.ppa_folder, "results.csv"
            )
            if not os.path.exists(solved_path):
                print(f"Skipping plots for {record.ppa_folder} (no solved dataframe)")
                continue

            try:
                solved_df = ElectricityMarketSolvedDataframe(path=solved_path)
            except Exception as exc:
                print(f"Skipping plots for {record.ppa_folder}: {exc}")
                continue

            images_dir = os.path.join(
                PathConfiguration.IMAGES_DIR_PATH,
                self.estimation_name,
                "bilevel",
                "blp",
                record.ppa_folder,
            )
            print(f"Saving plots for {record.ppa_folder}")
            solved_df.save_all_plots(
                images_dir, update=self.update, show_dashed_lines=True
            )

    def plot_price_vs_percentage(self, output_dir: str = None):
        if not self.results:
            print("No results to plot.")
            return

        usable_results = [record for record in self.results if record.price is not None]
        if not usable_results:
            print("No price values available to plot.")
            return

        prices = [record.price for record in usable_results]
        percentages = [
            record.ppa_percentage * 100 if record.ppa_percentage is not None else 0
            for record in usable_results
        ]

        plt.figure(figsize=(10, 6))
        plt.plot(prices, percentages, marker="o", linestyle="-", color="b", linewidth=2)
        plt.xlabel("PPA Price (€/MWh)")
        plt.ylabel("PPA Percentage (%)")
        plt.title(f"PPA Price vs Percentage - {self.estimation_name}")
        plt.grid(True, alpha=0.3)
        plt.ylim(-5, 105)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fname = os.path.join(output_dir, "ppa_price_vs_percentage.png")
            plt.savefig(fname, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {fname}")
            plt.close()


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
        return

    viz = PPAVisualizer(estimation_name=estimation_name, update=update)
    print(f"\nGenerating individual bilevel PPA plots for: {estimation_name}")
    # viz.plot_individual_results()

    print("\nGenerating bilevel comparison table")
    viz.save_comparison_table()

    print("\nGenerating PPA price vs percentage plot")
    images_dir = os.path.join(
        PathConfiguration.IMAGES_DIR_PATH,
        estimation_name,
        "bilevel",
        "blp",
        "comparisons",
    )
    viz.plot_price_vs_percentage(output_dir=images_dir)


def create_bilevel_models_comparison(output_dir: str | None = None):
    """Create a comparison table where errors are computed on the optimized
    PPA percentage (the bilevel decision variable) instead of market outputs.
    The function finds, for each estimation and PPA folder, the BLP reference
    `ppa_percentage` and the model's `optimized_ppa_percentage` produced by
    bilevel ML runs under `data/solved_estimations/<est>/bilevel/ml/<model>/<ppa>/results_info.json`.
    It aggregates per-model MAE/RMSE/R2 and average time per sample.
    """

    solved_root = PathConfiguration.SOLVED_ESTIMATIONS_DIR_PATH

    hours_cache: dict[str, int | None] = {}

    def _hours_from_results_dir(results_dir: str) -> int | None:
        if results_dir in hours_cache:
            return hours_cache[results_dir]

        results_csv_path = os.path.join(results_dir, "results.csv")
        if not os.path.exists(results_csv_path):
            hours_cache[results_dir] = None
            return None

        try:
            df = pd.read_csv(results_csv_path, usecols=["hour"])
            hours_value = int(df["hour"].nunique())
            hours_cache[results_dir] = hours_value
            return hours_value
        except Exception:
            hours_cache[results_dir] = None
            return None

    # map (estimation, ppa_folder) -> (blp_ppa_percentage, time_seconds, hours)
    blp_map: dict[tuple[str, str], tuple[float, float | None, int | None]] = {}
    if os.path.isdir(solved_root):
        for estimation_name in sorted(os.listdir(solved_root)):
            blp_base = os.path.join(solved_root, estimation_name, "bilevel", "blp")
            if not os.path.isdir(blp_base):
                continue
            for ppa_folder in sorted(os.listdir(blp_base)):
                run_dir = os.path.join(blp_base, ppa_folder)
                info_path = os.path.join(run_dir, "results_info.json")
                if not os.path.exists(info_path):
                    continue
                try:
                    with open(info_path, "r", encoding="utf-8") as fh:
                        info = json.load(fh)
                    ppa_pct = info.get("ppa_percentage")
                    if ppa_pct is None:
                        continue

                    time_seconds = info.get("execution_time_seconds")
                    if time_seconds is None:
                        time_seconds = info.get("inference_time_seconds")
                    time_seconds = (
                        float(time_seconds) if time_seconds is not None else None
                    )

                    hours = _hours_from_results_dir(run_dir)

                    blp_map[(estimation_name, ppa_folder)] = (
                        float(ppa_pct),
                        time_seconds,
                        hours,
                    )
                except Exception:
                    continue

    # fallback timing from model metadata (if run-level bilevel timing is unavailable)
    model_fallback_time_per_sample_ms: dict[str, float] = {}
    model_results_root = os.path.join("models", "bilstm", "results")
    if os.path.isdir(model_results_root):
        for model_name in os.listdir(model_results_root):
            metadata_path = os.path.join(
                model_results_root, model_name, "metadata.json"
            )
            if not os.path.exists(metadata_path):
                continue
            try:
                with open(metadata_path, "r", encoding="utf-8") as fh:
                    metadata = json.load(fh)
                per_sample = metadata.get("inference_time_per_sample_seconds")
                if per_sample is None:
                    per_sample = metadata.get("prediction_time_per_sample_seconds")
                if per_sample is not None:
                    model_fallback_time_per_sample_ms[model_name] = (
                        float(per_sample) * 1000.0
                    )
            except Exception:
                continue

    # collect model predictions: model_name -> list of (true, pred, time_seconds, hours)
    model_results: dict[str, list[tuple[float, float, float | None, int | None]]] = {}
    if os.path.isdir(solved_root):
        for estimation_name in sorted(os.listdir(solved_root)):
            ml_base = os.path.join(solved_root, estimation_name, "bilevel", "ml")
            if not os.path.isdir(ml_base):
                continue
            for model_name in sorted(os.listdir(ml_base)):
                model_dir = os.path.join(ml_base, model_name)
                if not os.path.isdir(model_dir):
                    continue
                for ppa_folder in sorted(os.listdir(model_dir)):
                    run_dir = os.path.join(model_dir, ppa_folder)
                    info_path = os.path.join(run_dir, "results_info.json")
                    if not os.path.exists(info_path):
                        continue
                    key = (estimation_name, ppa_folder)
                    if key not in blp_map:
                        # no BLP reference for this sample
                        continue
                    try:
                        with open(info_path, "r", encoding="utf-8") as fh:
                            info = json.load(fh)
                    except Exception:
                        continue

                    pred = info.get("optimized_ppa_percentage") or info.get(
                        "ppa_percentage"
                    )
                    if pred is None:
                        continue
                    pred = float(pred)

                    time_seconds = info.get("execution_time_seconds")
                    if time_seconds is None:
                        time_seconds = info.get("inference_time_seconds")
                    time_seconds = (
                        float(time_seconds) if time_seconds is not None else None
                    )

                    hours = _hours_from_results_dir(run_dir)
                    if hours is None:
                        # fallback to BLP hours for same estimation+ppa folder
                        _, _, blp_hours = blp_map[key]
                        hours = blp_hours

                    true, _, _ = blp_map[key]
                    model_results.setdefault(model_name, []).append(
                        (
                            true,
                            pred,
                            time_seconds,
                            hours,
                        )
                    )

    # Build rows
    rows: list[dict[str, Any]] = []

    blp_total_time = 0.0
    blp_timed_hours = 0
    blp_total_hours = 0
    for _, (_, time_seconds, hours) in blp_map.items():
        if hours is not None and hours > 0:
            blp_total_hours += int(hours)
        if time_seconds is None or hours is None or hours <= 0:
            continue
        blp_total_time += float(time_seconds)
        blp_timed_hours += int(hours)

    # BLP baseline row (zero error vs itself)
    rows.append(
        {
            "Model": "BLP",
            "MAE": 0.0,
            "RMSE": 0.0,
            "R2": 1.0,
            "Time / Sample (ms)": (
                (blp_total_time / float(blp_timed_hours)) * 1000.0
                if blp_timed_hours > 0
                else None
            ),
            "Samples": blp_total_hours,
        }
    )

    # Per-model aggregated metrics
    for model_name, entries in sorted(model_results.items()):
        trues = np.array([t for t, p, _, _ in entries], dtype=float)
        preds = np.array([p for t, p, _, _ in entries], dtype=float)
        maes = float(np.mean(np.abs(trues - preds))) if len(trues) else None
        rmse = float(np.sqrt(np.mean((trues - preds) ** 2))) if len(trues) else None
        denom = float(np.sum((trues - np.mean(trues)) ** 2)) if len(trues) else 0.0
        if denom == 0.0:
            r2 = 1.0 if np.allclose(trues, preds) else 0.0
        else:
            r2 = float(1.0 - np.sum((trues - preds) ** 2) / denom)

        total_time = 0.0
        timed_hours = 0
        total_hours = 0
        for _, _, time_seconds, hours in entries:
            if hours is not None and hours > 0:
                total_hours += int(hours)
            if time_seconds is None or hours is None or hours <= 0:
                continue
            total_time += float(time_seconds)
            timed_hours += int(hours)

        avg_per_sample_ms = (
            (total_time / float(timed_hours)) * 1000.0 if timed_hours > 0 else None
        )
        if avg_per_sample_ms is None:
            avg_per_sample_ms = model_fallback_time_per_sample_ms.get(model_name)

        rows.append(
            {
                "Model": model_name,
                "MAE": maes,
                "RMSE": rmse,
                "R2": r2,
                "Time / Sample (ms)": avg_per_sample_ms,
                "Samples": total_hours,
            }
        )

    df = pd.DataFrame(rows)

    out_dir = output_dir or os.path.join(
        PathConfiguration.IMAGES_DIR_PATH, "bilevel", "comparisons"
    )
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "comparison_summary_models.csv")
    tex_path = os.path.join(out_dir, "comparison_summary_models.tex")

    df.to_csv(csv_path, index=False)

    latex_table = df.to_latex(
        index=False,
        escape=True,
        na_rep="--",
        column_format="lrrrrr",
        float_format=lambda value: f"{value:.6g}",
        caption="Bilevel model comparison summary.",
        label="tab:bilevel_model_comparison_summary",
    )

    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write(latex_table)

    print(f"Saved bilevel models comparison to {csv_path} and {tex_path}")


if __name__ == "__main__":
    main()
