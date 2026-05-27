from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..configuration import PathConfiguration
from ..data import ElectricityMarketCurvesDataframe, ElectricityMarketSolvedDataframe


def _slugify_entity_name(name: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower()
    return slug or "entity"


def _ordered_entities(base_df: pd.DataFrame) -> list[str]:
    entity_info = base_df[["entity", "is_generator", "own"]].drop_duplicates().copy()
    entity_info["entity_name"] = entity_info["entity"].astype(str)
    entity_info["generator_rank"] = (~entity_info["is_generator"].astype(bool)).astype(
        int
    )
    entity_info["own_rank"] = (~entity_info["own"].astype(bool)).astype(int)
    entity_info = entity_info.sort_values(
        by=["generator_rank", "own_rank", "entity_name"],
        ascending=[True, True, True],
    )
    return entity_info["entity_name"].tolist()


def _common_hours(base_df: pd.DataFrame, solved_df: pd.DataFrame) -> list[int]:
    base_hours = set(base_df["hour"].unique())
    solved_hours = set(solved_df["hour"].unique())
    hours = sorted(base_hours & solved_hours)
    if not hours:
        raise ValueError("No common hours found between base and solved datasets.")
    return hours


def _profile_estimation_names() -> list[str]:
    if not os.path.isdir(PathConfiguration.ESTIMATIONS_DIR_PATH):
        return []

    return sorted(
        os.path.splitext(filename)[0]
        for filename in os.listdir(PathConfiguration.ESTIMATIONS_DIR_PATH)
        if filename.startswith("estimations_") and filename.endswith(".csv")
    )


def _all_estimation_names() -> list[str]:
    if not os.path.isdir(PathConfiguration.ESTIMATIONS_DIR_PATH):
        return []

    return sorted(
        os.path.splitext(filename)[0]
        for filename in os.listdir(PathConfiguration.ESTIMATIONS_DIR_PATH)
        if filename.endswith(".csv")
    )


def _build_feature_frame(
    base_df: pd.DataFrame, entity_order: list[str], hours: list[int]
) -> pd.DataFrame:
    features = pd.DataFrame(index=hours)
    for entity in entity_order:
        entity_df = (
            base_df[base_df["entity"] == entity]
            .sort_values("hour")
            .set_index("hour")
            .reindex(hours)
        )

        if entity_df[["offer", "limit", "is_generator", "own"]].isnull().any().any():
            raise ValueError(f"Missing feature values for entity '{entity}'.")

        slug = _slugify_entity_name(entity)
        features[f"{slug}__offer"] = entity_df["offer"].to_numpy()
        features[f"{slug}__limit"] = entity_df["limit"].astype(float).to_numpy()
        features[f"{slug}__is_generator"] = (
            entity_df["is_generator"].astype(int).to_numpy()
        )

    features.index.name = "hour"
    return features.reset_index()


def _build_target_frame(
    solved_df: pd.DataFrame, entity_order: list[str], hours: list[int]
) -> pd.DataFrame:
    solved = solved_df.copy()
    if "taken" not in solved.columns or "price" not in solved.columns:
        raise ValueError("Solved dataset must include 'taken' and 'price' columns.")

    price_series = solved.groupby("hour")["price"].first().reindex(hours)
    if price_series.isnull().any():
        raise ValueError(
            "Missing price values for at least one hour in the solved dataset."
        )

    taken_matrix = solved.pivot(index="hour", columns="entity", values="taken").reindex(
        index=hours, columns=entity_order
    )
    if taken_matrix.isnull().any().any():
        missing_entities = taken_matrix.columns[taken_matrix.isnull().any()].tolist()
        raise ValueError(f"Missing taken values for entities: {missing_entities}")

    targets = pd.DataFrame(index=hours)
    targets["price"] = price_series.to_numpy()
    for entity in entity_order:
        slug = _slugify_entity_name(entity)
        targets[f"{slug}__taken"] = taken_matrix[entity].to_numpy()

    targets.index.name = "hour"
    return targets.reset_index()


def top_feature_correlations(
    features: pd.DataFrame, top_k: int = 20, threshold: float = 0.90
) -> list[dict[str, Any]]:
    numeric = features.drop(columns=["hour"], errors="ignore").select_dtypes(
        include=[np.number]
    )
    if numeric.empty:
        return []

    corr = numeric.corr().abs()
    pairs: list[dict[str, Any]] = []
    columns = list(corr.columns)
    for idx, left in enumerate(columns):
        for right in columns[idx + 1 :]:
            value = float(corr.loc[left, right])
            if value >= threshold:
                pairs.append(
                    {"feature_a": left, "feature_b": right, "abs_correlation": value}
                )

    pairs.sort(key=lambda item: item["abs_correlation"], reverse=True)
    return pairs[:top_k]


def describe_feature_set(features: pd.DataFrame) -> dict[str, Any]:
    numeric = features.drop(columns=["hour"], errors="ignore").select_dtypes(
        include=[np.number]
    )
    return {
        "row_count": int(len(features)),
        "feature_count": int(numeric.shape[1]),
        "missing_values": int(numeric.isnull().sum().sum()),
        "correlations_over_threshold": top_feature_correlations(features),
    }


@dataclass
class PreparedMarketDataset:
    estimation_name: str
    base_path: str
    solved_path: str
    entity_order: list[str]
    feature_frame: pd.DataFrame
    target_frame: pd.DataFrame

    @property
    def feature_names(self) -> list[str]:
        return [column for column in self.feature_frame.columns if column != "hour"]

    @property
    def target_names(self) -> list[str]:
        return [column for column in self.target_frame.columns if column != "hour"]

    @property
    def hours(self) -> list[int]:
        return self.feature_frame["hour"].astype(int).tolist()

    def split(self, test_size: float = 0.2, random_state: int = 42):
        indices = np.arange(len(self.feature_frame))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )

        train_idx = np.sort(train_idx)
        test_idx = np.sort(test_idx)

        X_train = self.feature_frame.iloc[train_idx].reset_index(drop=True)
        X_test = self.feature_frame.iloc[test_idx].reset_index(drop=True)
        y_train = self.target_frame.iloc[train_idx].reset_index(drop=True)
        y_test = self.target_frame.iloc[test_idx].reset_index(drop=True)

        train_hours = X_train["hour"].astype(int).tolist()
        test_hours = X_test["hour"].astype(int).tolist()

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "train_hours": train_hours,
            "test_hours": test_hours,
            "train_indices": train_idx.tolist(),
            "test_indices": test_idx.tolist(),
        }

    def metadata(
        self, test_size: float, random_state: int, model_name: str
    ) -> dict[str, Any]:
        split = self.split(test_size=test_size, random_state=random_state)
        return {
            "model_name": model_name,
            "estimation_name": self.estimation_name,
            "base_path": self.base_path,
            "solved_path": self.solved_path,
            "entity_order": self.entity_order,
            "feature_names": self.feature_names,
            "target_names": self.target_names,
            "feature_count": len(self.feature_names),
            "target_count": len(self.target_names),
            "row_count": len(self.feature_frame),
            "test_size": test_size,
            "random_state": random_state,
            "train_hours": split["train_hours"],
            "test_hours": split["test_hours"],
            "feature_analysis": describe_feature_set(self.feature_frame),
        }


@dataclass
class PooledMarketDataset:
    estimation_names: list[str]
    feature_frame: pd.DataFrame
    target_frame: pd.DataFrame

    @property
    def feature_names(self) -> list[str]:
        return [
            column
            for column in self.feature_frame.columns
            if column not in {"hour", "dataset_name"}
        ]

    @property
    def target_names(self) -> list[str]:
        return [
            column
            for column in self.target_frame.columns
            if column not in {"hour", "dataset_name"}
        ]

    @property
    def dataset_count(self) -> int:
        return len(self.estimation_names)

    @property
    def row_count(self) -> int:
        return len(self.feature_frame)

    def split(self, test_size: float = 0.2, random_state: int = 42):
        if self.dataset_count < 2:
            raise ValueError(
                "At least two estimation datasets are required for pooled training."
            )

        train_names, test_names = train_test_split(
            self.estimation_names,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )

        train_mask = self.feature_frame["dataset_name"].isin(train_names)
        test_mask = self.feature_frame["dataset_name"].isin(test_names)

        X_train = self.feature_frame.loc[train_mask].reset_index(drop=True)
        X_test = self.feature_frame.loc[test_mask].reset_index(drop=True)
        y_train = self.target_frame.loc[train_mask].reset_index(drop=True)
        y_test = self.target_frame.loc[test_mask].reset_index(drop=True)

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "train_datasets": sorted(train_names),
            "test_datasets": sorted(test_names),
            "train_hours": X_train["hour"].astype(int).tolist(),
            "test_hours": X_test["hour"].astype(int).tolist(),
        }

    def metadata(
        self, test_size: float, random_state: int, model_name: str
    ) -> dict[str, Any]:
        split = self.split(test_size=test_size, random_state=random_state)
        return {
            "model_name": model_name,
            "estimation_names": self.estimation_names,
            "dataset_count": self.dataset_count,
            "row_count": self.row_count,
            "feature_names": self.feature_names,
            "target_names": self.target_names,
            "feature_count": len(self.feature_names),
            "target_count": len(self.target_names),
            "test_size": test_size,
            "random_state": random_state,
            "train_datasets": split["train_datasets"],
            "test_datasets": split["test_datasets"],
            "train_hours": split["train_hours"],
            "test_hours": split["test_hours"],
        }


def load_market_learning_dataset(estimation_name: str) -> PreparedMarketDataset:
    base_path = os.path.join(
        PathConfiguration.ESTIMATIONS_DIR_PATH, f"{estimation_name}.csv"
    )
    _, solved_path, _ = PathConfiguration.market_lp_paths(estimation_name)

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base dataset not found: {base_path}")
    if not os.path.exists(solved_path):
        raise FileNotFoundError(f"Solved LP dataset not found: {solved_path}")

    base_df = pd.read_csv(base_path)
    solved_df = pd.read_csv(solved_path)

    entity_order = _ordered_entities(base_df)
    hours = _common_hours(base_df, solved_df)

    feature_frame = _build_feature_frame(base_df, entity_order, hours)
    target_frame = _build_target_frame(solved_df, entity_order, hours)

    return PreparedMarketDataset(
        estimation_name=estimation_name,
        base_path=base_path,
        solved_path=solved_path,
        entity_order=entity_order,
        feature_frame=feature_frame,
        target_frame=target_frame,
    )


def build_solved_dataframe_from_predictions(
    dataset: PreparedMarketDataset, y_pred: np.ndarray
) -> ElectricityMarketSolvedDataframe:
    if y_pred.shape[1] != len(dataset.target_names):
        raise ValueError(
            f"Prediction shape {y_pred.shape} does not match target schema {len(dataset.target_names)}."
        )

    base_df = ElectricityMarketCurvesDataframe(dataset.base_path)
    taken: dict[tuple[str, int], float] = {}
    prices: dict[int, float] = {}

    target_columns = dataset.target_names
    price_index = target_columns.index("price") if "price" in target_columns else 0

    entity_target_pairs = [
        (entity_name, target_name)
        for entity_name, target_name in zip(dataset.entity_order, target_columns[1:])
    ]

    for row_index, hour in enumerate(dataset.hours):
        prices[hour] = float(y_pred[row_index, price_index])
        for entity_name, target_name in entity_target_pairs:
            target_index = target_columns.index(target_name)
            taken[(entity_name, hour)] = float(y_pred[row_index, target_index])

    return ElectricityMarketSolvedDataframe(base_df, taken=taken, prices=prices)


def load_pooled_market_learning_dataset() -> PooledMarketDataset:
    estimation_names = _profile_estimation_names()
    if not estimation_names:
        raise FileNotFoundError(
            "No profile estimations found in the estimations directory."
        )

    feature_frames: list[pd.DataFrame] = []
    target_frames: list[pd.DataFrame] = []
    reference_feature_names: list[str] | None = None
    reference_target_names: list[str] | None = None

    for estimation_name in estimation_names:
        dataset = load_market_learning_dataset(estimation_name)

        if reference_feature_names is None:
            reference_feature_names = dataset.feature_names
            reference_target_names = dataset.target_names
        else:
            if dataset.feature_names != reference_feature_names:
                raise ValueError(
                    f"Feature schema mismatch for dataset '{estimation_name}'."
                )
            if dataset.target_names != reference_target_names:
                raise ValueError(
                    f"Target schema mismatch for dataset '{estimation_name}'."
                )

        feature_frame = dataset.feature_frame.copy()
        target_frame = dataset.target_frame.copy()
        feature_frame.insert(0, "dataset_name", estimation_name)
        target_frame.insert(0, "dataset_name", estimation_name)
        feature_frames.append(feature_frame)
        target_frames.append(target_frame)

    pooled_features = pd.concat(feature_frames, ignore_index=True)
    pooled_targets = pd.concat(target_frames, ignore_index=True)

    return PooledMarketDataset(
        estimation_names=estimation_names,
        feature_frame=pooled_features,
        target_frame=pooled_targets,
    )


def load_all_market_estimation_names() -> list[str]:
    return _all_estimation_names()


def save_json(path: str, data: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
