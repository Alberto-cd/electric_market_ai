"""BiLSTM surrogate for the LP market-clearing solver."""

from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..configuration import PathConfiguration
from .ml_dataset import save_json

FEATURE_COLUMNS = ["offer", "limit", "is_generator"]
TARGET_NAMES = ["price", "total_generation", "total_demand"]


def _ordered_entities(base_df: pd.DataFrame) -> list[str]:
    entity_info = base_df[["entity", "is_generator"]].drop_duplicates().copy()
    entity_info["entity_name"] = entity_info["entity"].astype(str)
    entity_info["generator_rank"] = (~entity_info["is_generator"].astype(bool)).astype(
        int
    )
    entity_info = entity_info.sort_values(
        by=["generator_rank", "entity_name"],
        ascending=[True, True],
    )
    return entity_info["entity_name"].tolist()


def _available_estimation_names() -> list[str]:
    if not os.path.isdir(PathConfiguration.ESTIMATIONS_DIR_PATH):
        return []

    estimation_names = []
    for filename in sorted(os.listdir(PathConfiguration.ESTIMATIONS_DIR_PATH)):
        if not filename.endswith(".csv"):
            continue

        estimation_name = os.path.splitext(filename)[0]
        solved_path = _market_lp_results_path(estimation_name)
        if os.path.exists(solved_path):
            estimation_names.append(estimation_name)

    return estimation_names


def _market_lp_results_path(estimation_name: str) -> str:
    _, market_lp_results_path, _ = PathConfiguration.market_lp_paths(estimation_name)
    if os.path.exists(market_lp_results_path):
        return market_lp_results_path

    # Backward compatibility with pre-migration solved layout.
    return os.path.join(
        PathConfiguration.SOLVED_ESTIMATIONS_DIR_PATH,
        estimation_name,
        "lp",
        "results.csv",
    )


def _source_dataset_signature(
    estimation_names: Iterable[str],
) -> dict[str, dict[str, float]]:
    signature: dict[str, dict[str, float]] = {}

    for estimation_name in estimation_names:
        base_path = os.path.join(
            PathConfiguration.ESTIMATIONS_DIR_PATH, f"{estimation_name}.csv"
        )
        solved_path = _market_lp_results_path(estimation_name)
        signature[estimation_name] = {
            "base_mtime": os.path.getmtime(base_path),
            "solved_mtime": os.path.getmtime(solved_path),
        }

    return signature


def _load_preprocessing_cache(
    preprocessing_path: str,
    estimation_names: list[str],
    test_size: float,
    val_size: float,
    random_state: int,
) -> dict[str, Any] | None:
    if not os.path.exists(preprocessing_path):
        return None

    with open(preprocessing_path, "rb") as file:
        cached_state = pickle.load(file)

    if not isinstance(cached_state, dict):
        return None

    expected_signature = _source_dataset_signature(estimation_names)
    if cached_state.get("cache_version") != 1:
        return None

    if cached_state.get("dataset_names") != estimation_names:
        return None

    if cached_state.get("split_params") != {
        "test_size": test_size,
        "val_size": val_size,
        "random_state": random_state,
    }:
        return None

    if cached_state.get("source_signature") != expected_signature:
        return None

    return cached_state


def _save_preprocessing_cache(
    preprocessing_path: str,
    estimation_names: list[str],
    samples: list[BiLSTMSample],
    train_names: list[str],
    val_names: list[str],
    test_names: list[str],
    input_scaler: StandardScaler,
    target_scaler: StandardScaler,
    test_size: float,
    val_size: float,
    random_state: int,
) -> None:
    with open(preprocessing_path, "wb") as file:
        pickle.dump(
            {
                "cache_version": 1,
                "dataset_names": estimation_names,
                "source_signature": _source_dataset_signature(estimation_names),
                "split_params": {
                    "test_size": test_size,
                    "val_size": val_size,
                    "random_state": random_state,
                },
                "samples": samples,
                "input_scaler": input_scaler,
                "target_scaler": target_scaler,
                "feature_columns": FEATURE_COLUMNS,
                "target_names": TARGET_NAMES,
                "train_datasets": train_names,
                "val_datasets": val_names,
                "test_datasets": test_names,
            },
            file,
        )


def _common_hours(base_df: pd.DataFrame, solved_df: pd.DataFrame) -> list[int]:
    hours = sorted(set(base_df["hour"].unique()) & set(solved_df["hour"].unique()))
    if not hours:
        raise ValueError("No common hours found between base and solved datasets.")
    return hours


@dataclass
class BiLSTMSample:
    dataset_name: str
    hour: int
    features: torch.Tensor
    target: torch.Tensor


@dataclass
class BiLSTMForwardOutput:
    predictions: torch.Tensor
    score_logits: torch.Tensor
    selection_weights: torch.Tensor
    participation: torch.Tensor
    marginal_offer: torch.Tensor


class BiLSTMMarketDataset(Dataset[BiLSTMSample]):
    def __init__(self, samples: list[BiLSTMSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> BiLSTMSample:
        return self.samples[index]


def _sequence_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    positions = torch.arange(max_length, device=lengths.device).unsqueeze(0)
    return positions < lengths.unsqueeze(1)


def _masked_softmax(
    logits: torch.Tensor, mask: torch.Tensor, temperature: float
) -> torch.Tensor:
    scaled = logits / max(1e-6, temperature)
    scaled = scaled.masked_fill(~mask, torch.finfo(scaled.dtype).min)
    weights = torch.softmax(scaled, dim=1)
    weights = weights * mask.float()
    normalizer = weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return weights / normalizer


def _load_samples(estimation_names: Iterable[str]) -> list[BiLSTMSample]:
    samples: list[BiLSTMSample] = []

    for estimation_name in estimation_names:
        print(f"Loading BiLSTM data for {estimation_name}...", flush=True)
        base_path = os.path.join(
            PathConfiguration.ESTIMATIONS_DIR_PATH, f"{estimation_name}.csv"
        )
        solved_path = _market_lp_results_path(estimation_name)

        base_df = pd.read_csv(base_path)
        solved_df = pd.read_csv(solved_path)
        entity_order = _ordered_entities(base_df)
        hours = _common_hours(base_df, solved_df)
        print(
            f"  Found {len(hours)} common hours and {len(entity_order)} ordered entities.",
            flush=True,
        )

        for hour in hours:
            base_hour = base_df[base_df["hour"] == hour].copy()
            solved_hour = solved_df[solved_df["hour"] == hour].copy()

            if base_hour.empty or solved_hour.empty:
                continue

            base_indexed = base_hour.set_index("entity")
            sequence_rows: list[list[float]] = []

            for entity in entity_order:
                if entity not in base_indexed.index:
                    raise ValueError(
                        f"Missing entity '{entity}' in dataset '{estimation_name}' hour {hour}."
                    )

                row = base_indexed.loc[entity]
                sequence_rows.append(
                    [
                        float(row["offer"]),
                        float(row["limit"]),
                        float(int(bool(row["is_generator"]))),
                    ]
                )

            token_tensor = torch.tensor(sequence_rows, dtype=torch.float32)
            price = float(solved_hour["price"].iloc[0])
            total_generation = float(
                solved_hour.loc[solved_hour["is_generator"].astype(bool), "taken"].sum()
            )
            total_demand = float(
                solved_hour.loc[
                    ~solved_hour["is_generator"].astype(bool), "taken"
                ].sum()
            )
            target_tensor = torch.tensor(
                [price, total_generation, total_demand], dtype=torch.float32
            )

            samples.append(
                BiLSTMSample(
                    dataset_name=estimation_name,
                    hour=int(hour),
                    features=token_tensor,
                    target=target_tensor,
                )
            )

    if not samples:
        raise FileNotFoundError(
            "No BiLSTM samples could be loaded from the estimations directory."
        )

    return samples


def _split_dataset_names(
    dataset_names: list[str],
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    if len(dataset_names) < 3:
        raise ValueError(
            "At least three datasets are required to split train/val/test."
        )

    train_val_names, test_names = train_test_split(
        dataset_names,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    relative_val = val_size / max(1e-12, (1.0 - test_size))
    train_names, val_names = train_test_split(
        train_val_names,
        test_size=relative_val,
        random_state=random_state,
        shuffle=True,
    )

    return sorted(train_names), sorted(val_names), sorted(test_names)


def _samples_for_datasets(
    samples: list[BiLSTMSample], dataset_names: set[str]
) -> list[BiLSTMSample]:
    return [sample for sample in samples if sample.dataset_name in dataset_names]


def _fit_input_scaler(samples: list[BiLSTMSample]) -> StandardScaler:
    stacked = np.concatenate([sample.features.numpy() for sample in samples], axis=0)
    scaler = StandardScaler()
    scaler.fit(stacked)
    return scaler


def _fit_target_scaler(samples: list[BiLSTMSample]) -> StandardScaler:
    stacked = np.stack([sample.target.numpy() for sample in samples], axis=0)
    scaler = StandardScaler()
    scaler.fit(stacked)
    return scaler


class BiLSTMRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 3,
        selection_temperature: float = 1.5,
    ):
        super().__init__()
        self.selection_temperature = selection_temperature
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.score_head = nn.Linear(hidden_size * 2, 1)
        self.participation_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.price_head = nn.Sequential(
            nn.Linear(hidden_size * 2 + 1, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self, tokens: torch.Tensor, lengths: torch.Tensor
    ) -> BiLSTMForwardOutput:
        packed = pack_padded_sequence(
            tokens, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_outputs, _ = self.lstm(packed)
        encoded, _ = pad_packed_sequence(packed_outputs, batch_first=True)

        mask = _sequence_mask(lengths, encoded.shape[1]).to(encoded.device)
        score_logits = self.score_head(encoded).squeeze(-1)
        selection_weights = _masked_softmax(
            score_logits, mask, self.selection_temperature
        )

        participation = torch.sigmoid(self.participation_head(encoded)).squeeze(-1)
        participation = participation.masked_fill(~mask, 0.0)

        offers = tokens[..., 0]
        limits = tokens[..., 1]
        is_generator = tokens[..., 2]

        cleared_quantity = participation * limits * mask.float()
        total_generation = (cleared_quantity * is_generator).sum(dim=1, keepdim=True)
        total_demand = (cleared_quantity * (1.0 - is_generator)).sum(
            dim=1, keepdim=True
        )

        marginal_offer = (selection_weights * offers).sum(dim=1, keepdim=True)
        marginal_context = (selection_weights.unsqueeze(-1) * encoded).sum(dim=1)
        price_input = torch.cat([marginal_context, marginal_offer], dim=1)
        price = self.price_head(price_input)

        predictions = torch.cat([price, total_generation, total_demand], dim=1)

        return BiLSTMForwardOutput(
            predictions=predictions,
            score_logits=score_logits,
            selection_weights=selection_weights,
            participation=participation,
            marginal_offer=marginal_offer.squeeze(-1),
        )


def _collate_batch(batch: list[BiLSTMSample]) -> dict[str, Any]:
    lengths = torch.tensor(
        [sample.features.shape[0] for sample in batch], dtype=torch.long
    )
    padded_features = pad_sequence(
        [sample.features for sample in batch], batch_first=True
    )
    targets = torch.stack([sample.target for sample in batch], dim=0)
    return {
        "features": padded_features,
        "lengths": lengths,
        "targets": targets,
        "dataset_names": [sample.dataset_name for sample in batch],
        "hours": [sample.hour for sample in batch],
    }


def _target_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for idx, target_name in enumerate(TARGET_NAMES):
        true_values = y_true[:, idx]
        pred_values = y_pred[:, idx]
        metrics[target_name] = {
            "mae": float(mean_absolute_error(true_values, pred_values)),
            "rmse": float(np.sqrt(mean_squared_error(true_values, pred_values))),
            "r2": float(r2_score(true_values, pred_values)),
        }
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


def _tensorboard_run_name(
    batch_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    learning_rate: float,
    selection_temperature: float,
    selection_entropy_weight: float,
    random_state: int,
) -> str:
    return (
        f"bs{batch_size}_"
        f"h{hidden_size}_"
        f"l{num_layers}_"
        f"do{dropout:.2f}_"
        f"lr{learning_rate:.0e}_"
        f"st{selection_temperature:.2f}_"
        f"ent{selection_entropy_weight:.0e}_"
        f"rs{random_state}"
    )


def _training_hparams(
    batch_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    learning_rate: float,
    max_epochs: int,
    patience: int,
    selection_temperature: float,
    selection_entropy_weight: float,
    random_state: int,
) -> dict[str, float | int]:
    return {
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "max_epochs": max_epochs,
        "patience": patience,
        "selection_temperature": selection_temperature,
        "selection_entropy_weight": selection_entropy_weight,
        "random_state": random_state,
    }


def _prediction_frame(
    dataset_names: list[str],
    hours: list[int],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame({"dataset_name": dataset_names, "hour": hours})
    for idx, target_name in enumerate(TARGET_NAMES):
        frame[f"{target_name}_true"] = y_true[:, idx]
        frame[f"{target_name}_pred"] = y_pred[:, idx]
    return frame


def _evaluate_model(
    model: BiLSTMRegressor,
    loader: DataLoader,
    device: torch.device,
    target_scaler: StandardScaler,
) -> tuple[float, np.ndarray, np.ndarray, pd.DataFrame]:
    criterion = nn.MSELoss()
    model.eval()
    # tqdm.write("Running evaluation...")
    total_loss = 0.0
    total_count = 0
    true_batches: list[np.ndarray] = []
    pred_batches: list[np.ndarray] = []
    dataset_names: list[str] = []
    hours: list[int] = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            lengths = batch["lengths"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(features, lengths)
            loss = criterion(outputs.predictions, targets)

            batch_size = targets.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

            true_batches.append(targets.cpu().numpy())
            pred_batches.append(outputs.predictions.cpu().numpy())
            dataset_names.extend(batch["dataset_names"])
            hours.extend(batch["hours"])

    if total_count == 0:
        raise ValueError("Empty evaluation loader.")

    y_true_scaled = np.concatenate(true_batches, axis=0)
    y_pred_scaled = np.concatenate(pred_batches, axis=0)
    y_true = target_scaler.inverse_transform(y_true_scaled)
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    predictions = _prediction_frame(dataset_names, hours, y_true, y_pred)

    return total_loss / total_count, y_true, y_pred, predictions


def _load_trained_artifacts(
    model_path: str, metadata_path: str, preprocessing_path: str, device: torch.device
) -> tuple[BiLSTMRegressor, StandardScaler, StandardScaler, dict[str, Any]]:
    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    with open(preprocessing_path, "rb") as file:
        preprocessing = pickle.load(file)

    input_scaler = preprocessing["input_scaler"]
    target_scaler = preprocessing["target_scaler"]

    model = BiLSTMRegressor(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=int(metadata["hidden_size"]),
        num_layers=int(metadata["num_layers"]),
        dropout=float(metadata["dropout"]),
        output_size=len(TARGET_NAMES),
        selection_temperature=float(metadata["selection_temperature"]),
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, input_scaler, target_scaler, metadata


def _transform_samples_with_scaler(
    source_samples: list[BiLSTMSample], input_scaler: StandardScaler
) -> list[BiLSTMSample]:
    transformed: list[BiLSTMSample] = []
    for sample in source_samples:
        scaled_features = input_scaler.transform(sample.features.numpy())
        transformed.append(
            BiLSTMSample(
                dataset_name=sample.dataset_name,
                hour=sample.hour,
                features=torch.tensor(scaled_features, dtype=torch.float32),
                target=sample.target,
            )
        )
    return transformed


def _export_market_inference_outputs(
    model: BiLSTMRegressor,
    device: torch.device,
    input_scaler: StandardScaler,
    target_scaler: StandardScaler,
    model_path: str,
    metadata: dict[str, Any],
) -> None:
    estimation_names = _available_estimation_names()

    for estimation_name in estimation_names:
        try:
            samples = _load_samples([estimation_name])
        except Exception as exc:
            print(f"Skipping {estimation_name}: {exc}")
            continue

        transformed_samples = _transform_samples_with_scaler(samples, input_scaler)
        export_dataset = BiLSTMMarketDataset(transformed_samples)
        export_loader = DataLoader(
            export_dataset,
            batch_size=int(metadata.get("batch_size", 64)),
            shuffle=False,
            collate_fn=_collate_batch,
        )

        prediction_start = time.perf_counter()
        _, y_true, y_pred, predictions = _evaluate_model(
            model, export_loader, device, target_scaler
        )
        prediction_time = time.perf_counter() - prediction_start
        metrics = _target_metrics(y_true, y_pred)

        solved_dir, solved_path, info_path = PathConfiguration.market_ml_paths(
            estimation_name, "bilstm"
        )
        os.makedirs(solved_dir, exist_ok=True)
        predictions.to_csv(solved_path, index=False)

        info = {
            "model_name": "bilstm",
            "estimation_name": estimation_name,
            "model_path": model_path,
            "results_path": solved_path,
            "training_results_path": metadata.get("results_path"),
            "prediction_count": int(len(predictions)),
            "inference_time_seconds": float(prediction_time),
            "inference_time_per_sample_seconds": float(
                prediction_time / max(1, len(predictions))
            ),
            "error_metrics": metrics,
            "error_metrics_summary": _metrics_summary(metrics),
            "hidden_size": metadata.get("hidden_size"),
            "num_layers": metadata.get("num_layers"),
            "dropout": metadata.get("dropout"),
            "selection_temperature": metadata.get("selection_temperature"),
            "selection_entropy_weight": metadata.get("selection_entropy_weight"),
            "batch_size": metadata.get("batch_size"),
            "source_datasets": metadata.get("dataset_names"),
        }
        save_json(info_path, info)


def main(
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    batch_size: int = 64,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    learning_rate: float = 1e-2,
    max_epochs: int = 20,
    patience: int = 8,
    selection_temperature: float = 1.5,
    selection_entropy_weight: float = 1e-3,
    update: bool = True,
):
    output_dir = os.path.join("models", "bilstm")
    preprocessing_path = os.path.join(output_dir, "preprocessing.pkl")
    tensorboard_root_dir = os.path.join(output_dir, "tensorboard")
    results_root_dir = os.path.join(output_dir, "results")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tensorboard_root_dir, exist_ok=True)
    os.makedirs(results_root_dir, exist_ok=True)

    tensorboard_run_name = _tensorboard_run_name(
        batch_size=batch_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        learning_rate=learning_rate,
        selection_temperature=selection_temperature,
        selection_entropy_weight=selection_entropy_weight,
        random_state=random_state,
    )
    tensorboard_dir = os.path.join(tensorboard_root_dir, tensorboard_run_name)
    results_dir = os.path.join(results_root_dir, tensorboard_run_name)
    model_path = os.path.join(results_dir, "model.pt")
    metadata_path = os.path.join(results_dir, "metadata.json")
    predictions_path = os.path.join(results_dir, "predictions.csv")
    history_path = os.path.join(results_dir, "history.csv")

    estimation_names = _available_estimation_names()
    if not estimation_names:
        raise FileNotFoundError(
            "No solved estimation datasets found for BiLSTM training."
        )

    os.makedirs(results_dir, exist_ok=True)

    cached_state = _load_preprocessing_cache(
        preprocessing_path,
        estimation_names,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )

    print(
        f"Found {len(estimation_names)} estimations with solved LP results.",
        flush=True,
    )

    if cached_state is not None:
        print("Loaded BiLSTM preprocessing cache.", flush=True)
        samples = cached_state["samples"]
        train_names = cached_state["train_datasets"]
        val_names = cached_state["val_datasets"]
        test_names = cached_state["test_datasets"]
        input_scaler = cached_state["input_scaler"]
        target_scaler = cached_state["target_scaler"]
    else:
        print("Loading samples...", flush=True)

        samples = _load_samples(estimation_names)
        print(f"Loaded {len(samples)} training samples.", flush=True)
        train_names, val_names, test_names = _split_dataset_names(
            estimation_names,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
        )

    print(
        f"Split datasets -> train={len(train_names)}, val={len(val_names)}, test={len(test_names)}",
        flush=True,
    )

    train_samples = _samples_for_datasets(samples, set(train_names))
    val_samples = _samples_for_datasets(samples, set(val_names))
    test_samples = _samples_for_datasets(samples, set(test_names))

    print(
        "Sample counts -> "
        f"train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}",
        flush=True,
    )

    if not train_samples or not val_samples or not test_samples:
        raise ValueError("BiLSTM split produced an empty partition.")

    if cached_state is None:
        print("Fitting input and target scalers...", flush=True)
        input_scaler = _fit_input_scaler(train_samples)
        target_scaler = _fit_target_scaler(train_samples)
        _save_preprocessing_cache(
            preprocessing_path,
            estimation_names,
            samples,
            train_names,
            val_names,
            test_names,
            input_scaler,
            target_scaler,
            test_size,
            val_size,
            random_state,
        )

    def _transform_samples(source_samples: list[BiLSTMSample]) -> list[BiLSTMSample]:
        transformed: list[BiLSTMSample] = []
        for sample in source_samples:
            scaled_features = input_scaler.transform(sample.features.numpy())
            scaled_target = target_scaler.transform(
                sample.target.numpy().reshape(1, -1)
            )[0]
            transformed.append(
                BiLSTMSample(
                    dataset_name=sample.dataset_name,
                    hour=sample.hour,
                    features=torch.tensor(scaled_features, dtype=torch.float32),
                    target=torch.tensor(scaled_target, dtype=torch.float32),
                )
            )
        return transformed

    train_dataset = BiLSTMMarketDataset(_transform_samples(train_samples))
    val_dataset = BiLSTMMarketDataset(_transform_samples(val_samples))
    test_dataset = BiLSTMMarketDataset(_transform_samples(test_samples))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_batch,
    )

    print(
        "DataLoaders ready -> "
        f"train_batches={len(train_loader)}, val_batches={len(val_loader)}, test_batches={len(test_loader)}",
        flush=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    if not update and os.path.exists(model_path) and os.path.exists(metadata_path):
        print(f"Skipping (already exists): {model_path}")
        model, input_scaler, target_scaler, metadata = _load_trained_artifacts(
            model_path, metadata_path, preprocessing_path, device
        )
    else:
        model = BiLSTMRegressor(
            input_size=len(FEATURE_COLUMNS),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            output_size=len(TARGET_NAMES),
            selection_temperature=selection_temperature,
        ).to(device)

        print(
            "Model configured -> "
            f"hidden_size={hidden_size}, num_layers={num_layers}, dropout={dropout}, "
            f"selection_temperature={selection_temperature}",
            flush=True,
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        writer = SummaryWriter(log_dir=tensorboard_dir)
        writer.add_text("run/name", tensorboard_run_name, 0)
        writer.add_text(
            "run/hparams",
            "\n".join(
                f"{key}={value}"
                for key, value in _training_hparams(
                    batch_size=batch_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    learning_rate=learning_rate,
                    max_epochs=max_epochs,
                    patience=patience,
                    selection_temperature=selection_temperature,
                    selection_entropy_weight=selection_entropy_weight,
                    random_state=random_state,
                ).items()
            ),
            0,
        )
        best_state_dict: dict[str, Any] | None = None
        best_val_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        history_rows: list[dict[str, Any]] = []

        torch.manual_seed(random_state)
        np.random.seed(random_state)

        print("Training pooled BiLSTM surrogate")
        epoch_progress = tqdm(
            range(1, max_epochs + 1),
            desc="Epochs",
            total=max_epochs,
            unit="epoch",
            leave=True,
        )
        for epoch in epoch_progress:
            model.train()
            train_loss_total = 0.0
            train_count = 0

            train_progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch:03d}",
                total=len(train_loader),
                unit="batch",
                leave=False,
            )
            for batch in train_progress:
                features = batch["features"].to(device)
                lengths = batch["lengths"].to(device)
                targets = batch["targets"].to(device)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(features, lengths)
                prediction_loss = criterion(outputs.predictions, targets)
                selection_entropy = (
                    -(
                        outputs.selection_weights.clamp_min(1e-12)
                        * outputs.selection_weights.clamp_min(1e-12).log()
                    )
                    .sum(dim=1)
                    .mean()
                )
                loss = prediction_loss + selection_entropy_weight * selection_entropy
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                batch_size_actual = targets.shape[0]
                train_loss_total += float(loss.item()) * batch_size_actual
                train_count += batch_size_actual

                train_progress.set_postfix(
                    train_loss=f"{train_loss_total / max(1, train_count):.6f}"
                )

            train_loss = train_loss_total / max(1, train_count)
            epoch_progress.set_postfix(train_loss=f"{train_loss:.6f}", refresh=False)
            val_loss, y_true_val, y_pred_val, _ = _evaluate_model(
                model, val_loader, device, target_scaler
            )

            if writer is not None:
                writer.add_scalar("loss/train", train_loss, epoch)
                writer.add_scalar("loss/val", val_loss, epoch)
                writer.add_scalar("learning_rate", learning_rate, epoch)

                _, y_true_train, y_pred_train, _ = _evaluate_model(
                    model, train_loader, device, target_scaler
                )
                train_metrics = _target_metrics(y_true_train, y_pred_train)
                val_metrics = _target_metrics(y_true_val, y_pred_val)
                train_r2 = _metrics_summary(train_metrics)["r2"]
                val_r2 = _metrics_summary(val_metrics)["r2"]

                writer.add_scalar("r2/train", train_r2, epoch)
                writer.add_scalar("r2/val", val_r2, epoch)

            history_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
            )

            epoch_progress.set_postfix(
                train_loss=f"{train_loss:.6f}", val_loss=f"{val_loss:.6f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                epochs_without_improvement = 0
                if writer is not None:
                    writer.add_scalar("loss/best_val", best_val_loss, epoch)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    tqdm.write(f"Early stopping triggered at epoch {epoch}")
                    break

        if best_state_dict is None:
            best_state_dict = model.state_dict()

        model.load_state_dict(best_state_dict)
        model.eval()

        print("Evaluating best checkpoint on test set...", flush=True)
        prediction_start = time.perf_counter()
        test_loss, y_true, y_pred, predictions = _evaluate_model(
            model, test_loader, device, target_scaler
        )
        prediction_time = time.perf_counter() - prediction_start
        metrics = _target_metrics(y_true, y_pred)

        print(
            f"Test complete -> test_loss={test_loss:.6f}, prediction_time={prediction_time:.2f}s",
            flush=True,
        )

        if writer is not None:
            summary_metrics = _metrics_summary(metrics)
            writer.add_hparams(
                _training_hparams(
                    batch_size=batch_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    learning_rate=learning_rate,
                    max_epochs=max_epochs,
                    patience=patience,
                    selection_temperature=selection_temperature,
                    selection_entropy_weight=selection_entropy_weight,
                    random_state=random_state,
                ),
                {
                    "best_val_loss": float(best_val_loss),
                    "test_loss": float(test_loss),
                    "test_mae": float(summary_metrics["mae"]),
                    "test_rmse": float(summary_metrics["rmse"]),
                    "test_r2": float(summary_metrics["r2"]),
                },
            )
            writer.add_scalar("loss/test", test_loss, best_epoch)
            for target_name, target_metrics in metrics.items():
                for metric_name, metric_value in target_metrics.items():
                    writer.add_scalar(
                        f"metrics/{target_name}/{metric_name}",
                        metric_value,
                        best_epoch,
                    )
            writer.add_text(
                "summary",
                f"best_epoch={best_epoch}\nbest_val_loss={best_val_loss:.6f}\ntest_loss={test_loss:.6f}",
                best_epoch,
            )
            writer.flush()
            writer.close()

        with open(model_path, "wb") as file:
            torch.save(model.state_dict(), file)

        predictions.to_csv(predictions_path, index=False)
        pd.DataFrame(history_rows).to_csv(history_path, index=False)

        metadata = {
            "model_name": "bilstm",
            "model_path": model_path,
            "results_root_path": results_root_dir,
            "results_path": results_dir,
            "preprocessing_path": preprocessing_path,
            "predictions_path": predictions_path,
            "history_path": history_path,
            "dataset_names": estimation_names,
            "train_datasets": train_names,
            "val_datasets": val_names,
            "test_datasets": test_names,
            "tensorboard_root_path": tensorboard_root_dir,
            "tensorboard_path": tensorboard_dir,
            "tensorboard_run_name": tensorboard_run_name,
            "feature_columns": FEATURE_COLUMNS,
            "target_names": TARGET_NAMES,
            "target_count": len(TARGET_NAMES),
            "sample_count": len(samples),
            "train_sample_count": len(train_samples),
            "val_sample_count": len(val_samples),
            "test_sample_count": len(test_samples),
            "prediction_count": int(len(predictions)),
            "inference_time_seconds": float(prediction_time),
            "inference_time_per_sample_seconds": float(
                prediction_time / max(1, len(predictions))
            ),
            "batch_size": batch_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "selection_temperature": selection_temperature,
            "selection_entropy_weight": selection_entropy_weight,
            "max_epochs": max_epochs,
            "patience": patience,
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val_loss),
            "test_loss": float(test_loss),
            "error_metrics": metrics,
            "error_metrics_summary": _metrics_summary(metrics),
            "device": str(device),
            "random_state": random_state,
        }
        save_json(metadata_path, metadata)

        print(f"Saved BiLSTM model to: {model_path}")
        print(f"Saved BiLSTM results to: {results_dir}")
        print(f"Saved TensorBoard logs to: {tensorboard_dir}")
        print(f"Saved preprocessing to: {preprocessing_path}")
        print(f"Saved predictions to: {predictions_path}")
        print(f"Saved history to: {history_path}")

    _export_market_inference_outputs(
        model,
        device,
        input_scaler,
        target_scaler,
        model_path,
        metadata,
    )


if __name__ == "__main__":
    main()
