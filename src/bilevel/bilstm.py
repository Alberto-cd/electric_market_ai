import json
import os
import pickle
from time import perf_counter
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..configuration import ExecutionConfiguration, PathConfiguration
from ..data import ElectricityMarketCurvesDataframe, ElectricityMarketSolvedDataframe
from ..market.bilstm import BiLSTMRegressor


class BiLSTMSurrogate:
    """Inference wrapper for a frozen BiLSTM surrogate."""

    def __init__(self, model_path: str, device: Optional[str] = None):
        if device is None:
            device = "cpu"
        self.device = device
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.backend = "generic"
        self.input_scaler = None
        self.target_scaler = None
        self.model_metadata = {}

        loaded_object = torch.load(model_path, map_location=self.device)
        if isinstance(loaded_object, dict):
            self._load_market_backend(model_path, loaded_object)
        else:
            self.model = loaded_object
            try:
                self.model.eval()
            except Exception:
                pass
            if hasattr(self.model, "parameters"):
                for p in self.model.parameters():
                    p.requires_grad = False

    def _load_market_backend(self, model_path: str, state_dict: dict) -> None:
        """Load the trained market BiLSTM checkpoint and preprocessing cache."""
        run_dir = os.path.dirname(model_path)
        metadata_path = os.path.join(run_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Missing metadata for BiLSTM checkpoint: {metadata_path}"
            )

        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        preprocessing_candidates = []
        preprocessing_path = metadata.get("preprocessing_path")
        if isinstance(preprocessing_path, str) and preprocessing_path:
            preprocessing_candidates.append(os.path.normpath(preprocessing_path))
        preprocessing_candidates.append(
            os.path.join(os.path.dirname(os.path.dirname(run_dir)), "preprocessing.pkl")
        )
        preprocessing_candidates.append(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(run_dir))),
                "preprocessing.pkl",
            )
        )

        resolved_preprocessing_path = None
        for candidate in preprocessing_candidates:
            if candidate and os.path.exists(candidate):
                resolved_preprocessing_path = candidate
                break

        if resolved_preprocessing_path is None:
            raise FileNotFoundError(
                "Could not find BiLSTM preprocessing.pkl next to the checkpoint or metadata."
            )

        with open(resolved_preprocessing_path, "rb") as file:
            preprocessing = pickle.load(file)

        self.input_scaler = preprocessing["input_scaler"]
        self.target_scaler = preprocessing["target_scaler"]
        self.model_metadata = metadata

        hidden_size = int(metadata.get("hidden_size", 64))
        num_layers = int(metadata.get("num_layers", 2))
        dropout = float(metadata.get("dropout", 0.1))
        selection_temperature = float(metadata.get("selection_temperature", 1.5))

        self.model = BiLSTMRegressor(
            input_size=3,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            output_size=3,
            selection_temperature=selection_temperature,
        ).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.backend = "market"

    def _scale_features(self, features: torch.Tensor) -> torch.Tensor:
        if self.input_scaler is None:
            return features

        mean = torch.tensor(
            self.input_scaler.mean_, dtype=features.dtype, device=features.device
        )
        scale = torch.tensor(
            self.input_scaler.scale_, dtype=features.dtype, device=features.device
        )
        return (features - mean) / scale

    def _inverse_price(self, price_scaled: torch.Tensor) -> torch.Tensor:
        if self.target_scaler is None:
            return price_scaled

        mean = torch.tensor(
            self.target_scaler.mean_[0],
            dtype=price_scaled.dtype,
            device=price_scaled.device,
        )
        scale = torch.tensor(
            self.target_scaler.scale_[0],
            dtype=price_scaled.dtype,
            device=price_scaled.device,
        )
        return price_scaled * scale + mean

    def _build_hour_tensor(self, df: ElectricityMarketCurvesDataframe):
        hours = sorted(df.get_hours())
        samples = []
        entity_orders = []
        own_masks = []
        capacities = []

        for h in hours:
            df_h = df.df[df.df["hour"] == h]
            own_gens = df_h[(df_h["is_generator"]) & (df_h["own"])].copy()
            ext_gens = df_h[(df_h["is_generator"]) & (~df_h["own"])].copy()
            dems = df_h[~df_h["is_generator"]].copy()

            ordered = pd.concat([own_gens, ext_gens, dems], ignore_index=True)
            entity_orders.append(list(ordered["entity"]))
            samples.append(
                ordered[["offer", "limit", "is_generator"]].to_numpy(dtype=float)
            )
            own_masks.append(ordered[["own"]].to_numpy(dtype=float).reshape(-1))
            capacities.append(ordered[["limit"]].to_numpy(dtype=float).reshape(-1))

        max_len = max(s.shape[0] for s in samples)
        batch = np.zeros((len(samples), max_len, 3), dtype=float)
        mask = np.zeros((len(samples), max_len), dtype=float)
        own_mask = np.zeros((len(samples), max_len), dtype=float)
        capacities_arr = np.zeros((len(samples), max_len), dtype=float)

        for i, s in enumerate(samples):
            length = s.shape[0]
            batch[i, :length, :] = s
            mask[i, :length] = 1.0
            own_mask[i, :length] = own_masks[i]
            capacities_arr[i, :length] = capacities[i]

        return (
            torch.tensor(batch, dtype=torch.float32, device=self.device),
            torch.tensor(mask, dtype=torch.float32, device=self.device),
            hours,
            entity_orders,
            torch.tensor(own_mask, dtype=torch.float32, device=self.device),
            torch.tensor(capacities_arr, dtype=torch.float32, device=self.device),
        )

    def predict(
        self, df: ElectricityMarketCurvesDataframe, ppa_percentage: float = 0.0
    ) -> ElectricityMarketSolvedDataframe:
        df_copy = ElectricityMarketCurvesDataframe(df.path)
        if ppa_percentage > 0.0:
            own_mask = (df_copy.df["is_generator"]) & (df_copy.df["own"])  # type: ignore
            df_copy.df.loc[own_mask, "limit"] = df_copy.df.loc[
                own_mask, "limit"
            ].astype(float) * (1 - float(ppa_percentage))

        batch, mask, hours, entity_orders, own_mask, capacities = (
            self._build_hour_tensor(df_copy)
        )

        if self.backend == "market":
            lengths = mask.sum(dim=1).long()
            scaled_batch = self._scale_features(batch)
            with torch.no_grad():
                out = self.model(scaled_batch, lengths)

            price_scaled = out.predictions[:, 0]
            price_pred = self._inverse_price(price_scaled)
            participation = out.participation

            adjusted_limits = batch[:, :, 1]
            taken_tensor = participation * adjusted_limits

            taken = {}
            prices = {}
            for i, h in enumerate(hours):
                prices[h] = float(price_pred[i].detach().cpu().item())
                order = entity_orders[i]
                for j, ent in enumerate(order):
                    taken[(ent, h)] = float(taken_tensor[i, j].detach().cpu().item())
        else:
            with torch.no_grad():
                out = self.model(batch)

            if isinstance(out, (tuple, list)):
                taken_pred, price_pred = out[0], out[1]
            else:
                if out.ndim == 3:
                    taken_pred = out[:, :, 0]
                    price_pred = out[:, 0, 1]
                else:
                    raise RuntimeError(
                        "Unexpected model output shape for surrogate model"
                    )

            taken_np = taken_pred.cpu().numpy()
            price_np = price_pred.cpu().numpy().reshape(-1)

            taken = {}
            prices = {}
            for i, h in enumerate(hours):
                prices[h] = float(price_np[i])
                order = entity_orders[i]
                for j, ent in enumerate(order):
                    taken[(ent, h)] = float(taken_np[i, j])

        return ElectricityMarketSolvedDataframe(
            df_copy, taken=taken, prices=prices, ppa_percentage=ppa_percentage
        )

    def predict_from_path(
        self, path: str, ppa_percentage: float = 0.0
    ) -> ElectricityMarketSolvedDataframe:
        return self.predict(ElectricityMarketCurvesDataframe(path), ppa_percentage)

    def optimize_ppa_for_price(
        self,
        df: ElectricityMarketCurvesDataframe,
        ppa_price: float,
        epochs: int = 100,
        lr: float = 0.1,
        tol: float = 1,
        verbose: bool = False,
        tb_logdir: str | None = None,
    ) -> tuple[float, float]:
        """Optimize a differentiable `ppa_percentage` for a given `ppa_price`."""
        batch, mask, _, _, own_mask, capacities = self._build_hour_tensor(df)

        if self.backend == "market":
            lengths = mask.sum(dim=1).long()
            for p in self.model.parameters():
                p.requires_grad = False

            raw = torch.tensor(0.0, requires_grad=True, device=self.device)
            opt = torch.optim.Adam([raw], lr=lr)

            best_pp = None
            best_profit = None
            last_profit = None

            writer = None
            if tb_logdir is not None:
                os.makedirs(tb_logdir, exist_ok=True)
                writer = SummaryWriter(tb_logdir)

            epoch_iter = range(epochs)
            if verbose:
                epoch_iter = tqdm(
                    epoch_iter, desc=f"PPA opt ppa_price={ppa_price}", unit="epoch"
                )

            for epoch in epoch_iter:
                opt.zero_grad()
                ppa = torch.sigmoid(raw)

                offers = batch[..., 0]
                limits = batch[..., 1]
                is_gen = batch[..., 2]
                adjusted_limits = limits * (1.0 - ppa * own_mask)
                adjusted_batch = torch.stack([offers, adjusted_limits, is_gen], dim=-1)
                scaled_batch = self._scale_features(adjusted_batch)

                out = self.model(scaled_batch, lengths)
                price_pred = self._inverse_price(out.predictions[:, 0])
                participation = out.participation

                own_taken = participation * adjusted_limits * own_mask
                market_rev = (price_pred.reshape(-1) * own_taken.sum(dim=1)).sum()
                ppa_rev = ppa * float(ppa_price) * (capacities * own_mask).sum()
                profit = market_rev + ppa_rev

                loss = -profit
                loss.backward()
                opt.step()

                profit_val = float(profit.detach().cpu().item())
                if best_profit is None or profit_val > best_profit:
                    best_profit = profit_val
                    best_pp = float(torch.sigmoid(raw).detach().cpu().item())

                if writer is not None:
                    writer.add_scalar(
                        "ppa/percentage", float(ppa.detach().cpu().item()), epoch
                    )
                    writer.add_scalar("profit/value", profit_val, epoch)
                    writer.add_scalar(
                        "profit/market_rev",
                        float(market_rev.detach().cpu().item()),
                        epoch,
                    )
                    writer.add_scalar(
                        "profit/ppa_rev", float(ppa_rev.detach().cpu().item()), epoch
                    )

                if last_profit is not None and abs(profit_val - last_profit) < tol:
                    if verbose:
                        print(f"Converged at epoch {epoch}, profit={profit_val:.6f}")
                    break
                last_profit = profit_val

            if writer is not None:
                writer.close()

            return best_pp, best_profit

        # Generic fallback for already-scripted custom surrogate models.
        for p in self.model.parameters():
            p.requires_grad = False

        raw = torch.tensor(0.0, requires_grad=True, device=self.device)
        opt = torch.optim.Adam([raw], lr=lr)

        best_pp = None
        best_profit = None
        last_profit = None

        writer = None
        if tb_logdir is not None:
            os.makedirs(tb_logdir, exist_ok=True)
            writer = SummaryWriter(tb_logdir)

        epoch_iter = range(epochs)
        if verbose:
            epoch_iter = tqdm(
                epoch_iter, desc=f"PPA opt ppa_price={ppa_price}", unit="epoch"
            )

        for epoch in epoch_iter:
            opt.zero_grad()
            ppa = torch.sigmoid(raw)

            offers = batch[..., 0]
            limits = batch[..., 1]
            is_gen = batch[..., 2]
            adjusted_limits = limits * (1.0 - ppa * own_mask)
            adjusted_batch = torch.stack([offers, adjusted_limits, is_gen], dim=-1)

            out = self.model(adjusted_batch)
            if isinstance(out, (tuple, list)):
                taken_pred, price_pred = out[0], out[1]
            else:
                if out.ndim == 3:
                    taken_pred = out[:, :, 0]
                    price_pred = out[:, 0, 1]
                else:
                    raise RuntimeError(
                        "Unexpected model output shape for surrogate model"
                    )

            own_taken = taken_pred * own_mask
            market_rev = (price_pred.reshape(-1) * own_taken.sum(dim=1)).sum()
            ppa_rev = ppa * float(ppa_price) * (capacities * own_mask).sum()
            profit = market_rev + ppa_rev

            loss = -profit
            loss.backward()
            opt.step()

            profit_val = float(profit.detach().cpu().item())
            if best_profit is None or profit_val > best_profit:
                best_profit = profit_val
                best_pp = float(torch.sigmoid(raw).detach().cpu().item())

            if writer is not None:
                writer.add_scalar(
                    "ppa/percentage", float(ppa.detach().cpu().item()), epoch
                )
                writer.add_scalar("profit/value", profit_val, epoch)
                writer.add_scalar(
                    "profit/market_rev", float(market_rev.detach().cpu().item()), epoch
                )
                writer.add_scalar(
                    "profit/ppa_rev", float(ppa_rev.detach().cpu().item()), epoch
                )

            if last_profit is not None and abs(profit_val - last_profit) < tol:
                if verbose:
                    print(f"Converged at epoch {epoch}, profit={profit_val:.6f}")
                break
            last_profit = profit_val

        if writer is not None:
            writer.close()

        return best_pp, best_profit


def main(
    estimation_name: str,
    model_path: str,
    ppa_prices: list[float] | None = None,
    update: bool = True,
):
    """Run surrogate optimization for each PPA price and save results."""
    if ppa_prices is None:
        ppa_prices = ExecutionConfiguration.DEFAULT_PPA_PRICES

    path = os.path.join(
        PathConfiguration.ESTIMATIONS_DIR_PATH, f"{estimation_name}.csv"
    )
    # Use the checkpoint directory as model name (folder that contains model.pt)
    model_name = os.path.basename(os.path.dirname(os.path.normpath(model_path)))

    solved_dir, solved_path, info_path = PathConfiguration.bilevel_ml_paths(
        estimation_name, model_name
    )

    if not update and os.path.exists(solved_path) and os.path.exists(info_path):
        print(f"Skipping surrogate bilevel (already exists): {estimation_name}")
        return

    surrogate = BiLSTMSurrogate(model_path)
    df = ElectricityMarketCurvesDataframe(path)

    for pp in ppa_prices:
        out_dir = os.path.join(solved_dir, f"{pp}")
        os.makedirs(out_dir, exist_ok=True)
        solved_path_pp = os.path.join(out_dir, "results.csv")
        info_path_pp = os.path.join(out_dir, "results_info.json")

        if (
            not update
            and os.path.exists(solved_path_pp)
            and os.path.exists(info_path_pp)
        ):
            print(f"Skipping surrogate bilevel (already exists): ppa_price={pp}")
            continue

        tb_dir = os.path.join(out_dir, "tensorboard")
        run_start_time = perf_counter()

        try:
            opt_start_time = perf_counter()
            best_pp, best_profit = surrogate.optimize_ppa_for_price(
                df, pp, tb_logdir=tb_dir, verbose=True
            )
            optimization_time_seconds = float(perf_counter() - opt_start_time)
        except Exception as e:
            print(f"Surrogate optimization failed for ppa_price={pp}: {e}")
            continue

        print(
            f"ppa_price={pp:.3f} optimized_ppa_percentage={best_pp:.6f} profit={best_profit:.3f}"
        )

        try:
            pred_start_time = perf_counter()
            solved_df = surrogate.predict(df, ppa_percentage=best_pp)
            prediction_time_seconds = float(perf_counter() - pred_start_time)
        except Exception as e:
            print(f"Surrogate final predict failed for ppa={pp}: {e}")
            continue

        solved_df.save_dataframe(solved_path_pp)

        execution_time_seconds = float(perf_counter() - run_start_time)
        sample_count = int(solved_df.df["hour"].nunique())
        execution_time_per_sample_seconds = (
            execution_time_seconds / float(sample_count) if sample_count > 0 else None
        )

        results_info = {
            "ppa_price": pp,
            "optimized_ppa_percentage": best_pp,
            "optimized_profit": best_profit,
            "model": model_name,
            "ppa_prices": ppa_prices,
            "optimization_time_seconds": optimization_time_seconds,
            "prediction_time_seconds": prediction_time_seconds,
            "execution_time_seconds": execution_time_seconds,
            "sample_count": sample_count,
            "execution_time_per_sample_seconds": execution_time_per_sample_seconds,
        }

        with open(info_path_pp, "w", encoding="utf-8") as f:
            json.dump(results_info, f, indent=4)


if __name__ == "__main__":
    main()
