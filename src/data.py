import json
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .configuration import PathConfiguration


class DataProcessor:
    def __init__(self, json_path: str, csv_path: str, percent: float = 0.2):
        self.percent = percent
        gen_est, gen_prices, dem_est, dem_prices = self._get_market_curves_estimations(
            json_path, csv_path
        )
        self.dataframe: pd.DataFrame = self._combine_dataframes(
            gen_est, gen_prices, dem_est, dem_prices
        )

        self.columns: list[str] = [
            "is_generator",
            "own",
            "entity",
            "years",
            "hour",
            "offer",
            "limit",
        ]

    @staticmethod
    def normalize_profile(market_estimations, past_market_profiles, key):
        profile_keys = [gen["profile_key"] for gen in market_estimations[key]]
        columns_to_drop = [
            col for col in past_market_profiles.columns if col not in profile_keys
        ]
        filtered_profiles = past_market_profiles.drop(columns=columns_to_drop)

        estimations = pd.DataFrame(index=filtered_profiles.index)
        for gen in market_estimations[key]:
            profile_key = gen["profile_key"]
            annual_gwh = gen["annual_gwh"]
            if profile_key in filtered_profiles.columns:
                # Profile-based
                profile_col = filtered_profiles[profile_key]
                scaling = annual_gwh / profile_col.sum()
                profile_curve = profile_col * scaling
                estimations[gen["name"]] = profile_curve
            else:
                flat_curve = pd.Series(
                    [annual_gwh / 8760] * len(filtered_profiles),
                    index=filtered_profiles.index,
                )
                estimations[gen["name"]] = flat_curve

        prices = pd.DataFrame(
            [
                {
                    "name": gen["name"],
                    "min_price": gen["min_price"],
                    "max_price": gen["max_price"],
                }
                for gen in market_estimations[key]
            ]
        )

        return estimations, prices

    def _get_market_curves_estimations(self, json_path: str, csv_path: str):
        # Read data files
        with open(json_path, "r") as file:
            market_estimations = json.load(file)

        past_market_profiles = pd.read_csv(csv_path)

        # Normalize data
        gen_est, gen_prices = self.normalize_profile(
            market_estimations, past_market_profiles, "generators"
        )
        dem_est, dem_prices = self.normalize_profile(
            market_estimations, past_market_profiles, "demands"
        )

        return gen_est, gen_prices, dem_est, dem_prices

    def _combine_dataframes(self, gen_est, gen_prices, dem_est, dem_prices):
        # Combine generators
        gen_df = (
            gen_est.melt(var_name="name", value_name="limit", ignore_index=False)
            .reset_index()
            .rename(columns={"index": "hour"})
        )
        gen_df["is_generator"] = True
        gen_prices["offer"] = (gen_prices["min_price"] + gen_prices["max_price"]) / 2
        gen_df = pd.merge(gen_df, gen_prices[["name", "offer"]], on="name")

        # Split Solar PV into owned and non-owned portions
        solar_pv_mask = gen_df["name"] == "Solar PV"
        solar_pv_data = gen_df[solar_pv_mask].copy()
        non_solar_data = gen_df[~solar_pv_mask].copy()

        if not solar_pv_data.empty:
            # Create owned solar portion
            solar_owned = solar_pv_data.copy()
            solar_owned["limit"] = solar_owned["limit"] * self.percent
            solar_owned["name"] = "Solar PV (Own)"

            # Update non-owned solar portion
            solar_pv_data["limit"] = solar_pv_data["limit"] * (1 - self.percent)

            # Combine all generator data
            gen_df = pd.concat(
                [non_solar_data, solar_pv_data, solar_owned], ignore_index=True
            )

        # Combine demands
        dem_df = (
            dem_est.melt(var_name="name", value_name="limit", ignore_index=False)
            .reset_index()
            .rename(columns={"index": "hour"})
        )
        dem_df["is_generator"] = False
        dem_prices["offer"] = (dem_prices["min_price"] + dem_prices["max_price"]) / 2
        dem_df = pd.merge(dem_df, dem_prices[["name", "offer"]], on="name")

        # Concatenate and set ownership
        combined_df = pd.concat([gen_df, dem_df], ignore_index=True)
        combined_df["own"] = combined_df["name"] == "Solar PV (Own)"
        combined_df["years"] = 1  # Placeholder
        combined_df = combined_df.rename(columns={"name": "entity"})

        # Reorder and assign
        return combined_df[
            ["is_generator", "own", "entity", "years", "hour", "offer", "limit"]
        ]

    def save_dataframe(self, path: str, decimals: None | int = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = self.dataframe.copy()
        if decimals is not None:
            df = df.round(decimals)
        df.to_csv(path, index=False)


class ElectricityMarketCurvesDataframe:
    def __init__(self, path: str):
        self.columns: list[str] = [
            "is_generator",
            "own",
            "entity",
            "years",
            "hour",
            "offer",
            "limit",
        ]
        self.path: str = path
        self.df: pd.DataFrame = self._read_csv(path)

    def get_entities_name(
        self, is_generator: bool | None = None, own: bool | None = None
    ) -> np.ndarray:
        if own is None:
            if is_generator is None:
                return self.df["entity"].unique()
            else:
                return self.df[self.df["is_generator"] == is_generator][
                    "entity"
                ].unique()
        elif is_generator is None:
            return self.df[self.df["own"] == own]["entity"].unique()

        return self.df[
            (self.df["is_generator"] == is_generator) & (self.df["own"] == own)
        ]["entity"].unique()

    def get_hours(self) -> np.ndarray:
        return self.df["hour"].unique()

    def get_years(self) -> np.ndarray:
        return self.df["years"].unique()

    def get_entities_column_dict(
        self, column: str, is_generator: bool, own: bool | None = None
    ) -> dict:
        assert column in self.columns

        if own is None:
            df = self.df[(self.df["is_generator"] == is_generator)]
        else:
            df = self.df[
                (self.df["is_generator"] == is_generator) & (self.df["own"] == own)
            ]

        return df.set_index(["entity", "hour"])[column].to_dict()

    def get_own_generators_offers(self):
        self.get_entities_column_dict("offer", True)

    def get_own_generators_limit(self):
        self.get_entities_column_dict("limit", True)

    def _read_csv(self, path):
        # df = pd.read_csv(path, sep=";", decimal=",")
        df = pd.read_csv(path)

        assert len(df.columns) == len(self.columns)
        assert set(df.columns) == set(self.columns)

        return df


class ElectricityMarketSolvedDataframe(ElectricityMarketCurvesDataframe):
    def __init__(
        self,
        base_df: ElectricityMarketCurvesDataframe = None,
        taken: dict = None,
        prices: dict = None,
        ppa_percentage: float = 0.0,
        path: str = None,
    ):
        if path is not None:
            self.df = pd.read_csv(path)
            return
        # Copy base columns and add solved columns
        self.columns = base_df.columns + ["taken", "price"]
        # Copy base dataframe
        self.df = base_df.df.copy()
        # Add taken and price columns if provided
        if taken is not None:
            # taken: dict[(entity, hour)] = value
            self.df["taken"] = self.df.apply(
                lambda row: taken.get((row["entity"], row["hour"]), 0.0), axis=1
            )
        else:
            self.df["taken"] = 0.0
        if prices is not None:
            # prices: dict[hour] = value
            self.df["price"] = self.df["hour"].map(lambda h: prices.get(h, 0.0))
        else:
            self.df["price"] = 0.0
        # Change own generators limit to account for the ppa percentage
        if ppa_percentage > 0.0:
            self.df["limit"] = self.df["limit"].astype(float)
            own_mask = (self.df["is_generator"]) & (self.df["own"])
            self.df.loc[own_mask, "limit"] = self.df.loc[own_mask, "limit"] * (
                1 - ppa_percentage
            )

    def save_dataframe(self, path: str, decimals: None | int = None):
        if self.df is None:
            print("Unable to save dataframe: Dataframe not set.")
            return

        df = self.df.copy()
        if decimals is not None:
            df = df.round(decimals)
        df.to_csv(path, index=False)

    def save_market_plots(
        self,
        output_dir: str = "market_plots",
        limit=24,
        show_dashed_lines: bool = False,
    ):
        if self.df is None:
            print("Unable to save market plots: Dataframe not set.")
            return

        # If an output directory is provided, create it. If `None` or empty, show plots instead of saving.
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        hours = (
            list(self.df["hour"].unique())[:limit]
            if limit is not None
            else list(self.df["hour"].unique())
        )
        for hour in hours:
            df_hour = self.df[self.df["hour"] == hour]

            # Separate data by type
            demand_data = df_hour[~df_hour["is_generator"]]
            own_gen_data = df_hour[df_hour["is_generator"] & df_hour["own"]]
            ext_gen_data = df_hour[df_hour["is_generator"] & ~df_hour["own"]]

            # Create bid/offer curves data
            demands_info = [
                [row["limit"], row["offer"]] for _, row in demand_data.iterrows()
            ]

            # Sort data for plotting
            demands_info.sort(
                key=lambda x: x[1], reverse=True
            )  # Sort by bid (descending)

            # Build generators list preserving ownership flag and sort by offer
            generators_info = []
            for _, row in own_gen_data.iterrows():
                if row["limit"] > 0:
                    generators_info.append(
                        {"limit": row["limit"], "offer": row["offer"], "is_own": True}
                    )
            for _, row in ext_gen_data.iterrows():
                generators_info.append(
                    {"limit": row["limit"], "offer": row["offer"], "is_own": False}
                )
            generators_info.sort(key=lambda x: x["offer"])  # Sort by offer (ascending)

            # Plot market curves
            plt.figure(figsize=(10, 6))

            # Plot demand curve (blue)
            x_demand = 0
            last_bid = None
            for limit, bid in demands_info:
                if last_bid is not None:
                    plt.plot([x_demand, x_demand], [last_bid, bid], color="blue")
                plt.plot(
                    [x_demand, x_demand + limit], [bid, bid], color="blue", linewidth=2
                )
                x_demand += limit
                last_bid = bid

            # Plot supply curves
            x_supply = 0
            last_offer = None
            for info in generators_info:
                limit = info["limit"]
                offer = info["offer"]
                color = "green" if info.get("is_own") else "red"
                if last_offer is not None:
                    plt.plot([x_supply, x_supply], [last_offer, offer], color=color)
                plt.plot(
                    [x_supply, x_supply + limit],
                    [offer, offer],
                    color=color,
                    linewidth=2,
                )
                x_supply += limit
                last_offer = offer

            # Add market clearing lines
            max_quantity = (
                max(x_demand, x_supply) if max(x_demand, x_supply) > 0 else 100
            )
            max_price = (
                max(
                    (demands_info[0][1] if demands_info else 100),
                    (generators_info[-1]["offer"] if generators_info else 100),
                )
                * 1.1
            )

            market_price = df_hour["price"].iloc[0]
            supplied_demand = demand_data["taken"].sum()

            if show_dashed_lines:
                plt.axhline(
                    y=market_price,
                    xmax=supplied_demand / max_quantity,
                    color="orange",
                    linestyle="--",
                    linewidth=2,
                    label=f"Market Price: {market_price:.2f}",
                )
                plt.axvline(
                    x=supplied_demand,
                    ymax=market_price / max_price,
                    color="magenta",
                    linestyle="--",
                    linewidth=2,
                )

            # Set plot limits and labels
            min_price = (
                min(
                    (demands_info[-1][1] if demands_info else 0),
                    (generators_info[0]["offer"] if generators_info else 0),
                    0,
                )
                * 1.1
            )
            plt.xlim(0, max_quantity)
            plt.ylim(min_price, max_price)
            plt.xlabel("Quantity (MWh)")
            plt.ylabel("Price (€/MWh)")
            plt.title(f"Market Clearing - Hour {hour}")

            # Custom legend with correct colors and only present categories
            legend_elements = [Line2D([0], [0], color="blue", lw=2, label="Demand")]
            own_count = sum(1 for g in generators_info if g.get("is_own"))
            ext_count = len(generators_info) - own_count
            if own_count > 0:
                legend_elements.append(
                    Line2D([0], [0], color="green", lw=2, label="Own Generators")
                )
            if ext_count > 0:
                legend_elements.append(
                    Line2D([0], [0], color="red", lw=2, label="External Generators")
                )
            if show_dashed_lines:
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        color="orange",
                        lw=2,
                        linestyle="--",
                        label=f"Market Price: {market_price:.2f}",
                    )
                )
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        color="magenta",
                        lw=2,
                        linestyle="--",
                        label=f"Supplied Quantity: {supplied_demand:.2f}",
                    )
                )
            plt.legend(handles=legend_elements)
            plt.grid(True, alpha=0.3)

            if not output_dir:
                plt.show()
            else:
                plot_path = os.path.join(output_dir, f"market_hour_{hour}.png")
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()

    def save_monotone_price_curve(self, output_dir: str):
        if self.df is None:
            print("Unable to save monotone price curve: Dataframe not set.")
            return
        if not output_dir:
            # If no output_dir provided, show the plot instead of saving
            output_dir = None
        else:
            os.makedirs(output_dir, exist_ok=True)
        # Use only rows with valid market price and taken quantity
        df_valid = self.df[(self.df["price"] > 0) & (self.df["taken"] > 0)]
        # Group by hour, sum taken per hour, get market price per hour
        grouped = (
            df_valid.groupby("hour")
            .agg({"price": "first", "taken": "sum"})
            .reset_index()
        )
        # Sort by market price descending
        sorted_grouped = grouped.sort_values(by="price", ascending=False)
        # Compute cumulative quantity
        sorted_grouped["cum_quantity"] = sorted_grouped["taken"].cumsum()
        # Plot monotone curve
        plt.figure(figsize=(10, 6))
        plt.step(
            sorted_grouped["cum_quantity"],
            sorted_grouped["price"],
            where="post",
            color="purple",
            linewidth=2,
        )
        plt.xlabel("Cumulative Quantity (MWh)")
        plt.ylabel("Market Price (€/MWh)")
        plt.title("Monotone Market Price Curve (Year)")
        plt.grid(True, alpha=0.3)
        if output_dir is None:
            plt.show()
        else:
            output_path = os.path.join(output_dir, "monotone_price_curve.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

    def save_price_history(self, output_dir: str):
        """Plot the market price history across all hours of the year."""
        if self.df is None:
            print("Unable to save price history: Dataframe not set.")
            return
        if not output_dir:
            # If no output_dir provided, show the plot instead of saving
            output_dir = None
        else:
            os.makedirs(output_dir, exist_ok=True)

        # Get unique prices per hour (take first price for each hour)
        hourly_prices = (
            self.df[self.df["price"] > 0]
            .groupby("hour")
            .agg({"price": "first"})
            .reset_index()
        )
        hourly_prices = hourly_prices.sort_values(by="hour")

        # Plot price over time
        plt.figure(figsize=(12, 6))
        plt.plot(
            hourly_prices["hour"],
            hourly_prices["price"],
            color="steelblue",
            linewidth=1.5,
            marker="o",
            markersize=3,
        )
        plt.xlabel("Hour of Year")
        plt.ylabel("Market Price (€/MWh)")
        plt.title("Market Price History (Year)")
        plt.grid(True, alpha=0.3)

        if output_dir is None:
            plt.show()
        else:
            output_path = os.path.join(output_dir, "price_history.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

    def save_all_plots(
        self, output_dir: str, update: bool = True, show_dashed_lines: bool = True
    ):
        """Save all individual plots (market plots, monotone curve, and price history).

        Args:
            output_dir: Directory to save plots
            update: If True, regenerate all plots. If False, only create missing plots.
            show_dashed_lines: Whether to show dashed lines in market plots
        """
        if self.df is None:
            print("Unable to save plots: Dataframe not set.")
            return

        os.makedirs(output_dir, exist_ok=True)

        # Check which plots exist (only matters if update=False)
        market_plots_exist = False
        monotone_exists = os.path.exists(
            os.path.join(output_dir, "monotone_price_curve.png")
        )
        price_history_exists = os.path.exists(
            os.path.join(output_dir, "price_history.png")
        )

        if not update:
            # Check if at least one market_hour plot exists
            market_plots_exist = any(
                f.startswith("market_hour_") and f.endswith(".png")
                for f in os.listdir(output_dir)
                if os.path.isfile(os.path.join(output_dir, f))
            )

        # Save plots if needed
        if update or not market_plots_exist:
            self.save_market_plots(output_dir, show_dashed_lines=show_dashed_lines)

        if update or not monotone_exists:
            self.save_monotone_price_curve(output_dir=output_dir)

        if update or not price_history_exists:
            self.save_price_history(output_dir=output_dir)


def main(
    base_estimations: str,
    profile: str,
    estimation_name: str,
    solar_percent=None,
    update: bool = True,
):
    # Profile-derived estimations use a single fixed solar split.
    if solar_percent is None or isinstance(solar_percent, (list, tuple)):
        solar_percent = 0.2

    final_path = os.path.join(
        PathConfiguration.ESTIMATIONS_DIR_PATH, f"{estimation_name}.csv"
    )

    # Check if files already exist and update flag is False
    if not update and os.path.exists(final_path):
        print(f"Skipping (already exists): {final_path}")
        return

    base_est_path = os.path.join(
        PathConfiguration.BASE_ESTIMATIONS_DIR_PATH, base_estimations
    )
    profile_path = os.path.join(PathConfiguration.PROFILES_DIR_PATH, profile)

    print(f"Generating data for solar percent: {solar_percent*100}% -> {final_path}")
    DataProcessor(base_est_path, profile_path, solar_percent).save_dataframe(final_path)
    ElectricityMarketCurvesDataframe(final_path)


if __name__ == "__main__":
    main()
