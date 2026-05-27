import numpy as np
import pandas as pd
import os
from .configuration import PathConfiguration


class RandomDataGenerator:
    """Generate random electricity market estimations."""

    def __init__(
        self,
        num_hours: int = 8760,
        num_generators: int = 5,
        num_demands: int = 3,
        generator_price_range: tuple = (0, 100),
        demand_price_range: tuple = (20, 150),
        generator_capacity_range: tuple = (10, 100),
        demand_capacity_range: tuple = (50, 200),
        num_own_generators: int = 1,
        random_seed: int | None = None,
    ):
        """
        Initialize random data generator.

        Args:
            num_hours: Number of hours to generate data for (default: 8760 for a full year)
            num_generators: Number of external generators
            num_demands: Number of demand entities
            generator_price_range: Tuple (min, max) for generator offer prices
            demand_price_range: Tuple (min, max) for demand bid prices
            generator_capacity_range: Tuple (min, max) for generator capacity limits
            demand_capacity_range: Tuple (min, max) for demand limits
            num_own_generators: Number of owned generators (included in total generators)
            random_seed: Seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        self.num_hours = num_hours
        self.num_generators = num_generators
        self.num_demands = num_demands
        self.generator_price_range = generator_price_range
        self.demand_price_range = demand_price_range
        self.generator_capacity_range = generator_capacity_range
        self.demand_capacity_range = demand_capacity_range
        self.num_own_generators = min(num_own_generators, num_generators)

        self.dataframe = self._generate_random_data()

    def _generate_random_data(self) -> pd.DataFrame:
        """Generate random market data matching the ElectricityMarketCurvesDataframe format."""

        rows = []

        # Generate owned generators
        for gen_idx in range(self.num_own_generators):
            entity_name = f"OwnGen_{gen_idx + 1}"
            for hour in range(self.num_hours):
                offer = np.random.uniform(*self.generator_price_range)
                limit = np.random.uniform(*self.generator_capacity_range)
                rows.append(
                    {
                        "is_generator": True,
                        "own": True,
                        "entity": entity_name,
                        "years": 1,
                        "hour": hour,
                        "offer": offer,
                        "limit": limit,
                    }
                )

        # Generate external generators
        for gen_idx in range(self.num_generators - self.num_own_generators):
            entity_name = f"ExtGen_{gen_idx + 1}"
            for hour in range(self.num_hours):
                offer = np.random.uniform(*self.generator_price_range)
                limit = np.random.uniform(*self.generator_capacity_range)
                rows.append(
                    {
                        "is_generator": True,
                        "own": False,
                        "entity": entity_name,
                        "years": 1,
                        "hour": hour,
                        "offer": offer,
                        "limit": limit,
                    }
                )

        # Generate demands
        for dem_idx in range(self.num_demands):
            entity_name = f"Demand_{dem_idx + 1}"
            for hour in range(self.num_hours):
                offer = np.random.uniform(*self.demand_price_range)
                limit = np.random.uniform(*self.demand_capacity_range)
                rows.append(
                    {
                        "is_generator": False,
                        "own": False,
                        "entity": entity_name,
                        "years": 1,
                        "hour": hour,
                        "offer": offer,
                        "limit": limit,
                    }
                )

        df = pd.DataFrame(rows)

        # Ensure correct column order
        columns = ["is_generator", "own", "entity", "years", "hour", "offer", "limit"]
        return df[columns]

    def save_dataframe(self, path: str, decimals: int | None = None):
        """Save the generated dataframe to CSV file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = self.dataframe.copy()
        if decimals is not None:
            df = df.round(decimals)
        df.to_csv(path, index=False)
        print(f"Saved random estimation to: {path}")


def main(
    num_hours: int = 8760,
    num_generators: int = 5,
    num_demands: int = 3,
    generator_price_range: tuple = (0, 100),
    demand_price_range: tuple = (20, 150),
    generator_capacity_range: tuple = (10, 100),
    demand_capacity_range: tuple = (50, 200),
    num_own_generators: int = 1,
    random_seed: int | None = None,
    update: bool = True,
):
    """
    Generate and save random estimation data.

    Args:
        num_hours: Number of hours to generate
        num_generators: Number of generators (can be a list for batch generation)
        num_demands: Number of demands
        generator_price_range: Price range for generators
        demand_price_range: Price range for demands
        generator_capacity_range: Capacity range for generators
        demand_capacity_range: Capacity range for demands
        num_own_generators: Number of owned generators
        random_seed: Random seed for reproducibility (can be a list for batch generation)
        update: If False, skip if file already exists
    """
    # Handle list-type parameters (iterate over each configuration)
    if isinstance(num_generators, (list, tuple)):
        for num_gen in num_generators:
            main(
                num_hours=num_hours,
                num_generators=num_gen,
                num_demands=num_demands,
                generator_price_range=generator_price_range,
                demand_price_range=demand_price_range,
                generator_capacity_range=generator_capacity_range,
                demand_capacity_range=demand_capacity_range,
                num_own_generators=num_own_generators,
                random_seed=random_seed,
                update=update,
            )
        return

    if isinstance(random_seed, (list, tuple)):
        for seed in random_seed:
            main(
                num_hours=num_hours,
                num_generators=num_generators,
                num_demands=num_demands,
                generator_price_range=generator_price_range,
                demand_price_range=demand_price_range,
                generator_capacity_range=generator_capacity_range,
                demand_capacity_range=demand_capacity_range,
                num_own_generators=num_own_generators,
                random_seed=seed,
                update=update,
            )
        return

    # Generate folder name based on parameters
    seed_suffix = f"_seed{random_seed}" if random_seed is not None else ""
    estimation_folder_name = f"random_gen{num_generators}_dem{num_demands}{seed_suffix}"

    # Create output path: data/estimations/{estimation_folder_name}.csv
    final_path = os.path.join(
        PathConfiguration.ESTIMATIONS_DIR_PATH, f"{estimation_folder_name}.csv"
    )

    # Check if file already exists and update flag is False
    if not update and os.path.exists(final_path):
        print(f"Skipping (already exists): {final_path}")
        return

    print(f"Generating random estimation: {estimation_folder_name}.csv")
    print(f"  - Hours: {num_hours}")
    print(f"  - Generators: {num_generators} ({num_own_generators} owned)")
    print(f"  - Demands: {num_demands}")
    print(f"  - Generator prices: {generator_price_range} €/MWh")
    print(f"  - Demand prices: {demand_price_range} €/MWh")
    print(f"  - Generator capacity: {generator_capacity_range} MWh")
    print(f"  - Demand capacity: {demand_capacity_range} MWh")
    if random_seed is not None:
        print(f"  - Random seed: {random_seed}")

    generator = RandomDataGenerator(
        num_hours=num_hours,
        num_generators=num_generators,
        num_demands=num_demands,
        generator_price_range=generator_price_range,
        demand_price_range=demand_price_range,
        generator_capacity_range=generator_capacity_range,
        demand_capacity_range=demand_capacity_range,
        num_own_generators=num_own_generators,
        random_seed=random_seed,
    )
    generator.save_dataframe(final_path, decimals=2)


if __name__ == "__main__":
    main()
