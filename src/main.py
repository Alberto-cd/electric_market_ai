import os
import re

from .data import main as data_main
from .configuration import PathConfiguration
from .random_data import main as random_data_main
from .market.solver import main as market_solver_main
from .market.linear_regression import main as linear_regression_main
from .market.decision_tree import main as decision_tree_main
from .market.xgboost import main as xgboost_main
from .market.bilstm import main as bilstm_trainer_main
from .market.visualizer import main as market_visualizer_main
from .bilevel.solver import main as bilevel_solver_main
from .bilevel.visualizer import main as bilevel_visualizer_main
from .bilevel.bilstm import main as bilevel_bilstm_main


def _print_title(title: str):
    """Print a formatted title."""
    print(f"\n{'='*50}")
    print(title)
    print(f"{'='*50}")


def _print_subtitle(subtitle: str):
    """Print a formatted subtitle."""
    print(f"\n{'-'*50}")
    print(subtitle)
    print(f"{'-'*50}")


def _profile_to_estimation_name(profile_filename: str) -> str:
    match = re.search(r"esios_profiles_processed(\d{4})\.csv$", profile_filename)
    if not match:
        raise ValueError(f"Unsupported profile filename: {profile_filename}")
    return f"estimations_{match.group(1)}"


def _execute_profile_estimations():
    """Generate profile-based estimations using the year encoded in each profile filename."""
    _print_title("Profile Estimations")

    profile_files = sorted(
        filename
        for filename in os.listdir(PathConfiguration.PROFILES_DIR_PATH)
        if re.match(r"esios_profiles_processed\d{4}\.csv$", filename)
    )

    for profile_file in profile_files:
        estimation_name = _profile_to_estimation_name(profile_file)
        _print_subtitle(f"Executing: data ({estimation_name})")
        data_main(
            base_estimations="market_data.json",
            profile=profile_file,
            estimation_name=estimation_name,
            solar_percent=0.2,
            update=False,
        )


def _execute_random_estimations():
    """Generate random estimation sets for LP comparison and later BiLSTM training."""
    _print_title("Random Estimations")

    _print_subtitle("Generating: Random estimation (5 generators, 3 demands, seed=42)")
    random_data_main(
        num_hours=8760,
        num_generators=5,
        num_demands=3,
        num_own_generators=1,
        generator_price_range=(10, 80),
        demand_price_range=(30, 150),
        generator_capacity_range=(10, 100),
        demand_capacity_range=(50, 200),
        random_seed=42,
        update=False,
    )

    _print_subtitle("Generating: Random estimation with multiple seeds")
    random_data_main(
        num_hours=8760,
        num_generators=8,
        num_demands=4,
        num_own_generators=2,
        generator_price_range=(5, 100),
        demand_price_range=(20, 180),
        generator_capacity_range=(20, 150),
        demand_capacity_range=(80, 300),
        random_seed=[123, 456, 789],
        update=False,
    )

    _print_subtitle("Generating: Random estimation with varying generator counts")
    random_data_main(
        num_hours=8760,
        num_generators=[3, 5, 10, 15],
        num_demands=3,
        num_own_generators=1,
        generator_price_range=(15, 90),
        demand_price_range=(25, 160),
        generator_capacity_range=(15, 120),
        demand_capacity_range=(60, 250),
        random_seed=99,
        update=False,
    )


def _execute_market_lp():
    """Solve the LP market model and generate baseline plots for generated estimations."""
    _print_title("Market Model Tests")

    estimation_files = sorted(
        filename
        for filename in os.listdir(PathConfiguration.ESTIMATIONS_DIR_PATH)
        if filename.endswith(".csv")
        and (filename.startswith("estimations_") or filename.startswith("random_"))
    )

    for estimation_file in estimation_files:
        estimation_name = os.path.splitext(estimation_file)[0]
        _print_subtitle(f"Executing: market.solver ({estimation_name})")
        market_solver_main(estimation_name=estimation_name, update=False)

        _print_subtitle(f"Executing: market.visualizer ({estimation_name})")
        market_visualizer_main(estimation_name=estimation_name, update=False)


def _execute_ml_baselines():
    """Train the pooled sklearn baselines once across all profile estimations."""
    _print_title("ML Baselines")

    _print_subtitle("Executing: linear_regression (pooled)")
    linear_regression_main(update=False)

    _print_subtitle("Executing: decision_tree (pooled)")
    decision_tree_main(update=False)

    _print_subtitle("Executing: xgboost (pooled)")
    xgboost_main(update=False)


def _execute_bilstm_training():
    """Train the pooled BiLSTM surrogate over all solved estimation datasets."""
    _print_title("BiLSTM Training")

    _print_subtitle("Executing: bilstm_trainer (pooled)")
    bilstm_trainer_main(update=False)


def _execute_bilevel_solver():
    """Run the bilevel LP solver (strategic offering) for all estimation CSVs."""
    _print_title("Bilevel LP Solver")

    estimation_files = sorted(
        filename
        for filename in os.listdir(PathConfiguration.ESTIMATIONS_DIR_PATH)
        if filename.endswith(".csv")
        and (filename.startswith("estimations_") or filename.startswith("random_"))
    )

    for estimation_file in estimation_files:
        estimation_name = os.path.splitext(estimation_file)[0]
        _print_subtitle(f"Executing: bilevel.solver (LP bilevel) ({estimation_name})")
        # Default call: no explicit ppa_price (strategic offering problem)
        bilevel_solver_main(estimation_name=estimation_name, update=False)


def _execute_bilevel_visualizer():
    """Generate bilevel plots and comparison summaries."""
    _print_title("Bilevel Visualizer")
    bilevel_visualizer_main(update=False)


def _execute_bilevel_bilstm():
    """Run the bilevel BiLSTM surrogate for all estimation CSVs."""
    _print_title("Bilevel BiLSTM Surrogate")

    bilstm_model_path = os.path.join(
        "models",
        "bilstm",
        "results",
        "bs64_h64_l2_do0.10_lr1e-02_st1.50_ent1e-03_rs42",
        "model.pt",
    )

    if not os.path.exists(bilstm_model_path):
        fallback_results_dir = os.path.join("models", "bilstm", "results")
        latest_model_path = None
        latest_mtime = -1.0

        if os.path.isdir(fallback_results_dir):
            for root, _, files in os.walk(fallback_results_dir):
                if "model.pt" not in files:
                    continue
                candidate_path = os.path.join(root, "model.pt")
                candidate_mtime = os.path.getmtime(candidate_path)
                if candidate_mtime > latest_mtime:
                    latest_mtime = candidate_mtime
                    latest_model_path = candidate_path

        if latest_model_path is None:
            _print_subtitle(
                f"Skipping bilevel surrogate: model not found at {bilstm_model_path}"
            )
            return

        bilstm_model_path = latest_model_path
        _print_subtitle(f"Using latest BiLSTM checkpoint at {bilstm_model_path}")

    if not os.path.exists(bilstm_model_path):
        _print_subtitle(
            f"Skipping bilevel surrogate: model not found at {bilstm_model_path}"
        )
        return

    estimation_files = sorted(
        filename
        for filename in os.listdir(PathConfiguration.ESTIMATIONS_DIR_PATH)
        if filename.endswith(".csv")
        and (filename.startswith("estimations_") or filename.startswith("random_"))
    )

    for estimation_file in estimation_files:
        estimation_name = os.path.splitext(estimation_file)[0]
        _print_subtitle(
            f"Executing: bilevel.bilstm (surrogate bilevel) ({estimation_name})"
        )
        bilevel_bilstm_main(
            estimation_name=estimation_name,
            model_path=bilstm_model_path,
            update=False,
        )


def main():
    """Execute all solvers with explicit function calls."""

    try:
        # # --- DATA ---
        # _execute_profile_estimations()
        # _execute_random_estimations()

        # # --- MARKET ---
        # # LP
        # _execute_market_lp()

        # # ML Baselines
        # _execute_ml_baselines()

        # # BiLSTM
        # _execute_bilstm_training()

        # # Visualizer
        # _print_subtitle("Executing: market.visualizer (models comparison)")
        # market_visualizer_main(update=False)

        # # --- BILEVEL ---
        # # MILP
        # _execute_bilevel_solver()

        # # BiLSTM
        # _execute_bilevel_bilstm()

        # Visualizer
        _execute_bilevel_visualizer()

        print("\n" + "=" * 50)
        print("EXECUTION COMPLETED SUCCESSFULLY")
        print("=" * 50)

    except KeyboardInterrupt:
        print("\nExecution cancelled by user. Exiting...")
        return
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        raise


if __name__ == "__main__":
    main()
