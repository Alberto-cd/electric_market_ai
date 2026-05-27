import os

# Configuration for running the optimization solvers


class PathConfiguration:
    DATA_DIR_PATH = "data"
    BASE_ESTIMATIONS_DIR_PATH = os.path.join(DATA_DIR_PATH, "base_estimations")
    PROFILES_DIR_PATH = os.path.join(DATA_DIR_PATH, "profiles")
    ESTIMATIONS_DIR_PATH = os.path.join(DATA_DIR_PATH, "estimations")
    SOLVED_ESTIMATIONS_DIR_PATH = os.path.join(DATA_DIR_PATH, "solved_estimations")
    IMAGES_DIR_PATH = "images"

    @classmethod
    def _solved_results_dir(cls, *parts: str) -> str:
        return os.path.join(cls.SOLVED_ESTIMATIONS_DIR_PATH, *parts)

    @classmethod
    def market_lp_dir(cls, estimation_name: str) -> str:
        return cls._solved_results_dir(estimation_name, "market", "lp")

    @classmethod
    def market_lp_paths(cls, estimation_name: str) -> tuple[str, str, str]:
        solved_dir = cls.market_lp_dir(estimation_name)
        return (
            solved_dir,
            os.path.join(solved_dir, "results.csv"),
            os.path.join(solved_dir, "results_info.json"),
        )

    @classmethod
    def market_ml_dir(cls, estimation_name: str, model_name: str) -> str:
        return cls._solved_results_dir(estimation_name, "market", "ml", model_name)

    @classmethod
    def market_ml_paths(
        cls, estimation_name: str, model_name: str
    ) -> tuple[str, str, str]:
        solved_dir = cls.market_ml_dir(estimation_name, model_name)
        return (
            solved_dir,
            os.path.join(solved_dir, "results.csv"),
            os.path.join(solved_dir, "results_info.json"),
        )

    @classmethod
    def bilevel_blp_dir(
        cls, estimation_name: str, ppa_price: float | int | str | None
    ) -> str:
        ppa_folder = "default" if ppa_price is None else str(ppa_price)
        return cls._solved_results_dir(estimation_name, "bilevel", "blp", ppa_folder)

    @classmethod
    def bilevel_blp_paths(
        cls, estimation_name: str, ppa_price: float | int | str | None
    ) -> tuple[str, str, str]:
        solved_dir = cls.bilevel_blp_dir(estimation_name, ppa_price)
        return (
            solved_dir,
            os.path.join(solved_dir, "results.csv"),
            os.path.join(solved_dir, "results_info.json"),
        )

    @classmethod
    def bilevel_ml_dir(cls, estimation_name: str, model_name: str) -> str:
        return cls._solved_results_dir(estimation_name, "bilevel", "ml", model_name)

    @classmethod
    def bilevel_ml_paths(
        cls, estimation_name: str, model_name: str
    ) -> tuple[str, str, str]:
        solved_dir = cls.bilevel_ml_dir(estimation_name, model_name)
        return (
            solved_dir,
            os.path.join(solved_dir, "results.csv"),
            os.path.join(solved_dir, "results_info.json"),
        )

    @classmethod
    def create_directories(cls):
        for attr in cls.__dict__:
            if attr.endswith("_DIR_PATH"):
                os.makedirs(getattr(cls, attr), exist_ok=True)


PathConfiguration.create_directories()


class ExecutionConfiguration:
    M_MARGIN_MULTIPLIER = 1.2
    SOLVER_TIMEOUT_SECONDS = 6000  # 100 minutes
    # Default grid of PPA prices (currency units per MWh) used for bilevel experiments
    DEFAULT_PPA_PRICES = [i for i in range(0, 31, 2)]
