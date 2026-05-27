import json
import time

import pyomo.environ as pe
import pyomo.opt as po
import os

from ..data import ElectricityMarketCurvesDataframe, ElectricityMarketSolvedDataframe
from ..configuration import PathConfiguration


class MarketClearingProblem:
    def __init__(self, path: str):
        # Load base dataframe
        self.df = ElectricityMarketCurvesDataframe(path)

        self.solved_df = None
        self.results_info = {}
        self.model = self.get_model()
        self.solver = po.SolverFactory("gurobi")

    def get_model(self):
        model = pe.ConcreteModel()

        # Set dual
        model.dual = pe.Suffix(direction=pe.Suffix.IMPORT)

        # Sets
        model.generators = pe.Set(initialize=list(self.df.get_entities_name(True)))  # j
        model.consumers = pe.Set(initialize=list(self.df.get_entities_name(False)))  # l
        model.years = pe.Set(initialize=list(self.df.get_years()))  # y
        model.hours = pe.Set(initialize=list(self.df.get_hours()))  # t

        # Parameters
        model.generator_marginal_cost = pe.Param(
            model.generators,
            model.hours,
            default=1_000_000,
            initialize=self.df.get_entities_column_dict("offer", True),
        )
        model.generator_maximum_capacity = pe.Param(
            model.generators,
            model.hours,
            default=0,
            initialize=self.df.get_entities_column_dict("limit", True),
        )
        model.demand_marginal_utility = pe.Param(
            model.consumers,
            model.hours,
            default=0,
            initialize=self.df.get_entities_column_dict("offer", False),
        )
        model.demand_maximum = pe.Param(
            model.consumers,
            model.hours,
            default=0,
            initialize=self.df.get_entities_column_dict("limit", False),
        )

        # Variables
        model.production = pe.Var(
            model.generators, model.hours, within=pe.NonNegativeReals
        )
        model.consumption = pe.Var(
            model.consumers, model.hours, within=pe.NonNegativeReals
        )

        # Objective function
        def obj_rule(model):
            return sum(
                sum(
                    model.demand_marginal_utility[l, t] * model.consumption[l, t]
                    for l in model.consumers
                )
                - sum(
                    model.generator_marginal_cost[j, t] * model.production[j, t]
                    for j in model.generators
                )
                for t in model.hours
            )

        model.cost = pe.Objective(rule=obj_rule, sense=pe.maximize)

        # Constraints
        def power_balance_rule(model, t):
            return (
                sum(model.consumption[l, t] for l in model.consumers)
                - sum(model.production[j, t] for j in model.generators)
                == 0
            )

        model.constraint_power_balance = pe.Constraint(
            model.hours, rule=power_balance_rule
        )

        def consumption_limit_rule(model, l, t):
            return model.consumption[l, t] <= model.demand_maximum[l, t]

        model.constraint_consumption_limit = pe.Constraint(
            model.consumers, model.hours, rule=consumption_limit_rule
        )

        def production_limit_rule(model, j, t):
            return model.production[j, t] <= model.generator_maximum_capacity[j, t]

        model.constraint_production_limit = pe.Constraint(
            model.generators, model.hours, rule=production_limit_rule
        )

        num_constraints = len(
            list(model.component_data_objects(pe.Constraint, active=True))
        )
        num_variables = len(list(model.component_data_objects(pe.Var)))

        return model

    def solve(self):
        start_time = time.perf_counter()
        res = self.solver.solve(self.model, tee=False)
        end_time = time.perf_counter()

        self.results_info = {
            "execution_time_seconds": end_time - start_time,
            "solver_status": str(res.solver.status),
            "solver_termination_condition": str(res.solver.termination_condition),
        }

        # Get market prices (dual of power balance constraint)
        market_price = {
            t: self.model.dual[self.model.constraint_power_balance[t]]
            for t in self.model.hours
        }

        # Get taken for all entities (generators: production, consumers: consumption)
        taken = {}
        # Generators
        for j in self.model.generators:
            for t in self.model.hours:
                taken[(j, t)] = pe.value(self.model.production[j, t])
        # Consumers
        for l in self.model.consumers:
            for t in self.model.hours:
                taken[(l, t)] = pe.value(self.model.consumption[l, t])

        # Store raw results so caller can re-create solved dataframe with different ppa percentages
        self.taken = taken
        self.market_price = market_price

        # Default solved dataframe without PPA reduction applied
        solved_df = ElectricityMarketSolvedDataframe(
            self.df, taken=taken, prices=market_price
        )
        self.solved_df = solved_df

    def save_results_info(self, path: str):
        if not self.results_info:
            print("Unable to save results info: Info not set.")
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.results_info, f, indent=4)


def _check_market_files_exist(solved_path: str, info_path: str) -> bool:
    """Check if the market output files already exist."""
    return os.path.exists(solved_path) and os.path.exists(info_path)


def main(estimation_name: str, update: bool = True):
    # Single dataset - do the actual computation
    path = os.path.join(
        PathConfiguration.ESTIMATIONS_DIR_PATH, f"{estimation_name}.csv"
    )

    # LP baseline solved directory
    solved_dir, solved_path, info_path = PathConfiguration.market_lp_paths(
        estimation_name
    )

    # Check if files already exist and update flag is False
    if not update and _check_market_files_exist(solved_path, info_path):
        print(f"Skipping (already exists): lp baseline for {estimation_name}")
        return

    print(f"Computing LP baseline for {estimation_name}")

    # Build and solve the market without any PPA transformation.
    p = MarketClearingProblem(path)
    p.solve()

    # Solved dataframe reflects the original dataset.
    solved_df = ElectricityMarketSolvedDataframe(
        p.df, taken=p.taken, prices=p.market_price
    )

    # Save solved dataframe
    os.makedirs(solved_dir, exist_ok=True)
    solved_df.save_dataframe(solved_path)
    p.save_results_info(info_path)


if __name__ == "__main__":
    main()
