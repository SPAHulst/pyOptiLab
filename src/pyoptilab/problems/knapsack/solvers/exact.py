import pulp

from pyoptilab.problems.knapsack.generators import KnapsackInstance
from pyoptilab.problems.knapsack.solvers.solution_types import KnapsackSolution


def ILP_solver(instance: KnapsackInstance) -> KnapsackSolution:
    n = instance.num_items

    knapsack_lp = pulp.LpProblem("Exact_Knapsack_ILP", pulp.LpMaximize)
    solver = pulp.getSolver("COIN_CMD", msg=0)

    x = {
        j: knapsack_lp.add_variable(f"x_{j}", lowBound=0, upBound=1, cat="Integer")
        for j in range(n)
    }

    knapsack_lp += (
        pulp.lpSum(instance.weights[i] * x[i] for i in range(n)) <= instance.capacity,
        "capacity_constraint",
    )
    knapsack_lp += (
        pulp.lpSum(instance.values[i] * x[i] for i in range(n)),
        "objective",
    )

    knapsack_lp.solve(solver)
    if knapsack_lp.status != pulp.constants.LpStatusOptimal:
        raise RuntimeError("No feasible solution found")

    return KnapsackSolution(
        value=int(pulp.value(knapsack_lp.objective)),
        selected_items=tuple(j for j in range(n) if pulp.value(x[j]) == 1),
    )
