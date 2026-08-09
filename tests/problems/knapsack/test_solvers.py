import numpy as np
import pytest

from pyoptilab.problems.knapsack.generators import KnapsackInstance
from pyoptilab.problems.knapsack.solvers.exact import ILP_solver

EXACT_SOLVERS = [
    pytest.param(ILP_solver, id="ILP"),
]


@pytest.mark.parametrize("solve_exact", EXACT_SOLVERS)
def test_exact_solver_returns_optimal_solution(solve_exact):
    instance = KnapsackInstance(
        capacity=5,
        weights=np.array([2, 3, 4], dtype=np.int32),
        values=np.array([3, 4, 5], dtype=np.int32),
    )

    solution = solve_exact(instance)

    assert solution.value == 7
    assert solution.selected_items == (0, 1)
