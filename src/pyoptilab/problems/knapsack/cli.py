from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich import print

from pyoptilab.problems.knapsack.generators import (
    ExperimentFileError,
    KnapsackGenerator,
    KnapsackInstance,
)
from pyoptilab.problems.knapsack.solvers.exact import ILP_solver
from pyoptilab.problems.knapsack.solvers.solution_types import KnapsackSolution
from pyoptilab.typer_config import command, create_app


@dataclass(frozen=True, slots=True)
class KnapsackInstanceParameters:
    seed: int
    items: int
    capacity: int | None
    capacity_ratio: float | None
    weight_bounds: tuple[int, int]
    value_bounds: tuple[int, int]
    num_instances: int


class Solver(StrEnum):
    ILP = "ilp"

    @classmethod
    def list_values(cls) -> str:
        return ", ".join(solver for solver in cls)


SOLVERS: dict[Solver, Callable] = {Solver.ILP: ILP_solver}


app = create_app()
generate_app = create_app()

app.add_typer(generate_app, name="generate")


def validate_capacity(capacity: int | None, capacity_ratio: float | None) -> None:
    if capacity is None and capacity_ratio is None:
        raise typer.BadParameter("Specify either --capacity or --capacity-ratio")

    if capacity is not None and capacity_ratio is not None:
        raise typer.BadParameter("Specify only one of --capacity or --capacity-ratio")


def validate_range(min: int, max: int, *, name: str) -> None:
    if min > max:
        raise typer.BadParameter(
            f"--{name}-min must be less than or equal to --{name}-max"
        )


def get_seed_sequence(base_seed: int, size: int) -> list[int]:
    sequence_generator = np.random.SeedSequence(base_seed)
    seed_sequence = sequence_generator.generate_state(size)
    return list(seed_sequence)


def solve_instance(instance: KnapsackInstance, solver: Solver) -> KnapsackSolution:
    return SOLVERS[solver](instance)


def generate_instances(
    parameters: KnapsackInstanceParameters,
) -> list[KnapsackInstance]:
    validate_capacity(parameters.capacity, parameters.capacity_ratio)
    validate_range(
        parameters.weight_bounds[0], parameters.weight_bounds[1], name="weight"
    )
    validate_range(parameters.value_bounds[0], parameters.value_bounds[1], name="value")

    seeds = (
        get_seed_sequence(parameters.seed, parameters.num_instances)
        if parameters.num_instances != 1
        else [parameters.seed]
    )

    experiment = [
        KnapsackGenerator.uniform(
            num_items=parameters.items,
            capacity=parameters.capacity,
            capacity_ratio=parameters.capacity_ratio,
            seed=seed,
            weight_range=parameters.weight_bounds,
            value_range=parameters.value_bounds,
        )
        for seed in seeds
    ]

    return experiment


@command(generate_app, "uniform")
def uniform(
    base_seed: Annotated[int, typer.Option(min=0, help="Seed to initialize PRNG")],
    items: Annotated[int, typer.Option(min=1, help="Number of items to generate")],
    capacity: Annotated[
        int | None, typer.Option(min=1, help="Capacity of knapsack")
    ] = None,
    capacity_ratio: Annotated[
        float | None,
        typer.Option(
            min=0,
            max=1,
            help="Capacity of knapsack as ratio of total weight of items",
        ),
    ] = None,
    weight_min: Annotated[int, typer.Option(min=1, help="Minimum weight of items")] = 1,
    weight_max: Annotated[
        int, typer.Option(min=1, help="Maximum weight of items")
    ] = 100,
    value_min: Annotated[int, typer.Option(min=0, help="Minimum value of items")] = 1,
    value_max: Annotated[int, typer.Option(min=0, help="Maximum value of items")] = 100,
    instances: Annotated[
        int, typer.Option(min=1, help="Number of instances to generate")
    ] = 1,
    tag: Annotated[
        str | None,
        typer.Option(help="Tag for the JSON file of instances (instances_{tag}.json)"),
    ] = None,
) -> None:

    experiment = generate_instances(
        parameters=KnapsackInstanceParameters(
            seed=base_seed,
            items=items,
            capacity=capacity,
            capacity_ratio=capacity_ratio,
            weight_bounds=(weight_min, weight_max),
            value_bounds=(value_min, value_max),
            num_instances=instances,
        )
    )

    print(
        f"Generated and exported instances to {KnapsackInstance.export_to_json(experiment, tag)}"
    )


@command(app, "solve")
def solve(
    tags: Annotated[
        list[str],
        typer.Option(
            help="Tags to identify instance JSON files (instances_{tag}.json)"
        ),
    ],
    solvers: Annotated[
        list[Solver],
        typer.Option(case_sensitive=False, help="Solvers to use on the instances"),
    ],
) -> None:
    for tag in tags:
        path = Path(__file__).parent / "instances" / f"instances_{tag}.json"

        try:
            instances = KnapsackInstance.from_json(path)
        except ExperimentFileError as e:
            print(f"Skipping tag {tag} due to error loading {path}: {e}")
            continue

        for solver in solvers:
            for instance in instances:
                print(solve_instance(instance, solver))


@command(app, "run")
def run(
    solvers: Annotated[
        list[Solver],
        typer.Option(case_sensitive=False, help="Solvers to use on the instances"),
    ],
    base_seed: Annotated[int, typer.Option(min=0, help="Seed to initialize PRNG")],
    items: Annotated[int, typer.Option(min=1, help="Number of items to generate")],
    capacity: Annotated[
        int | None, typer.Option(min=1, help="Capacity of knapsack")
    ] = None,
    capacity_ratio: Annotated[
        float | None,
        typer.Option(
            min=0,
            max=1,
            help="Capacity of knapsack as ratio of total weight of items",
        ),
    ] = None,
    weight_min: Annotated[int, typer.Option(min=1, help="Minimum weight of items")] = 1,
    weight_max: Annotated[
        int, typer.Option(min=1, help="Maximum weight of items")
    ] = 100,
    value_min: Annotated[int, typer.Option(min=0, help="Minimum value of items")] = 1,
    value_max: Annotated[int, typer.Option(min=0, help="Maximum value of items")] = 100,
    instances: Annotated[
        int, typer.Option(min=1, help="Number of instances to generate")
    ] = 1,
) -> None:

    experiment = generate_instances(
        parameters=KnapsackInstanceParameters(
            seed=base_seed,
            items=items,
            capacity=capacity,
            capacity_ratio=capacity_ratio,
            weight_bounds=(weight_min, weight_max),
            value_bounds=(value_min, value_max),
            num_instances=instances,
        )
    )

    for instance in experiment:
        for solver in solvers:
            print(solve_instance(instance, solver))
