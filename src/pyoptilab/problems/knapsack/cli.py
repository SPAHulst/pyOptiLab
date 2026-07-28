import typer
from rich import print

from pyoptilab.problems.knapsack.generators import KnapsackGenerator
from pyoptilab.typer_config import command, create_app

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


@command(generate_app, "uniform")
def uniform(
    items: int = typer.Option(default=..., min=1, help="Number of items to generate"),
    capacity: int | None = typer.Option(
        default=None, min=1, help="Capacity of knapsack"
    ),
    capacity_ratio: float | None = typer.Option(
        default=None,
        min=0,
        max=1,
        help="Capacity of knapsack as ratio of total weight of items",
    ),
    seed: int = typer.Option(default=..., min=0, help="Seed to initialize PRNG"),
    weight_min: int = typer.Option(default=1, min=1, help="Minimum weight of items"),
    weight_max: int = typer.Option(default=100, min=1, help="Maximum weight of items"),
    value_min: int = typer.Option(default=1, min=0, help="Minimum value of items"),
    value_max: int = typer.Option(default=100, min=0, help="Maximum value of items"),
    instances: int = typer.Option(
        default=1, min=1, help="Number of instances to generate"
    ),
) -> None:
    """
    Generate an instance of the Knapsack Problem with weights and values drawn from uniform distribution
    """

    validate_capacity(capacity, capacity_ratio)
    validate_range(weight_min, weight_max, name="weight")
    validate_range(value_min, value_max, name="value")

    for i in range(instances):
        instance = KnapsackGenerator.uniform(
            num_items=items,
            capacity=capacity,
            capacity_ratio=capacity_ratio,
            seed=seed,
            weight_range=(weight_min, weight_max),
            value_range=(value_min, value_max),
        )

        print(f"{i}-th instance: {instance}")
