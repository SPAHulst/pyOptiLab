from pyoptilab.problems.knapsack.cli import app as knapsack_app
from pyoptilab.typer_config import create_app

app = create_app()

app.add_typer(knapsack_app, name="knapsack")

if __name__ == "__main__":
    app()
