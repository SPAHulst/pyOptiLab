from collections.abc import Callable

import typer


def create_app(**kwargs) -> typer.Typer:
    """
    Set no_args_is_help=True for each command group
    """
    return typer.Typer(
        no_args_is_help=True,
        **kwargs,
    )


def command(app: typer.Typer, *args, **kwargs) -> Callable:
    """
    Set no_args_is_help=True for each command
    """
    kwargs.setdefault("no_args_is_help", True)
    return app.command(*args, **kwargs)
