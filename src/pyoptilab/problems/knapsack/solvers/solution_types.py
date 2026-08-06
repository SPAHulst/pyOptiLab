from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KnapsackSolution:
    value: int
    selected_items: tuple[int, ...]

    def __rich__(self) -> str:
        return f"Solution achieves a value of {self.value} with the following items: {self.selected_items}"
