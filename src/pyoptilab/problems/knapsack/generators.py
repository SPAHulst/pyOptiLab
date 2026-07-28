from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class KnapsackInstance:
    capacity: int
    weights: NDArray[np.int32]
    values: NDArray[np.int32]

    @property
    def num_items(self) -> int:
        return self.weights.size

    def __post_init__(self):
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")

        if self.weights.shape != self.values.shape:
            raise ValueError("weights and values must have identical shape")

        if np.any(self.weights <= 0):
            raise ValueError("weights must be positive")

        if np.any(self.values < 0):
            raise ValueError("values must be non-negative")

        self.weights.flags.writeable = False
        self.values.flags.writeable = False


class KnapsackGenerator:
    @staticmethod
    def uniform(
        *,
        num_items: int,
        seed: int,
        capacity: int | None = None,
        capacity_ratio: float | None = None,
        weight_range: tuple[int, int],
        value_range: tuple[int, int],
    ) -> KnapsackInstance:

        rng = np.random.default_rng(seed)

        weights = rng.integers(
            weight_range[0],
            weight_range[1],
            size=num_items,
            dtype=np.int32,
            endpoint=True,
        )

        values = rng.integers(
            value_range[0],
            value_range[1],
            size=num_items,
            dtype=np.int32,
            endpoint=True,
        )

        if capacity is None:
            capacity = int(np.sum(weights) * capacity_ratio)

        return KnapsackInstance(capacity=capacity, weights=weights, values=values)
