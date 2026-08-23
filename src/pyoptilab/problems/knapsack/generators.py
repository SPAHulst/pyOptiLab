import dataclasses
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

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

    def __eq__(self, other) -> bool:
        if not isinstance(other, KnapsackInstance):
            return NotImplemented

        return (
            (self.capacity == other.capacity)
            and (np.array_equal(self.weights, other.weights))
            and (np.array_equal(self.values, other.values))
        )

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")

        if self.weights.ndim != 1 or self.values.ndim != 1:
            raise ValueError("weights and values must be 1-D arrays")

        if self.weights.shape != self.values.shape:
            raise ValueError("weights and values must have identical shape")

        if np.any(self.weights <= 0):
            raise ValueError("weights must be positive")

        if np.any(self.values < 0):
            raise ValueError("values must be non-negative")

        self.weights.flags.writeable = False
        self.values.flags.writeable = False

    @staticmethod
    def export_to_json(
        instances: list[KnapsackInstance], tag: str | None = None
    ) -> Path:
        instance_dicts = []

        for i, instance in enumerate(instances):
            dict = dataclasses.asdict(instance)
            for key, value in dict.items():
                if key != "capacity":
                    dict[key] = value.tolist()

            dict["id"] = i
            instance_dicts.append(dict)

        experiment = {
            "generated_at": datetime.now().astimezone().isoformat(),
            "num_instances": len(instance_dicts),
            "instances": instance_dicts,
        }

        tag_part = f"{tag}_" if tag else ""
        file_name = (
            f"instances_{tag_part}{datetime.now().astimezone():%Y%m%d_%H%M%S}.json"
        )
        path = Path(__file__).parent / "instances" / file_name

        path.write_text(json.dumps(experiment, indent=2))
        return path


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

        if capacity is None and capacity_ratio is not None:
            capacity = int(np.sum(weights) * capacity_ratio)
        assert capacity is not None

        return KnapsackInstance(capacity=capacity, weights=weights, values=values)
