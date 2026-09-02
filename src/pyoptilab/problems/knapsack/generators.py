import dataclasses
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from rich import print


class ExperimentFileError(Exception):
    """The experiment JSON file is missing, unreadable or malformed such that makes it impossible to proceed"""


class InstanceParsingError(Exception):
    """An instance dict could not be converted to a KnapsackInstance"""


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
    def export_to_json(instances: list[KnapsackInstance], tag: str | None) -> Path:
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
            "instances": instance_dicts,
        }

        tag_part = (
            f"{tag}"
            if tag is not None
            else f"{datetime.now().astimezone():%Y%m%d_%H%M%S}"
        )
        file_name = f"instances_{tag_part}.json"
        path = Path(__file__).parent / "instances" / file_name

        path.write_text(json.dumps(experiment, indent=2))
        return path

    @classmethod
    def from_json(cls, instances_path: Path) -> list[KnapsackInstance]:
        experiment_dict = cls._load_experiment_dict(instances_path)

        try:
            instance_dicts = experiment_dict["instances"]
        except KeyError as e:
            raise ExperimentFileError(
                f"Key 'instances' is missing from {instances_path}: {e}"
            ) from e

        instances: list[KnapsackInstance] = []
        for index, instance_dict in enumerate(instance_dicts):
            try:
                instances.append(cls._instance_from_dict(instance_dict))
            except InstanceParsingError as e:
                print(f"Skipping instance {index} from {instances_path} due to {e}")

        return instances

    @staticmethod
    def _load_experiment_dict(instances_path: Path) -> dict[str, Any]:
        try:
            with open(instances_path, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            raise ExperimentFileError(f"No such file: {instances_path.name}")
        except json.JSONDecodeError as e:
            raise ExperimentFileError(
                f"Failed to decode JSON file: {instances_path.name}, due to {e}"
            )

    @classmethod
    def _instance_from_dict(cls, instance_dict: dict) -> KnapsackInstance:
        try:
            capacity = int(instance_dict["capacity"])
            weights = np.asarray(instance_dict["weights"], dtype=np.int32)
            values = np.asarray(instance_dict["values"], dtype=np.int32)
        except KeyError as e:
            raise InstanceParsingError(f"Missing key {e}") from e
        except (ValueError, TypeError) as e:
            raise InstanceParsingError(f"Invalid value: {e}") from e

        return cls(capacity=capacity, weights=weights, values=values)


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
