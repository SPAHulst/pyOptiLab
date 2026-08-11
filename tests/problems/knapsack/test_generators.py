import numpy as np
import pytest

from pyoptilab.problems.knapsack.generators import KnapsackGenerator, KnapsackInstance

GENERATORS = [
    pytest.param(
        lambda seed: KnapsackGenerator.uniform(
            num_items=100,
            capacity=250,
            capacity_ratio=None,
            weight_range=(1, 100),
            value_range=(1, 100),
            seed=seed,
        ),
        id="uniform_capacity",
    ),
    pytest.param(
        lambda seed: KnapsackGenerator.uniform(
            num_items=100,
            capacity=None,
            capacity_ratio=0.25,
            weight_range=(1, 100),
            value_range=(1, 100),
            seed=seed,
        ),
        id="uniform_capacity_ratio",
    ),
]


@pytest.mark.parametrize(
    ("capacity", "weights", "values", "error"),
    [
        (
            0,
            np.array([1, 2], dtype=np.int32),
            np.array([10, 20], dtype=np.int32),
            "capacity must be positive",
        ),
        (
            10,
            np.array([(1, 2), (3, 4)], dtype=np.int32),
            np.array([(10, 20), (30, 40)], dtype=np.int32),
            "weights and values must be 1-D arrays",
        ),
        (
            10,
            np.array([1, 2], dtype=np.int32),
            np.array([10], dtype=np.int32),
            "weights and values must have identical shape",
        ),
        (
            10,
            np.array([1, 0, 2], dtype=np.int32),
            np.array([10, 20, 30], dtype=np.int32),
            "weights must be positive",
        ),
        (
            10,
            np.array([1, 2, 3], dtype=np.int32),
            np.array([10, -5, 20], dtype=np.int32),
            "values must be non-negative",
        ),
    ],
)
def test_invalid_instances_raise_value_error(capacity, weights, values, error):
    with pytest.raises(ValueError, match=error):
        KnapsackInstance(capacity=capacity, weights=weights, values=values)


def test_valid_instance():
    instance = KnapsackInstance(
        capacity=10,
        weights=np.array([1, 2, 3], dtype=np.int32),
        values=np.array([10, 20, 30], dtype=np.int32),
    )

    assert instance.capacity == 10
    assert instance.num_items == 3
    assert np.array_equal(instance.weights, [1, 2, 3])
    assert np.array_equal(instance.values, [10, 20, 30])
    assert not instance.weights.flags.writeable
    assert not instance.values.flags.writeable


def test_knapsack_instance_equality():
    instance_a = KnapsackInstance(
        capacity=10,
        weights=np.array([1, 2, 3], dtype=np.int32),
        values=np.array([10, 20, 30], dtype=np.int32),
    )

    instance_b = KnapsackInstance(
        capacity=10,
        weights=np.array([1, 2, 3], dtype=np.int32),
        values=np.array([10, 20, 30], dtype=np.int32),
    )

    instance_c = KnapsackInstance(
        capacity=12,
        weights=np.array([1, 2, 3], dtype=np.int32),
        values=np.array([10, 20, 30], dtype=np.int32),
    )

    instance_d = KnapsackInstance(
        capacity=10,
        weights=np.array([1, 4, 3], dtype=np.int32),
        values=np.array([10, 20, 30], dtype=np.int32),
    )

    instance_e = KnapsackInstance(
        capacity=10,
        weights=np.array([1, 2, 3], dtype=np.int32),
        values=np.array([10, 21, 30], dtype=np.int32),
    )

    assert instance_a == instance_b
    assert instance_a != instance_c
    assert instance_a != instance_d
    assert instance_a != instance_e


@pytest.mark.parametrize("generator", GENERATORS)
def test_generator_returns_valid_instance(generator):
    instance = generator(seed=23)

    assert isinstance(instance, KnapsackInstance)


@pytest.mark.parametrize("generator", GENERATORS)
def test_generator_is_reproducible(generator):
    instance_a = generator(seed=23)
    instance_b = generator(seed=23)

    assert instance_a == instance_b


@pytest.mark.parametrize("generator", GENERATORS)
def test_different_seeds(generator):
    instance_a = generator(seed=23)
    instance_b = generator(seed=32)

    assert instance_a != instance_b
