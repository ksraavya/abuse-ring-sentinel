import numpy as np

from models.baseline_a import CostConfig, choose_economic_threshold, threshold_grid


def test_threshold_grid_is_0_01_to_0_99():
    grid = threshold_grid()
    assert len(grid) == 99
    assert grid[0] == 0.01
    assert grid[-1] == 0.99


def test_threshold_selection_uses_economic_cost():
    y_true = np.array([0, 0, 0, 1])
    probabilities = np.array([0.1, 0.2, 0.6, 0.7])

    threshold, best, all_results = choose_economic_threshold(
        y_true,
        probabilities,
        CostConfig(500, 5000),
    )

    assert threshold == 0.61
    assert best["economic_cost"] == 0.0
    assert len(all_results) == 99
