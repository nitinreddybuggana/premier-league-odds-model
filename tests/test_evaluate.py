import numpy as np
import pytest

from plodds.evaluate import brier_score, calibration_table, log_loss

OUTCOMES = ["H", "D", "A", "H"]
UNIFORM = np.full((4, 3), 1 / 3)
PERFECT = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)


def test_log_loss_of_uniform_forecast_is_log_three():
    assert log_loss(UNIFORM, OUTCOMES) == pytest.approx(np.log(3))


def test_log_loss_by_hand():
    probs = np.array([[0.5, 0.3, 0.2], [0.2, 0.2, 0.6]])
    expected = -(np.log(0.5) + np.log(0.6)) / 2
    assert log_loss(probs, ["H", "A"]) == pytest.approx(expected)


def test_log_loss_rewards_better_forecasts():
    sharp = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7], [0.7, 0.2, 0.1]])
    assert log_loss(sharp, OUTCOMES) < log_loss(UNIFORM, OUTCOMES)


def test_brier_score_reference_values():
    assert brier_score(PERFECT, OUTCOMES) == pytest.approx(0.0)
    assert brier_score(UNIFORM, OUTCOMES) == pytest.approx(2 / 3)


def test_brier_score_by_hand():
    probs = np.array([[0.5, 0.3, 0.2]])
    assert brier_score(probs, ["H"]) == pytest.approx(0.25 + 0.09 + 0.04)


def test_calibration_table():
    # 100 matches, every forecast is (0.65, 0.25, 0.10), and outcomes occur
    # at exactly those rates: a perfectly calibrated forecaster.
    probs = np.tile([0.65, 0.25, 0.10], (100, 1))
    outcomes = ["H"] * 65 + ["D"] * 25 + ["A"] * 10
    table = calibration_table(probs, outcomes, bins=10)

    assert {"mean_predicted", "observed", "count"} <= set(table.columns)
    assert len(table) == 3  # only three of the ten bins have anything in them
    assert table["count"].sum() == 300
    assert table["observed"].tolist() == pytest.approx(table["mean_predicted"].tolist())
