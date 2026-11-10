import numpy as np
import pandas as pd
import pytest

from plodds.backtest import (
    kelly_fraction,
    max_drawdown,
    simulate_bets,
    summarise_bets,
    walk_forward,
)


# ── walk_forward ──────────────────────────────────────────────────────────────

class SpyModel:
    """Stands in for the real model. Predicts 50/30/20 for every match and
    refuses teams it was not trained on, like the real one should."""

    fits = []

    def fit(self, matches):
        self.teams = set(matches["home_team"]) | set(matches["away_team"])
        SpyModel.fits.append(matches["date"].max())
        return self

    def outcome_probabilities(self, home_team, away_team):
        if home_team not in self.teams or away_team not in self.teams:
            raise KeyError(home_team if home_team not in self.teams else away_team)
        return np.array([0.5, 0.3, 0.2])


def fixture_list():
    """Two seasons of weekly rounds between four teams. 'Promoted' replaces
    'Relegated' in the second season."""
    rows = []

    def add_season(season, start, teams, rounds):
        pairs = [(teams[0], teams[1]), (teams[2], teams[3])]
        for week in range(rounds):
            date = pd.Timestamp(start) + pd.Timedelta(weeks=week)
            for home, away in pairs:
                rows.append((season, date, home, away, 1, 0, "H", 2.0, 3.5, 4.0))
            pairs = [(pairs[0][0], pairs[1][1]), (pairs[1][0], pairs[0][1])]

    add_season("2022-23", "2022-08-06", ["A", "B", "C", "Relegated"], rounds=6)
    add_season("2023-24", "2023-08-05", ["A", "B", "C", "Promoted"], rounds=6)
    matches = pd.DataFrame(
        rows,
        columns=["season", "date", "home_team", "away_team", "home_goals", "away_goals",
                 "result", "odds_home", "odds_draw", "odds_away"],
    )
    matches.insert(0, "match_id", range(1, len(matches) + 1))
    return matches


@pytest.fixture
def predictions():
    SpyModel.fits = []
    return walk_forward(fixture_list(), SpyModel, test_seasons=["2023-24"], refit_days=7)


def test_never_trains_on_the_match_being_predicted(predictions):
    assert (predictions["trained_through"] < predictions["date"]).all()


def test_only_predicts_the_test_seasons(predictions):
    assert set(predictions["season"]) == {"2023-24"}


def test_adds_probability_columns(predictions):
    assert predictions[["p_home", "p_draw", "p_away"]].values == pytest.approx(
        np.tile([0.5, 0.3, 0.2], (len(predictions), 1))
    )
    assert "odds_home" in predictions.columns


def test_promoted_team_is_skipped_until_it_has_history(predictions):
    # Week 1: the promoted side has never played, so its match is skipped.
    # From week 2 on, the weekly refit has seen it, so everything is predicted.
    assert len(predictions) == 11
    first_week = predictions[predictions["date"] == pd.Timestamp("2023-08-05")]
    assert len(first_week) == 1
    assert "Promoted" not in set(first_week["home_team"]) | set(first_week["away_team"])


def test_refits_on_schedule():
    matches = fixture_list()
    SpyModel.fits = []
    walk_forward(matches, SpyModel, test_seasons=["2023-24"], refit_days=7)
    weekly = len(SpyModel.fits)
    SpyModel.fits = []
    walk_forward(matches, SpyModel, test_seasons=["2023-24"], refit_days=14)
    fortnightly = len(SpyModel.fits)
    assert (weekly, fortnightly) == (6, 3)


# ── staking and metrics ───────────────────────────────────────────────────────

def test_kelly_fraction():
    assert kelly_fraction(0.6, 2.0) == pytest.approx(0.2)
    assert kelly_fraction(0.25, 5.0) == pytest.approx(0.0625)


def test_kelly_fraction_is_zero_without_an_edge():
    assert kelly_fraction(0.5, 2.0) == 0
    assert kelly_fraction(0.4, 2.0) == 0


def test_max_drawdown():
    assert max_drawdown([100, 120, 90, 130]) == pytest.approx(-0.25)
    assert max_drawdown([100, 110, 120]) == 0
    assert max_drawdown(pd.Series([100, 50, 75, 25])) == pytest.approx(-0.75)


def bet_predictions():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22"]),
            "home_team": ["A", "B", "C", "D"],
            "away_team": ["E", "F", "G", "H"],
            "result": ["H", "A", "H", "D"],
            "p_home": [0.60, 0.60, 0.50, 0.30],
            "p_draw": [0.25, 0.25, 0.30, 0.40],
            "p_away": [0.15, 0.15, 0.20, 0.30],
            "odds_home": [2.0, 2.0, 2.0, np.nan],
            "odds_draw": [3.5, 3.5, 3.3, np.nan],
            "odds_away": [5.0, 5.0, 4.5, np.nan],
        }
    )
    # Match 1: home edge 0.60 * 2.0 - 1 = +0.20 -> bet home, wins
    # Match 2: same bet, loses
    # Match 3: best edge is 0.50 * 2.0 - 1 = 0 -> below min_edge, no bet
    # Match 4: no odds -> no bet


def test_flat_staking():
    bets = simulate_bets(bet_predictions(), min_edge=0.05, staking="flat", stake=10, bankroll=1000)
    assert len(bets) == 2
    assert bets["selection"].tolist() == ["H", "H"]
    assert bets["edge"].tolist() == pytest.approx([0.2, 0.2])
    assert bets["stake"].tolist() == [10, 10]
    assert bets["profit"].tolist() == pytest.approx([10, -10])
    assert bets["bankroll"].tolist() == pytest.approx([1010, 1000])


def test_kelly_staking_scales_with_the_bankroll():
    bets = simulate_bets(
        bet_predictions(), min_edge=0.05, staking="kelly", bankroll=1000, kelly_scale=0.25
    )
    # Kelly fraction is 0.2, a quarter of that is 5% of the current bankroll.
    assert bets["stake"].tolist() == pytest.approx([50, 52.5])
    assert bets["bankroll"].tolist() == pytest.approx([1050, 997.5])


def test_picks_the_outcome_with_the_largest_edge():
    predictions = bet_predictions().iloc[[0]].assign(p_home=0.30, p_draw=0.30, p_away=0.40)
    bets = simulate_bets(predictions, min_edge=0.05)
    # Edges: home -0.40, draw +0.05, away +1.00
    assert bets["selection"].tolist() == ["A"]
    assert bets["profit"].tolist() == pytest.approx([-10])


def test_summarise_bets():
    bets = simulate_bets(bet_predictions(), min_edge=0.05, staking="flat", stake=10, bankroll=1000)
    summary = summarise_bets(bets)
    assert summary["n_bets"] == 2
    assert summary["win_rate"] == pytest.approx(0.5)
    assert summary["total_staked"] == pytest.approx(20)
    assert summary["profit"] == pytest.approx(0)
    assert summary["roi"] == pytest.approx(0)
    assert summary["max_drawdown"] == pytest.approx(-10 / 1010)
