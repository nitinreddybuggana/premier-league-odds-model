import pandas as pd
import pytest

from plodds import clean


def test_season_label():
    assert clean.season_label("2324") == "2023-24"
    assert clean.season_label("1617") == "2016-17"
    with pytest.raises(ValueError):
        clean.season_label("2023-24")  # codes are four digits, labels are not codes

def test_parse_dates_handles_two_and_four_digit_years():
    parsed = clean.parse_dates(pd.Series(["13/08/16", "01/02/2024", "9/3/19"]))
    assert list(parsed) == [
        pd.Timestamp("2016-08-13"),
        pd.Timestamp("2024-02-01"),
        pd.Timestamp("2019-03-09"),
    ]


def test_matches_are_typed_and_empty_rows_dropped(cleaned_sample):
    matches, _ = cleaned_sample
    assert len(matches) == 6
    assert list(matches.columns) == [
        "season", "date", "home_team", "away_team", "home_goals", "away_goals", "result",
    ]
    assert (matches["season"] == "2023-24").all()
    assert matches["date"].iloc[0] == pd.Timestamp("2023-08-11")
    assert matches["home_goals"].tolist() == [3, 1, 0, 2, 2, 1]
    assert matches["result"].tolist() == ["H", "D", "A", "H", "D", "H"]


def test_odds_are_long_with_one_row_per_match_bookmaker_timing(cleaned_sample):
    matches, odds = cleaned_sample
    assert not odds.duplicated(["row", "bookmaker", "timing"]).any()
    assert set(odds["bookmaker"]) == {"B365", "PS", "Avg", "Max"}
    assert set(odds["timing"]) == {"pre", "closing"}
    assert set(odds["row"]) <= set(matches.index)
    # 6 matches x 4 bookmakers x 2 timings, minus the one missing Pinnacle closing line
    assert len(odds) == 6 * 4 * 2 - 1

    first = odds[(odds["row"] == 0) & (odds["bookmaker"] == "Avg") & (odds["timing"] == "closing")]
    assert first[["home", "draw", "away"]].values.tolist() == [[1.37, 5.20, 8.40]]


def test_missing_odds_are_dropped_not_filled(cleaned_sample):
    _, odds = cleaned_sample
    pinnacle_closing = odds[(odds["bookmaker"] == "PS") & (odds["timing"] == "closing")]
    assert 2 not in set(pinnacle_closing["row"])
    assert odds[["home", "draw", "away"]].notna().all().all()


def test_old_column_names_for_market_average(raw_sample):
    old_style = raw_sample.rename(
        columns={"AvgH": "BbAvH", "AvgD": "BbAvD", "AvgA": "BbAvA"}
    ).drop(columns=["AvgCH", "AvgCD", "AvgCA"])
    _, odds = clean.clean_season(old_style, "1718")
    avg = odds[odds["bookmaker"] == "Avg"]
    assert set(avg["timing"]) == {"pre"}
    assert len(avg) == 6


def test_result_that_contradicts_score_is_an_error(raw_sample):
    raw_sample.loc[0, "FTR"] = "A"  # the score was 3-0 to the home side
    with pytest.raises(ValueError, match="does not match the score"):
        clean.clean_season(raw_sample, "2324")


def test_missing_required_column_is_an_error(raw_sample):
    with pytest.raises(ValueError, match="missing columns"):
        clean.clean_season(raw_sample.drop(columns=["FTHG"]), "2324")
