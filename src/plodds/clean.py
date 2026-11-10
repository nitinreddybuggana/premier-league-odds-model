"""Turn one raw season file into two tidy tables: matches and odds.

The raw files are wide: one row per match, with three columns (home, draw,
away) for every bookmaker, and the set of columns changes between seasons.
Cleaning produces

* ``matches``: one row per match, with typed columns and consistent names
* ``odds``: one row per match, bookmaker and timing (pre-match or closing)

Keeping odds in long form means a new bookmaker is more rows, not more columns.
"""

from pathlib import Path

import pandas as pd

from plodds.config import BOOKMAKER_PREFIXES

REQUIRED_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]


def season_label(season: str) -> str:
    """'2324' -> '2023-24'."""
    if len(season) != 4 or not season.isdigit():
        raise ValueError(f"Season code should be four digits like '2324', got {season!r}")
    return f"20{season[:2]}-{season[2:]}"


def read_raw(path: Path) -> pd.DataFrame:
    """Read a raw CSV as it comes from the site, without interpreting it."""
    try:
        raw = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        raw = pd.read_csv(path, encoding="latin-1")
    raw.columns = [str(column).strip() for column in raw.columns]
    return raw


def parse_dates(dates: pd.Series) -> pd.Series:
    """Parse the site's day-first dates, which use 2-digit years in older files
    (13/08/16) and 4-digit years in newer ones (13/08/2016)."""
    text = dates.astype("string").str.strip()
    two_digit_year = text.str.fullmatch(r"\d{1,2}/\d{1,2}/\d{2}").fillna(False)
    parsed = pd.Series(pd.NaT, index=dates.index, dtype="datetime64[ns]")
    parsed[two_digit_year] = pd.to_datetime(text[two_digit_year], format="%d/%m/%y")
    parsed[~two_digit_year] = pd.to_datetime(
        text[~two_digit_year], format="%d/%m/%Y", errors="coerce"
    )
    return parsed


def clean_season(raw: pd.DataFrame, season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean one season. Returns ``(matches, odds)``.

    ``matches`` has columns season, date, home_team, away_team, home_goals,
    away_goals, result ('H', 'D' or 'A'). Its index, named ``row``, is what
    ``odds.row`` refers to.

    Rows without teams or a score (blank lines, unplayed fixtures) are dropped.
    A row whose recorded result disagrees with its score is an error, because
    that means the file is not what we think it is.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"Season {season}: raw file is missing columns {missing}")

    played = raw.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).reset_index(drop=True)

    matches = pd.DataFrame(
        {
            "season": season_label(season),
            "date": parse_dates(played["Date"]),
            "home_team": played["HomeTeam"].astype(str).str.strip(),
            "away_team": played["AwayTeam"].astype(str).str.strip(),
            "home_goals": played["FTHG"].astype(int),
            "away_goals": played["FTAG"].astype(int),
            "result": played["FTR"].astype(str).str.strip(),
        }
    )
    matches.index.name = "row"

    if matches["date"].isna().any():
        bad = played.loc[matches["date"].isna(), "Date"].tolist()
        raise ValueError(f"Season {season}: could not parse dates {bad[:5]}")

    from_score = pd.Series("D", index=matches.index)
    from_score[matches["home_goals"] > matches["away_goals"]] = "H"
    from_score[matches["home_goals"] < matches["away_goals"]] = "A"
    disagree = matches["result"] != from_score
    if disagree.any():
        raise ValueError(
            f"Season {season}: {int(disagree.sum())} rows have a result that "
            "does not match the score"
        )

    return matches, _extract_odds(played)


def _extract_odds(played: pd.DataFrame) -> pd.DataFrame:
    """Reshape the wide bookmaker columns into one row per (match, bookmaker, timing)."""
    pieces = []
    for bookmaker, prefixes in BOOKMAKER_PREFIXES.items():
        for prefix in prefixes:
            for timing, middle in (("pre", ""), ("closing", "C")):
                columns = [f"{prefix}{middle}{outcome}" for outcome in "HDA"]
                if not all(column in played.columns for column in columns):
                    continue
                piece = played[columns].apply(pd.to_numeric, errors="coerce")
                piece.columns = ["home", "draw", "away"]
                piece.insert(0, "timing", timing)
                piece.insert(0, "bookmaker", bookmaker)
                piece.insert(0, "row", played.index)
                pieces.append(piece)

    columns = ["row", "bookmaker", "timing", "home", "draw", "away"]
    if not pieces:
        return pd.DataFrame(columns=columns)

    odds = pd.concat(pieces, ignore_index=True)
    # Decimal odds are always above 1. Anything else is a gap or a typo in the file.
    valid = (odds[["home", "draw", "away"]] > 1).all(axis=1)
    odds = odds[valid].drop_duplicates(subset=["row", "bookmaker", "timing"])
    return odds.sort_values(["row", "bookmaker", "timing"]).reset_index(drop=True)
