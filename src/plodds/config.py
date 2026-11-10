"""Project settings: which seasons to use and where files live.

Paths are relative to the directory you run commands from, so run everything
from the repository root.
"""

from pathlib import Path

# football-data.co.uk names each season by its two years, e.g. "2324" is 2023-24.
SEASONS = ["1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526"]

# E0 is the Premier League.
URL_TEMPLATE = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "pl.db"

# Bookmaker code -> column prefixes used for it in the raw files.
# "Avg" and "Max" are the average and best price across many bookmakers; before
# 2019-20 the files call them BbAv and BbMx.
BOOKMAKER_PREFIXES = {
    "B365": ["B365"],
    "PS": ["PS"],
    "Avg": ["Avg", "BbAv"],
    "Max": ["Max", "BbMx"],
}

BOOKMAKER_NAMES = {
    "B365": "Bet365",
    "PS": "Pinnacle",
    "Avg": "Market average",
    "Max": "Market best price",
}
