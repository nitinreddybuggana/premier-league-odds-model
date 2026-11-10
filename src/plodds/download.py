"""Download the raw season files from football-data.co.uk."""

import time
import urllib.request
from pathlib import Path

from plodds.config import RAW_DIR, SEASONS, URL_TEMPLATE


def raw_path(season: str, raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / f"E0_{season}.csv"


def download_season(season: str, raw_dir: Path = RAW_DIR, force: bool = False) -> Path:
    """Download one season's CSV and return where it was saved.

    A file that is already on disk is kept unless ``force`` is set, so re-running
    the pipeline does not hit the site again. Use ``force`` for the current
    season, whose file grows as matches are played.
    """
    path = raw_path(season, raw_dir)
    if path.exists() and not force:
        return path

    raw_dir.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        URL_TEMPLATE.format(season=season),
        headers={"User-Agent": "plodds (student research project)"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        content = response.read()
    if not content.strip():
        raise ValueError(f"Season {season}: the downloaded file is empty")
    path.write_bytes(content)
    return path


def download_all(seasons=SEASONS, raw_dir: Path = RAW_DIR, force: bool = False) -> list[Path]:
    paths = []
    for season in seasons:
        already_there = raw_path(season, raw_dir).exists() and not force
        paths.append(download_season(season, raw_dir, force))
        if not already_there:
            time.sleep(1)  # be polite to a free site
    return paths
