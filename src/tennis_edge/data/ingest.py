"""Download and ingest Jeff Sackmann's tennis_atp data into SQLite."""

from __future__ import annotations

import csv
import io
import logging
from pathlib import Path

import httpx
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..config import AppConfig
from .db import Database

logger = logging.getLogger(__name__)

SACKMANN_FILES = {
    "players": "atp_players.csv",
    "rankings_current": "atp_rankings_current.csv",
    "rankings_10s": "atp_rankings_10s.csv",
    "rankings_20s": "atp_rankings_20s.csv",
}


def _download(base_url: str, filename: str, raw_dir: Path) -> Path:
    """Download a file if not already cached locally."""
    local_path = raw_dir / filename
    if local_path.exists():
        logger.info("Using cached %s", filename)
        return local_path

    url = f"{base_url}/{filename}"
    logger.info("Downloading %s", url)
    resp = httpx.get(url, follow_redirects=True, timeout=60.0)
    resp.raise_for_status()

    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(resp.content)
    return local_path


def _safe_int(val: str) -> int | None:
    if not val or val.strip() == "":
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def _safe_date(val: str) -> str | None:
    """Convert YYYYMMDD to YYYY-MM-DD."""
    if not val or len(val) < 8:
        return None
    try:
        return f"{val[:4]}-{val[4:6]}-{val[6:8]}"
    except (IndexError, TypeError):
        return None


def ingest_players(db: Database, base_url: str, raw_dir: Path) -> int:
    """Download and ingest atp_players.csv."""
    path = _download(base_url, "atp_players.csv", raw_dir)
    text = path.read_text(encoding="utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))

    rows = []
    for row in reader:
        pid = _safe_int(row.get("player_id", ""))
        if pid is None:
            continue
        birth = _safe_date(row.get("dob", ""))
        rows.append((
            pid,
            row.get("name_first", "").strip(),
            row.get("name_last", "").strip(),
            row.get("hand", "U").strip(),
            birth,
            row.get("ioc", "").strip(),
            _safe_int(row.get("height", "")),
        ))

    db.executemany(
        "INSERT OR IGNORE INTO players (player_id, first_name, last_name, hand, birth_date, country_code, height_cm) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    db.commit()
    logger.info("Ingested %d players", len(rows))
    return len(rows)


def ingest_matches_year(db: Database, base_url: str, raw_dir: Path, year: int) -> int:
    """Download and ingest atp_matches_{year}.csv."""
    filename = f"atp_matches_{year}.csv"
    try:
        path = _download(base_url, filename, raw_dir)
    except httpx.HTTPStatusError:
        logger.warning("No match file for year %d", year)
        return 0

    text = path.read_text(encoding="utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))

    rows = []
    for row in reader:
        tourney_date_raw = row.get("tourney_date", "")
        tourney_date = _safe_date(tourney_date_raw)
        if not tourney_date:
            continue

        winner_id = _safe_int(row.get("winner_id", ""))
        loser_id = _safe_int(row.get("loser_id", ""))
        if winner_id is None or loser_id is None:
            continue

        rows.append((
            row.get("tourney_id", ""),
            row.get("tourney_name", ""),
            row.get("surface", ""),
            _safe_int(row.get("draw_size", "")),
            row.get("tourney_level", ""),
            tourney_date,
            _safe_int(row.get("match_num", "")),
            winner_id,
            loser_id,
            row.get("score", ""),
            _safe_int(row.get("best_of", "")),
            row.get("round", ""),
            _safe_int(row.get("minutes", "")),
            _safe_int(row.get("winner_rank", "")),
            _safe_int(row.get("loser_rank", "")),
            _safe_int(row.get("winner_seed", "")),
            _safe_int(row.get("loser_seed", "")),
            # Winner stats
            _safe_int(row.get("w_ace", "")),
            _safe_int(row.get("w_df", "")),
            _safe_int(row.get("w_svpt", "")),
            _safe_int(row.get("w_1stIn", "")),
            _safe_int(row.get("w_1stWon", "")),
            _safe_int(row.get("w_2ndWon", "")),
            _safe_int(row.get("w_SvGms", "")),
            _safe_int(row.get("w_bpSaved", "")),
            _safe_int(row.get("w_bpFaced", "")),
            # Loser stats
            _safe_int(row.get("l_ace", "")),
            _safe_int(row.get("l_df", "")),
            _safe_int(row.get("l_svpt", "")),
            _safe_int(row.get("l_1stIn", "")),
            _safe_int(row.get("l_1stWon", "")),
            _safe_int(row.get("l_2ndWon", "")),
            _safe_int(row.get("l_SvGms", "")),
            _safe_int(row.get("l_bpSaved", "")),
            _safe_int(row.get("l_bpFaced", "")),
        ))

    db.executemany(
        "INSERT OR IGNORE INTO matches "
        "(tourney_id, tourney_name, surface, draw_size, tourney_level, tourney_date, match_num, "
        "winner_id, loser_id, score, best_of, round, minutes, winner_rank, loser_rank, "
        "winner_seed, loser_seed, "
        "w_ace, w_df, w_svpt, w_1st_in, w_1st_won, w_2nd_won, w_sv_gms, w_bp_saved, w_bp_faced, "
        "l_ace, l_df, l_svpt, l_1st_in, l_1st_won, l_2nd_won, l_sv_gms, l_bp_saved, l_bp_faced) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    db.commit()
    return len(rows)


def ingest_rankings(db: Database, base_url: str, raw_dir: Path) -> int:
    """Download and ingest ranking files."""
    total = 0
    for key, filename in SACKMANN_FILES.items():
        if not key.startswith("rankings"):
            continue
        try:
            path = _download(base_url, filename, raw_dir)
        except httpx.HTTPStatusError:
            logger.warning("Ranking file %s not found", filename)
            continue

        text = path.read_text(encoding="utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text))

        rows = []
        for row in reader:
            date_raw = row.get("ranking_date", "")
            ranking_date = _safe_date(date_raw)
            player_id = _safe_int(row.get("player", ""))
            ranking = _safe_int(row.get("rank", ""))
            if not ranking_date or player_id is None or ranking is None:
                continue
            rows.append((
                ranking_date,
                ranking,
                player_id,
                _safe_int(row.get("points", "")),
            ))

        db.executemany(
            "INSERT OR IGNORE INTO rankings (ranking_date, ranking, player_id, ranking_points) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )
        db.commit()
        total += len(rows)
        logger.info("Ingested %d rankings from %s", len(rows), filename)

    return total


# --- TennisMyLife ingestion ---

TML_LEVEL_MAP = {
    "250": "D",
    "500": "A",
    "G": "G",
    "M": "M",
    "A": "A",
    "D": "D",
    "F": "F",
}


def _normalize_name(name: str) -> str:
    """Normalize player name for matching."""
    return name.lower().replace("-", " ").replace("  ", " ").strip()


def _build_name_to_id(db: Database) -> dict[str, int]:
    """Build a mapping from normalized name -> player_id from existing players table."""
    rows = db.query_all("SELECT player_id, first_name, last_name FROM players")
    name_map: dict[str, int] = {}
    for row in rows:
        full = _normalize_name(f"{row['first_name']} {row['last_name']}")
        name_map[full] = row["player_id"]
        # Also index by last_name only for fallback
        last = _normalize_name(row["last_name"])
        if last not in name_map:
            name_map[last] = row["player_id"]
    return name_map


def _resolve_player(
    name: str, name_map: dict[str, int], db: Database, _next_id: list[int]
) -> int:
    """Resolve a player name to a player_id, creating a new player if needed."""
    norm = _normalize_name(name)
    if norm in name_map:
        return name_map[norm]

    # Try with last name only
    parts = name.strip().split()
    if len(parts) >= 2:
        last = _normalize_name(parts[-1])
        # Check full name variants
        for stored_name, pid in name_map.items():
            if stored_name.endswith(last) and stored_name[0] == _normalize_name(parts[0])[0]:
                name_map[norm] = pid
                return pid

    # Create new player
    pid = _next_id[0]
    _next_id[0] += 1
    first = parts[0] if parts else name
    last_name = " ".join(parts[1:]) if len(parts) > 1 else name
    db.execute(
        "INSERT OR IGNORE INTO players (player_id, first_name, last_name, hand, country_code) "
        "VALUES (?, ?, ?, 'U', '')",
        (pid, first, last_name),
    )
    name_map[norm] = pid
    logger.info("Created new player: %s (id=%d)", name, pid)
    return pid


def ingest_tml_file(
    db: Database,
    csv_path: Path,
    name_map: dict[str, int],
    next_id: list[int],
) -> int:
    """Ingest a TennisMyLife CSV file into the matches table."""
    text = csv_path.read_text(encoding="utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))

    rows = []
    for row in reader:
        tourney_date_raw = row.get("tourney_date", "")
        tourney_date = _safe_date(tourney_date_raw)
        if not tourney_date:
            continue

        winner_name = row.get("winner_name", "").strip()
        loser_name = row.get("loser_name", "").strip()
        if not winner_name or not loser_name:
            continue

        winner_id = _resolve_player(winner_name, name_map, db, next_id)
        loser_id = _resolve_player(loser_name, name_map, db, next_id)

        level = TML_LEVEL_MAP.get(row.get("tourney_level", ""), row.get("tourney_level", "D"))

        rows.append((
            row.get("tourney_id", ""),
            row.get("tourney_name", ""),
            row.get("surface", ""),
            _safe_int(row.get("draw_size", "")),
            level,
            tourney_date,
            _safe_int(row.get("match_num", "")),
            winner_id,
            loser_id,
            row.get("score", ""),
            _safe_int(row.get("best_of", "")),
            row.get("round", ""),
            _safe_int(row.get("minutes", "")),
            _safe_int(row.get("winner_rank", "")),
            _safe_int(row.get("loser_rank", "")),
            _safe_int(row.get("winner_seed", "")),
            _safe_int(row.get("loser_seed", "")),
            _safe_int(row.get("w_ace", "")),
            _safe_int(row.get("w_df", "")),
            _safe_int(row.get("w_svpt", "")),
            _safe_int(row.get("w_1stIn", "")),
            _safe_int(row.get("w_1stWon", "")),
            _safe_int(row.get("w_2ndWon", "")),
            _safe_int(row.get("w_SvGms", "")),
            _safe_int(row.get("w_bpSaved", "")),
            _safe_int(row.get("w_bpFaced", "")),
            _safe_int(row.get("l_ace", "")),
            _safe_int(row.get("l_df", "")),
            _safe_int(row.get("l_svpt", "")),
            _safe_int(row.get("l_1stIn", "")),
            _safe_int(row.get("l_1stWon", "")),
            _safe_int(row.get("l_2ndWon", "")),
            _safe_int(row.get("l_SvGms", "")),
            _safe_int(row.get("l_bpSaved", "")),
            _safe_int(row.get("l_bpFaced", "")),
        ))

    db.executemany(
        "INSERT OR IGNORE INTO matches "
        "(tourney_id, tourney_name, surface, draw_size, tourney_level, tourney_date, match_num, "
        "winner_id, loser_id, score, best_of, round, minutes, winner_rank, loser_rank, "
        "winner_seed, loser_seed, "
        "w_ace, w_df, w_svpt, w_1st_in, w_1st_won, w_2nd_won, w_sv_gms, w_bp_saved, w_bp_faced, "
        "l_ace, l_df, l_svpt, l_1st_in, l_1st_won, l_2nd_won, l_sv_gms, l_bp_saved, l_bp_faced) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    db.commit()
    return len(rows)


def ingest_all(config: AppConfig) -> dict[str, int]:
    """Run the full ingestion pipeline. Returns counts per category."""
    base_url = config.data.sackmann_base_url
    raw_dir = Path(config.project_root) / config.data.raw_dir
    db_path = Path(config.project_root) / config.database.path

    with Database(db_path) as db:
        db.initialize()

        counts: dict[str, int] = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ) as progress:
            # Players
            task = progress.add_task("Ingesting players...", total=None)
            counts["players"] = ingest_players(db, base_url, raw_dir)
            progress.update(task, completed=True, description=f"Players: {counts['players']}")

            # Matches by year
            total_matches = 0
            for year in range(config.data.match_years_start, config.data.match_years_end + 1):
                task = progress.add_task(f"Ingesting matches {year}...", total=None)
                n = ingest_matches_year(db, base_url, raw_dir, year)
                total_matches += n
                progress.update(task, completed=True, description=f"Matches {year}: {n}")
            counts["matches"] = total_matches

            # Rankings
            task = progress.add_task("Ingesting rankings...", total=None)
            counts["rankings"] = ingest_rankings(db, base_url, raw_dir)
            progress.update(task, completed=True, description=f"Rankings: {counts['rankings']}")

            # TennisMyLife supplementary files (2025, 2026, etc.)
            tml_dir = Path(config.project_root) / config.data.tml_dir
            if tml_dir.exists():
                name_map = _build_name_to_id(db)
                max_id_row = db.query_one("SELECT MAX(player_id) as m FROM players")
                next_id = [(max_id_row["m"] or 999999) + 1]
                total_tml = 0

                for csv_file in sorted(tml_dir.glob("*.csv")):
                    task = progress.add_task(f"Ingesting TML {csv_file.name}...", total=None)
                    n = ingest_tml_file(db, csv_file, name_map, next_id)
                    total_tml += n
                    progress.update(task, completed=True, description=f"TML {csv_file.name}: {n}")

                db.commit()
                counts["tml_matches"] = total_tml

    return counts
