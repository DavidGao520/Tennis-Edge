"""Configuration loading: YAML file + environment variable overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DatabaseConfig:
    path: str = "data/tennis_edge.db"


@dataclass(frozen=True)
class DataConfig:
    raw_dir: str = "data/raw"
    tml_dir: str = "data/tml"
    sackmann_base_url: str = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
    match_years_start: int = 2000
    match_years_end: int = 2026


@dataclass(frozen=True)
class RatingsConfig:
    initial_mu: float = 1500.0
    initial_phi: float = 350.0
    initial_sigma: float = 0.06
    tau: float = 0.5
    rating_period_days: int = 30


@dataclass(frozen=True)
class ModelConfig:
    type: str = "logistic"
    features: list[str] = field(
        default_factory=lambda: ["glicko2", "surface", "fatigue", "h2h", "form", "tournament"]
    )
    train_start_year: int = 2005
    test_start_year: int = 2020
    artifacts_dir: str = "data/models"


@dataclass(frozen=True)
class StrategyConfig:
    kelly_fraction: float = 0.25
    max_bet_fraction: float = 0.05
    min_edge: float = 0.03
    bankroll: float = 1000.0


@dataclass(frozen=True)
class RiskConfig:
    max_position_per_market: float = 50.0
    max_total_exposure: float = 200.0
    daily_loss_limit: float = 100.0
    kill_switch: bool = False


@dataclass(frozen=True)
class KalshiConfig:
    base_url: str = "https://trading-api.kalshi.com/trade-api/v2"
    demo_base_url: str = "https://demo-api.kalshi.co/trade-api/v2"
    api_key_id: str = ""
    private_key_path: str = ""
    use_demo: bool = True
    paper_mode: bool = True

    @property
    def effective_base_url(self) -> str:
        return self.demo_base_url if self.use_demo else self.base_url


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    file: str = "data/tennis_edge.log"


@dataclass(frozen=True)
class BacktestConfig:
    start_year: int = 2015
    end_year: int = 2024
    initial_bankroll: float = 10000.0
    reports_dir: str = "data/reports"
    retrain_interval_days: int = 180


@dataclass(frozen=True)
class AppConfig:
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    data: DataConfig = field(default_factory=DataConfig)
    ratings: RatingsConfig = field(default_factory=RatingsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    kalshi: KalshiConfig = field(default_factory=KalshiConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    project_root: str = "."


_SECTION_MAP = {
    "database": DatabaseConfig,
    "data": DataConfig,
    "ratings": RatingsConfig,
    "model": ModelConfig,
    "strategy": StrategyConfig,
    "risk": RiskConfig,
    "kalshi": KalshiConfig,
    "logging": LoggingConfig,
    "backtest": BacktestConfig,
}


def _apply_env_overrides(raw: dict) -> dict:
    """Override config values from env vars like TENNIS_EDGE__KALSHI__API_KEY_ID."""
    prefix = "TENNIS_EDGE__"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix) :].lower().split("__")
        if len(parts) == 2:
            section, field_name = parts
            if section not in raw:
                raw[section] = {}
            raw[section][field_name] = value
    return raw


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load configuration from YAML file with env var overrides."""
    raw: dict = {}

    if config_path is not None:
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                raw = yaml.safe_load(f) or {}

    raw = _apply_env_overrides(raw)

    sections = {}
    for section_name, cls in _SECTION_MAP.items():
        section_data = raw.get(section_name, {})
        if isinstance(section_data, dict):
            sections[section_name] = cls(**{k: v for k, v in section_data.items() if k in cls.__dataclass_fields__})
        else:
            sections[section_name] = cls()

    project_root = str(Path(config_path).parent.parent) if config_path else "."

    return AppConfig(**sections, project_root=project_root)
