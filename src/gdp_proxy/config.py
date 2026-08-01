"""Configuration loading. No country-specific constants belong outside countries.yaml."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "countries.yaml"
DATA_DIR = REPO_ROOT / "data"


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or malformed."""


def load_env() -> None:
    """Load .env from the repo root. Safe to call repeatedly."""
    load_dotenv(REPO_ROOT / ".env", override=False)


def require_env(key: str) -> str:
    """Return an environment variable or fail with an actionable message."""
    load_env()
    value = os.getenv(key, "").strip()
    if not value:
        raise ConfigError(
            f"{key} is not set. Copy .env.example to .env and fill it in. "
            f"See SETUP.md for where to find this value."
        )
    return value


@lru_cache(maxsize=1)
def _raw_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise ConfigError(f"Missing config file at {CONFIG_PATH}")
    with CONFIG_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def country_config(name: str | None = None) -> dict[str, Any]:
    """Return the merged defaults plus country block for the target country."""
    load_env()
    name = (name or os.getenv("TARGET_COUNTRY", "india")).lower()
    raw = _raw_config()
    if name not in raw:
        available = sorted(k for k in raw if k != "defaults")
        raise ConfigError(f"Unknown country '{name}'. Available: {', '.join(available)}")
    merged = dict(raw.get("defaults", {}))
    merged.update(raw[name])
    merged["country"] = name
    return merged
