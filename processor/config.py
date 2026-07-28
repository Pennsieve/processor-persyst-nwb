"""Runtime configuration drawn from the environment.

Every setting is an environment variable so a containerised run can be retuned
without a rebuild. (On Pennsieve specifically, a workflow's ``params`` arrive as
upper-cased environment variables, which is what makes these settable per
workflow.) Construction takes an explicit mapping rather than reading
``os.environ``, which keeps it pure and testable.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from processor.constants import (
    DEFAULT_COMPRESSION_LEVEL,
    MAX_COMPRESSION_LEVEL,
    TARGET_CHUNK_BYTES,
)

logger = logging.getLogger(__name__)

_TRUTHY: frozenset[str] = frozenset({"true", "1", "yes", "on"})
"""Environment spellings accepted as boolean true."""


@dataclass(frozen=True, slots=True)
class Config:
    """Settings for one conversion run."""

    input_dir: Path
    output_dir: Path
    output_filename: str | None
    timezone: ZoneInfo
    target_chunk_bytes: int
    compression_level: int
    strip_ref_suffix: bool
    write_comments: bool

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "Config":
        """Build a Config from environment variables, applying defaults.

        Raise ValueError when ``PERSYST_TIMEZONE`` names an unknown zone or a
        numeric setting cannot be read, rather than silently falling back.
        """
        return cls(
            input_dir=Path(env.get("INPUT_DIR", "/data/input")),
            output_dir=Path(env.get("OUTPUT_DIR", "/data/output")),
            output_filename=env.get("OUTPUT_FILENAME") or None,
            timezone=_zone(env.get("PERSYST_TIMEZONE", "UTC")),
            target_chunk_bytes=_positive_int(
                env, "CHUNK_TARGET_BYTES", TARGET_CHUNK_BYTES
            ),
            compression_level=_compression_level(
                env, "COMPRESSION_LEVEL", DEFAULT_COMPRESSION_LEVEL
            ),
            strip_ref_suffix=_boolean(env, "STRIP_REF_SUFFIX", default=False),
            write_comments=_boolean(env, "WRITE_COMMENTS", default=True),
        )


def _zone(name: str) -> ZoneInfo:
    """Resolve a timezone name.

    Persyst stores ``TestDate``/``TestTime`` as bare wall-clock with no zone, so
    this supplies the missing context. Raise ValueError on an unknown name.
    """
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"unknown PERSYST_TIMEZONE: {name!r}") from exc


def _positive_int(env: Mapping[str, str], key: str, default: int) -> int:
    """Read a strictly positive integer setting."""
    value = _int(env, key, default)
    if value <= 0:
        raise ValueError(f"{key} must be positive, got {value!r}")
    return value


def _compression_level(env: Mapping[str, str], key: str, default: int) -> int:
    """Read a gzip level in 0..9, where 0 disables compression."""
    value = _int(env, key, default)
    if not 0 <= value <= MAX_COMPRESSION_LEVEL:
        raise ValueError(
            f"{key} must be between 0 and {MAX_COMPRESSION_LEVEL}, got {value!r}"
        )
    return value


def _int(env: Mapping[str, str], key: str, default: int) -> int:
    """Read an integer setting, raising ValueError on a non-numeric value."""
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} is not an integer: {raw!r}") from exc


def _boolean(env: Mapping[str, str], key: str, *, default: bool) -> bool:
    """Read a boolean setting."""
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in _TRUTHY
