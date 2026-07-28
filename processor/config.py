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

_FALSY: frozenset[str] = frozenset({"false", "0", "no", "off"})
"""Environment spellings accepted as boolean false.

This set is explicit so that ``_boolean`` can reject any other value. If an
unknown value gave false, then ``WRITE_COMMENTS=flase`` discarded every
annotation and gave no error. A mistyped numeric setting always raises.
"""


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
    write_subject_metadata: bool

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "Config":
        """Build a Config from environment variables, applying defaults.

        Raise ValueError, and do not use a default, if ``PERSYST_TIMEZONE`` names
        an unknown zone, if a numeric or boolean setting is unreadable, or if
        ``OUTPUT_FILENAME`` is not a bare filename.
        """
        return cls(
            input_dir=Path(env.get("INPUT_DIR", "/data/input")),
            output_dir=Path(env.get("OUTPUT_DIR", "/data/output")),
            output_filename=_filename(env.get("OUTPUT_FILENAME")),
            timezone=_zone(env.get("PERSYST_TIMEZONE", "UTC")),
            target_chunk_bytes=_positive_int(
                env, "CHUNK_TARGET_BYTES", TARGET_CHUNK_BYTES
            ),
            compression_level=_compression_level(
                env, "COMPRESSION_LEVEL", DEFAULT_COMPRESSION_LEVEL
            ),
            strip_ref_suffix=_boolean(env, "STRIP_REF_SUFFIX", default=False),
            write_comments=_boolean(env, "WRITE_COMMENTS", default=True),
            write_subject_metadata=_boolean(
                env, "WRITE_SUBJECT_METADATA", default=True
            ),
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


def _filename(raw: str | None) -> str | None:
    """Validate ``OUTPUT_FILENAME`` as a bare filename, or None if it is unset.

    ``output_dir / name`` discards the directory if ``name`` is absolute, and
    ``..`` moves out of the directory. An unchecked value can therefore write to
    any path that the process can reach. On Pennsieve these values arrive as
    workflow parameters, and an operator does not always type them. A second
    command-line argument has no such restriction.
    """
    name = (raw or "").strip()
    if not name:
        return None
    if (
        name in {".", ".."}
        or "/" in name
        or "\\" in name
        or Path(name).is_absolute()
    ):
        raise ValueError(
            f"OUTPUT_FILENAME must be a bare filename, got {name!r}"
        )
    return name


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
    """Read a boolean setting, or raise ValueError on an unknown spelling."""
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    raise ValueError(
        f"{key} must be one of {sorted(_TRUTHY | _FALSY)}, got {raw!r}"
    )
