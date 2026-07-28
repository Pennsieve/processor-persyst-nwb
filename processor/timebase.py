"""Absolute session start, segment layout, and per-sample timestamps.

`[SampleTimes]` values use file-dependent epochs.
In one sample, they are Unix timestamps aligned with `TestDate` and `TestTime`.
In another, they correspond to neither Unix time nor seconds since midnight.
Therefore, only relative offsets between entries are reliable,
and the absolute start time must be obtained from another field.

The values are also rounded to millisecond precision.
At sampling rates above approximately 1024 Hz, this resolution exceeds the
inter-sample gap threshold. See `segment_spans`.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import numpy as np
import numpy.typing as npt

from processor.constants import (
    FILETIME_EPOCH_OFFSET_S,
    FILETIME_TICKS_PER_SECOND,
    GAP_THRESHOLD_PERIODS,
    MIN_SEGMENTS_FOR_SPANS,
)
from processor.layout import Layout, Segment

logger = logging.getLogger(__name__)

_DATE_FORMATS: tuple[str, ...] = (
    "%m/%d/%Y",
    "%m/%d/%y",
    "%d-%m-%Y",
    "%d-%m-%y",
    "%Y.%m.%d",
)
"""``TestDate`` spellings seen in the wild; Persyst changes these without notice."""

_TIME_FORMATS: tuple[str, ...] = ("%H:%M:%S.%f", "%H:%M:%S", "%H:%M")
"""``TestTime`` spellings; fractional seconds appear in real files."""

_PLAUSIBLE_EPOCH_RANGE: tuple[float, float] = (1e8, 4e9)
"""Range within which a ``[SampleTimes]`` value is believable as Unix seconds.

The floor sits at 1973, far above any offset-style value Persyst writes (the
largest seen is ~1.9e5) and far below any real recording date. A year-2000
recording is 9.5e8, so a 1e9 floor would wrongly reject it. The ceiling is 2096.
"""

_MIN_SNAP_TOLERANCE_S = 0.002
"""Floor on the millisecond-snapping tolerance, at ~4x the quantisation error."""

_FALLBACK_START = datetime(1970, 1, 1, tzinfo=UTC)
"""Last-resort session start; pynwb requires a tz-aware instant."""


@dataclass(frozen=True, slots=True)
class SegmentSpan:
    """A contiguous run of samples and when it starts.

    ``stop_sample`` is exclusive.
    ``offset_s`` is seconds from the first sample of the recording,
      so span 0 always has ``offset_s == 0.0``.
    """

    start_sample: int
    stop_sample: int
    offset_s: float

    @property
    def n_samples(self) -> int:
        """Samples this span covers."""
        return self.stop_sample - self.start_sample


def filetime_to_datetime(ticks: int) -> datetime:
    """Convert a Windows FILETIME to a UTC datetime.

    FILETIME counts 100 ns ticks since 1601-01-01 UTC. Sub-second precision is
    preserved. Real NeuroPace stamps carry a half-second fraction.
    """
    seconds = ticks / FILETIME_TICKS_PER_SECOND - FILETIME_EPOCH_OFFSET_S
    return datetime.fromtimestamp(seconds, tz=UTC)


def parse_test_datetime(
    testdate: str | None, testtime: str | None, tz: ZoneInfo
) -> datetime | None:
    """Combine ``TestDate`` and ``TestTime`` into an instant in ``tz``.

    Return None when either field is absent, empty, or in no recognised format.
    Persyst records no timezone with these fields, so ``tz`` supplies the missing
    context and the result is only as good as that assumption.
    """
    if not testdate or not testtime:
        return None

    date = _try_formats(testdate.strip(), _DATE_FORMATS)
    if date is None:
        logger.warning("unrecognised TestDate format: %r", testdate)
        return None

    time = _try_formats(testtime.strip(), _TIME_FORMATS)
    if time is None:
        logger.warning("unrecognised TestTime format: %r", testtime)
        return None

    return datetime(
        date.year,
        date.month,
        date.day,
        time.hour,
        time.minute,
        time.second,
        time.microsecond,
        tzinfo=tz,
    )


def resolve_session_start(layout: Layout, tz: ZoneInfo) -> tuple[datetime, str]:
    """Determine the recording's absolute start and say where it came from.

    A NeuroPace FILETIME is preferred when present because it is an unambiguous UTC
    instant, whereas ``TestDate``/``TestTime`` are bare wall-clock and have to be
    reinterpreted in ``tz``. A ``[SampleTimes]`` value is used only when it is
    believable as Unix seconds. The returned source string is logged and recorded
    in the NWB session description.
    """
    for key in ("ecogtimestampasutc", "layoutfiletimestampasutc"):
        raw = layout.np_file_info.get(key)
        if raw:
            try:
                return filetime_to_datetime(int(raw)), f"[NP_FileInfo] {key}"
            except ValueError:
                logger.warning("unreadable FILETIME in %s: %r", key, raw)

    from_patient = parse_test_datetime(
        layout.patient.get("testdate"), layout.patient.get("testtime"), tz
    )
    if from_patient is not None:
        return from_patient, f"[Patient] TestDate/TestTime in {tz.key}"

    if layout.segments:
        first = layout.segments[0].start_time_s
        low, high = _PLAUSIBLE_EPOCH_RANGE
        if low < first < high:
            return (
                datetime.fromtimestamp(first, tz=UTC),
                "[SampleTimes] first value as Unix seconds",
            )

    logger.warning(
        "no usable start time in the lay header; defaulting to %s",
        _FALLBACK_START.isoformat(),
    )
    return _FALLBACK_START, "fallback (epoch)"


def segment_spans(
    segments: tuple[Segment, ...], n_samples: int, rate: float
) -> tuple[SegmentSpan, ...]:
    """Turn ``[SampleTimes]`` entries into spans tiling ``[0, n_samples)``.

    Offsets are relative to the first entry, so the file's unknown reference epoch
    never matters. Falls back to a single contiguous span when there is no usable
    ``[SampleTimes]`` data or when its times run backwards.

    Boundaries within ``max(1.5 / rate, 0.002)`` seconds of where a gapless
    recording would put them are snapped to the exact value. Persyst rounds these
    times to milliseconds, an error of up to 500 us. Above ~1024 Hz that exceeds
    the ``2 / rate`` gap threshold. Without snapping, a contiguous 2048 Hz
    recording reports false gaps and can emit non-monotonic timestamps.

    Raise ValueError if ``n_samples`` is not positive.
    """
    if n_samples <= 0:
        raise ValueError(f"recording has no samples: {n_samples!r}")

    usable = _usable_segments(segments, n_samples)
    if len(usable) < MIN_SEGMENTS_FOR_SPANS:
        return (SegmentSpan(0, n_samples, 0.0),)

    if any(
        usable[i].start_time_s < usable[i - 1].start_time_s
        for i in range(1, len(usable))
    ):
        logger.warning(
            "[SampleTimes] times are not monotonic; treating data as contiguous"
        )
        return (SegmentSpan(0, n_samples, 0.0),)

    starts = [seg.start_sample for seg in usable]
    stops = [*starts[1:], n_samples]
    tolerance = max(1.5 / rate, _MIN_SNAP_TOLERANCE_S)
    base_sample, base_time = starts[0], usable[0].start_time_s

    spans = []
    snapped = 0
    for seg, start, stop in zip(usable, starts, stops, strict=True):
        expected = (start - base_sample) / rate
        observed = seg.start_time_s - base_time
        if abs(observed - expected) <= tolerance:
            offset = expected
            snapped += observed != expected
        else:
            offset = observed
        spans.append(SegmentSpan(start, stop, offset))

    if snapped:
        logger.info(
            "snapped %d millisecond-quantised segment boundary/ies to exact times",
            snapped,
        )
    return tuple(spans)


def has_gaps(spans: tuple[SegmentSpan, ...], rate: float) -> bool:
    """Whether any span starts later than continuous sampling would place it.

    A jump must exceed ``GAP_THRESHOLD_PERIODS`` sample periods to count. That
    tolerates ordinary jitter and still catches a dropped sample.
    """
    threshold = GAP_THRESHOLD_PERIODS / rate
    for previous, current in zip(spans, spans[1:], strict=False):
        expected = previous.offset_s + previous.n_samples / rate
        if current.offset_s - expected > threshold:
            return True
    return False


def timestamps_seconds(
    spans: tuple[SegmentSpan, ...], rate: float
) -> npt.NDArray[np.float64]:
    """Build one timestamp per sample, in seconds from the session start.

    Offsets stay small because they are relative, so float64 spacing is orders of
    magnitude finer than the gap threshold even across a multi-week recording.
    """
    parts = [
        span.offset_s + np.arange(span.n_samples, dtype=np.float64) / rate
        for span in spans
        if span.n_samples > 0
    ]
    if not parts:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(parts)


def _try_formats(value: str, formats: tuple[str, ...]) -> datetime | None:
    """Return the first successful ``strptime`` parse, or None if none match."""
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)  # noqa: DTZ007
        except ValueError:
            continue
    return None


def _usable_segments(
    segments: tuple[Segment, ...], n_samples: int
) -> list[Segment]:
    """Sort, deduplicate, and drop entries that fall outside the data.

    Persyst may leave stale ``[SampleTimes]`` entries describing a longer original
    recording than the ``.dat`` actually holds.
    """
    seen: set[int] = set()
    usable = []
    dropped = 0
    for seg in sorted(segments, key=lambda s: s.start_sample):
        if seg.start_sample < 0 or seg.start_sample >= n_samples:
            dropped += 1
            continue
        if seg.start_sample in seen:
            continue
        seen.add(seg.start_sample)
        usable.append(seg)

    if dropped:
        logger.warning(
            "dropped %d [SampleTimes] entry/ies at or beyond sample %d",
            dropped,
            n_samples,
        )

    if usable and usable[0].start_sample != 0:
        logger.warning(
            "[SampleTimes] starts at sample %d, not 0; prepending a span",
            usable[0].start_sample,
        )
        usable.insert(0, Segment(0, usable[0].start_time_s))

    return usable
