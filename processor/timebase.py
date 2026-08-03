"""Absolute session start, segment layout, and per-sample timestamps.

``[SampleTimes]`` values sit on a file-dependent epoch. In one file they are Unix
timestamps that agree with ``TestDate`` and ``TestTime``; in another they match
neither Unix time nor seconds since midnight. Only the differences between entries
are reliable, so the absolute start time comes from another field.

The values are also rounded to milliseconds. Above about 1024 Hz that rounding
error exceeds the gap threshold between samples. See ``segment_spans``.
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
"""``TestDate`` spellings real files use; Persyst changes them without notice."""

_TIME_FORMATS: tuple[str, ...] = ("%H:%M:%S.%f", "%H:%M:%S", "%H:%M")
"""``TestTime`` spellings; fractional seconds appear in real files."""

_PLAUSIBLE_EPOCH_RANGE: tuple[float, float] = (1e8, 4e9)
"""Range within which a ``[SampleTimes]`` value is believable as Unix seconds.

The floor sits at 1973, far above any offset-style value Persyst writes (the
largest seen is ~1.9e5) and far below any real recording date. A year-2000
recording is 9.5e8, so a 1e9 floor would wrongly reject it. The ceiling is 2096.
"""

_PLAUSIBLE_FILETIME_YEARS: tuple[int, int] = (1990, 2100)
"""Years in which a NeuroPace FILETIME is credible.

A FILETIME of 0, or a negative value, converts to 1601 and raises nothing. Such
a value then takes precedence over a usable ``TestDate``/``TestTime``. No RNS
recording is older than the device.
"""

_MIN_SNAP_TOLERANCE_S = 0.002
"""Floor on the millisecond-snapping tolerance, at ~4x the quantization error."""

_FALLBACK_START = datetime(1970, 1, 1, tzinfo=UTC)
"""Last-resort session start; pynwb requires a tz-aware instant."""


@dataclass(frozen=True, slots=True)
class SegmentSpan:
    """A contiguous run of samples and when it starts.

    ``stop_sample`` is exclusive. ``offset_s`` is seconds from the first sample of
    the recording, so span 0 always has ``offset_s == 0.0``.
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

    Return None when either field is absent, empty, or in no recognized format.
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

    A NeuroPace FILETIME wins when it is present, because it is an unambiguous UTC
    instant, while ``TestDate``/``TestTime`` are bare wall-clock that ``tz`` has to
    reinterpret. A ``[SampleTimes]`` value serves only when it is believable as
    Unix seconds. The source string returned here goes to the log and into the NWB
    session description.
    """
    for key in ("ecogtimestampasutc", "layoutfiletimestampasutc"):
        raw = layout.np_file_info.get(key)
        if not raw:
            continue
        stamp = _filetime_or_none(raw, key)
        if stamp is not None:
            return stamp, f"[NP_FileInfo] {key}"

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
    never matters. With no usable ``[SampleTimes]`` data, or with times that run
    backwards, the result is a single contiguous span.

    Boundaries within ``max(1.5 / rate, 0.002)`` seconds of where a gapless
    recording would put them are snapped to the exact value. Persyst rounds these
    times to milliseconds, an error of up to 500 us, which above ~1024 Hz exceeds
    the ``2 / rate`` gap threshold. Without snapping, a contiguous 2048 Hz
    recording reports false gaps and can emit non-monotonic timestamps.

    Stored times can be in order and still describe an overlap. At 250 Hz, 250
    samples fill one second, so an entry 0.5 s later covers samples that the
    previous segment also covers. An overlap makes timestamps decrease, exactly as
    times out of order do, so such a boundary moves forward to the end of the
    previous segment and the move is logged.

    The returned spans always tile ``[0, n_samples)`` in order, and their offsets
    never decrease. See ``_check_spans``.

    Raise ValueError if ``n_samples`` is not positive.
    """
    if n_samples <= 0:
        raise ValueError(f"recording has no samples: {n_samples!r}")

    usable = _usable_segments(segments, n_samples, rate)
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

    spans: list[SegmentSpan] = []
    snapped = 0
    overlapped = 0
    for seg, start, stop in zip(usable, starts, stops, strict=True):
        expected = (start - base_sample) / rate
        observed = seg.start_time_s - base_time
        if abs(observed - expected) <= tolerance:
            offset = expected
            snapped += observed != expected
        else:
            offset = observed

        if spans:
            earliest = spans[-1].offset_s + spans[-1].n_samples / rate
            if offset < earliest:
                # Always move the boundary so the invariant stays exact, but
                # report only an overlap larger than half a sample period: a sum
                # of float64 offsets can fall one ULP short of `expected`.
                overlapped += earliest - offset > 0.5 / rate
                offset = earliest
        spans.append(SegmentSpan(start, stop, offset))

    if snapped:
        logger.info(
            "snapped %d millisecond-quantised segment boundary/ies to exact times",
            snapped,
        )
    if overlapped:
        logger.warning(
            "%d [SampleTimes] entry/ies overlap the preceding segment; "
            "clamping them to where the previous segment ends",
            overlapped,
        )

    result = tuple(spans)
    _check_spans(result, n_samples, rate)
    return result


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


def timestamps_window(
    spans: tuple[SegmentSpan, ...], rate: float, start: int, stop: int
) -> npt.NDArray[np.float64]:
    """Timestamps for samples ``[start, stop)``, in seconds from session start.

    A window, rather than the whole recording, lets the writer stream the
    timestamps as it streams the samples. One float64 timestamp uses 8 bytes per
    sample. For a 4-channel int16 recording, the full array is as large as the
    ``.dat`` file, which cancels the benefit of the memory-mapped read.

    Offsets stay small because they are relative, so the float64 spacing stays
    much finer than the gap threshold even for a recording of several weeks.
    """
    parts = []
    for span in spans:
        lo = max(span.start_sample, start)
        hi = min(span.stop_sample, stop)
        if hi <= lo:
            continue
        offsets = np.arange(lo - span.start_sample, hi - span.start_sample)
        parts.append(span.offset_s + offsets / rate)

    if not parts:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(parts)


def timestamps_seconds(
    spans: tuple[SegmentSpan, ...], rate: float
) -> npt.NDArray[np.float64]:
    """Build one timestamp per sample, in seconds from the session start.

    The writer streams windows through ``timestamps_window`` instead. Use this
    only when you need the complete array.
    """
    if not spans:
        return np.empty(0, dtype=np.float64)
    return timestamps_window(spans, rate, 0, spans[-1].stop_sample)


def _filetime_or_none(raw: str, key: str) -> datetime | None:
    """Convert a FILETIME string, or None if it is unreadable or not credible.

    The conversion succeeds for any integer in range: a value of 0 gives
    1601-01-01 and raises nothing. The year check matters because such a value
    would otherwise take precedence over the later start-time sources.
    """
    try:
        stamp = filetime_to_datetime(int(raw))
    except (ValueError, OverflowError, OSError):
        logger.warning("unreadable FILETIME in %s: %r", key, raw)
        return None

    low, high = _PLAUSIBLE_FILETIME_YEARS
    if not low <= stamp.year <= high:
        logger.warning(
            "implausible FILETIME in %s: %r resolves to %s; ignoring it",
            key,
            raw,
            stamp.isoformat(),
        )
        return None
    return stamp


def _try_formats(value: str, formats: tuple[str, ...]) -> datetime | None:
    """Return the first successful ``strptime`` parse, or None if none match."""
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)  # noqa: DTZ007
        except ValueError:
            continue
    return None


def _check_spans(
    spans: tuple[SegmentSpan, ...], n_samples: int, rate: float
) -> None:
    """Check that spans tile ``[0, n_samples)`` and never move backwards.

    NWB requires timestamps that increase, which holds only if each span starts at
    or after the end of the previous span. Checking the spans costs one pass over
    the segments, and segment counts are always small; checking the timestamps
    themselves would cost one pass over every sample.

    Raise ValueError if a span breaks either rule. No reader can use a file that
    breaks them.
    """
    if spans[0].start_sample != 0 or spans[-1].stop_sample != n_samples:
        raise ValueError(
            f"segment spans do not tile [0, {n_samples}): "
            f"{spans[0].start_sample}..{spans[-1].stop_sample}"
        )
    for previous, current in zip(spans, spans[1:], strict=False):
        if current.start_sample != previous.stop_sample:
            raise ValueError(
                f"segment spans leave a hole at sample "
                f"{previous.stop_sample}..{current.start_sample}"
            )
        earliest = previous.offset_s + previous.n_samples / rate
        if current.offset_s < earliest:
            raise ValueError(
                f"segment starting at sample {current.start_sample} begins at "
                f"{current.offset_s} s, before the previous segment ends at "
                f"{earliest} s"
            )


def _usable_segments(
    segments: tuple[Segment, ...], n_samples: int, rate: float
) -> list[Segment]:
    """Sort, deduplicate, and drop entries that fall outside the data.

    Persyst may leave stale ``[SampleTimes]`` entries describing a longer original
    recording than the ``.dat`` actually holds.

    If the first entry that remains is not sample 0, this function adds a span for
    the samples before it. It calculates the time of that span backward from the
    entry at ``rate``, which assumes that those samples are contiguous with the
    entry. The entry's own time must not be used: two spans then have the same
    offset, and the timestamps decrease.
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
            "[SampleTimes] starts at sample %d, not 0; prepending a span "
            "extrapolated back to sample 0",
            usable[0].start_sample,
        )
        usable.insert(
            0,
            Segment(0, usable[0].start_time_s - usable[0].start_sample / rate),
        )

    return usable
