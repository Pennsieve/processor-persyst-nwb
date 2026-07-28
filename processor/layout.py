"""Parsing of the Persyst ``.lay`` header into plain values.

A ``.lay`` file looks like an INI file but is not one. ``configparser`` cannot
read it: it lower-cases option keys (destroying channel labels such as
``Fp1-Ref``), raises on the duplicate ``Annotations=`` keys that NeuroPace files
carry, treats ``:`` as a key/value delimiter (shredding free-text comments), and
chokes entirely on the comma-separated ``[Comments]`` body. So the scanner here
is hand-rolled and section-aware.

Nothing in this module touches the ``.dat`` file, so every header variant is
testable from a string.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from processor.constants import DATATYPE_TO_DTYPE, SampleDtype

logger = logging.getLogger(__name__)

_COMMENT_SECTIONS: frozenset[str] = frozenset({"comments"})
"""Sections whose bodies are comma-separated rows rather than key/value pairs."""

_COMMENT_FIELDS = 5
"""Fields in a ``[Comments]`` row: onset, duration, state, var_type, text."""

_NP_ANNOTATION_FIELDS = 4
"""Fields consumed from an ``[NP_Comments] Annotations=`` value."""

_MIN_SECTION_HEADER_LEN = 3
"""Shortest possible ``[x]`` section header, so ``[]`` is not mistaken for one."""


@dataclass(frozen=True, slots=True)
class FileInfo:
    """Decoded ``[FileInfo]`` section.

    ``dat_name`` is copied verbatim and may be a Windows absolute path, so
    resolving it to a real file is the reader's job. ``calibration`` converts raw
    counts to microvolts. ``waveform_count`` is recorded but not trusted. See
    ``Layout.n_channels``.
    """

    dat_name: str
    file_type: str
    sampling_rate: float
    header_length: int
    calibration: float
    waveform_count: int | None
    dtype: SampleDtype


@dataclass(frozen=True, slots=True)
class Segment:
    """One ``[SampleTimes]`` entry: where a contiguous run of samples begins.

    ``start_time_s`` is on an unspecified epoch that varies between files, so only
    differences between segments are meaningful.
    """

    start_sample: int
    start_time_s: float


@dataclass(frozen=True, slots=True)
class Comment:
    """One annotation, in seconds relative to the start of the recording.

    ``onset_s`` may be negative (events logged before recording began) or beyond
    the end of the data, and ``text`` may hold several kilobytes of XML.
    """

    onset_s: float
    duration_s: float
    text: str


@dataclass(frozen=True, slots=True)
class Layout:
    """Everything the ``.lay`` header states about a recording."""

    file_info: FileInfo
    patient: Mapping[str, str]
    np_file_info: Mapping[str, str]
    channel_names: tuple[str, ...]
    segments: tuple[Segment, ...]
    comments: tuple[Comment, ...]

    @property
    def n_channels(self) -> int:
        """Number of interleaved channels in the ``.dat`` file.

        Taken from ``[ChannelMap]`` rather than ``WaveformCount``, which real
        files get wrong: ``wave_sin.lay`` declares 4 waveforms for a genuinely
        2-channel recording.
        """
        return len(self.channel_names)


def read_layout(path: Path) -> Layout:
    """Parse the ``.lay`` file at the given path.

    Decodes with ``errors="replace"`` because patient fields written by the
    Windows application may hold cp1252 bytes, and relies on universal newlines
    since real files mix CRLF and LF.
    """
    return parse_layout(path.read_text(encoding="utf-8", errors="replace"))


def parse_layout(text: str) -> Layout:
    """Parse ``.lay`` text.

    Raise ValueError when ``[FileInfo]`` is missing or its required numeric
    fields cannot be read, when ``[ChannelMap]`` is empty, or when the channel
    indices are not exactly 1..n.
    """
    sections = _scan_sections(text)

    file_info = _build_file_info(_as_mapping(sections.get("fileinfo", [])))
    channel_names = _build_channel_names(sections.get("channelmap", []))
    segments = _build_segments(sections.get("sampletimes", []))
    comments = _build_comments(
        sections.get("comments", []), sections.get("np_comments", [])
    )

    waveform_count = file_info.waveform_count
    if waveform_count is not None and waveform_count != len(channel_names):
        logger.warning(
            "WaveformCount is %d but [ChannelMap] names %d channels; "
            "trusting [ChannelMap]",
            waveform_count,
            len(channel_names),
        )

    return Layout(
        file_info=file_info,
        patient=_as_mapping(sections.get("patient", [])),
        np_file_info=_as_mapping(sections.get("np_fileinfo", [])),
        channel_names=channel_names,
        segments=segments,
        comments=comments,
    )


def _scan_sections(text: str) -> dict[str, list[tuple[str, str]]]:
    """Split ``.lay`` text into ordered key/value pairs per lower-cased section.

    Pairs are kept as a list, not a dict, so duplicate keys survive. NeuroPace
    files repeat ``Annotations=`` once per annotation. Inside a comments section
    every non-blank line is stored whole under an empty key, because comment text
    contains ``=`` within XML attributes and must never be split on it.
    """
    sections: dict[str, list[tuple[str, str]]] = {}
    section = ""

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if (
            line.startswith("[")
            and line.endswith("]")
            and len(line) >= _MIN_SECTION_HEADER_LEN
        ):
            section = line[1:-1].strip().lower()
            sections.setdefault(section, [])
            continue

        entries = sections.setdefault(section, [])

        if section in _COMMENT_SECTIONS:
            entries.append(("", line))
            continue

        key, sep, value = line.partition("=")
        if not sep:
            logger.debug(
                "skipping line without '=' in section %r: %.60s", section, line
            )
            continue
        entries.append((key.strip(), value.strip()))

    return sections


def _as_mapping(entries: Sequence[tuple[str, str]]) -> Mapping[str, str]:
    """Collapse entries into a lower-cased-key mapping, last value winning."""
    return {key.lower(): value for key, value in entries if key}


def _build_file_info(fileinfo: Mapping[str, str]) -> FileInfo:
    """Assemble ``FileInfo``, raising ValueError on anything unusable."""
    if not fileinfo:
        raise ValueError(".lay file has no [FileInfo] section")

    file_type = fileinfo.get("filetype", "")

    return FileInfo(
        dat_name=fileinfo.get("file", ""),
        file_type=file_type,
        sampling_rate=_required_float(fileinfo, "samplingrate"),
        header_length=int(_optional_float(fileinfo, "headerlength") or 0.0),
        calibration=_required_float(fileinfo, "calibration"),
        waveform_count=_optional_int(fileinfo, "waveformcount"),
        dtype=_resolve_dtype(fileinfo.get("datatype"), file_type),
    )


def _resolve_dtype(datatype: str | None, file_type: str) -> SampleDtype:
    """Map ``DataType`` to a sample dtype.

    Fall back to ``FileType`` when ``DataType`` is absent, since
    ``32BitInterleaved`` says the same thing. Raise ValueError on a value that is
    present but unrecognised rather than guessing: the legacy processor treated
    everything except 7 as int16, which silently mis-decodes anything new.
    """
    if datatype is None or datatype == "":
        inferred = (
            np.dtype("<i4") if "32bit" in file_type.lower() else np.dtype("<i2")
        )
        logger.warning(
            "no DataType in [FileInfo]; inferring %s from FileType %r",
            inferred,
            file_type,
        )
        return inferred

    try:
        code = int(datatype)
    except ValueError as exc:
        raise ValueError(f"DataType is not an integer: {datatype!r}") from exc

    if code not in DATATYPE_TO_DTYPE:
        raise ValueError(
            f"unsupported DataType {code!r}; "
            f"expected one of {sorted(DATATYPE_TO_DTYPE)}"
        )
    return DATATYPE_TO_DTYPE[code]


def _build_channel_names(
    entries: Sequence[tuple[str, str]],
) -> tuple[str, ...]:
    """Extract channel labels from ``[ChannelMap]`` in file order.

    Labels keep their original case and inner spacing (``Fp1-Ref``,
    ``Lhip1 - Lhip2``, ``Sin 20Hz``). Raise ValueError when the section is empty
    or its indices are not exactly 1..n, because a sparse map leaves the
    interleave width ambiguous and file size alone cannot disambiguate it.
    """
    names = tuple(key.strip() for key, _ in entries if key.strip())
    if not names:
        raise ValueError(".lay file has no [ChannelMap] entries")

    indices = []
    for _, value in entries:
        try:
            indices.append(int(value))
        except ValueError:
            indices.append(-1)

    if sorted(indices) != list(range(1, len(names) + 1)):
        raise ValueError(
            f"[ChannelMap] indices are non-sequential: {indices!r}; "
            f"expected a permutation of 1..{len(names)}"
        )

    return names


def _build_segments(entries: Sequence[tuple[str, str]]) -> tuple[Segment, ...]:
    """Extract ``[SampleTimes]`` entries, skipping rows that do not parse."""
    segments = []
    for key, value in entries:
        try:
            segments.append(
                Segment(start_sample=int(key), start_time_s=float(value))
            )
        except ValueError:
            logger.warning(
                "skipping malformed [SampleTimes] row: %r=%r", key, value
            )
    return tuple(segments)


def _build_comments(
    comment_entries: Sequence[tuple[str, str]],
    np_entries: Sequence[tuple[str, str]],
) -> tuple[Comment, ...]:
    """Extract annotations from ``[Comments]`` and ``[NP_Comments]``."""
    comments = list(_parse_comment_rows(comment_entries))
    comments.extend(_parse_np_annotations(np_entries))
    return tuple(comments)


def _parse_comment_rows(
    entries: Sequence[tuple[str, str]],
) -> list[Comment]:
    """Parse ``onset,duration,state,var_type,text`` rows.

    Splits at most four times so commas inside the free text survive.
    """
    comments = []
    skipped = 0
    for _, line in entries:
        fields = line.split(",", _COMMENT_FIELDS - 1)
        if len(fields) < _COMMENT_FIELDS:
            skipped += 1
            continue
        try:
            onset = float(fields[0])
            duration = float(fields[1])
        except ValueError:
            skipped += 1
            continue
        comments.append(
            Comment(onset_s=onset, duration_s=duration, text=fields[4].strip())
        )

    if skipped:
        logger.warning("skipped %d malformed [Comments] row(s)", skipped)
    return comments


def _parse_np_annotations(
    entries: Sequence[tuple[str, str]],
) -> list[Comment]:
    """Parse NeuroPace ``Annotations=source,label,channel,onset,...`` values.

    These carry no duration, so the interval is a point in time.
    """
    comments = []
    for key, value in entries:
        if key.lower() != "annotations":
            continue
        fields = value.split(",")
        if len(fields) < _NP_ANNOTATION_FIELDS:
            logger.warning("skipping malformed NP annotation: %.60s", value)
            continue
        try:
            onset = float(fields[3])
        except ValueError:
            logger.warning(
                "skipping NP annotation with bad onset: %.60s", value
            )
            continue
        comments.append(
            Comment(onset_s=onset, duration_s=0.0, text=fields[1].strip())
        )
    return comments


def _required_float(fileinfo: Mapping[str, str], key: str) -> float:
    """Read a mandatory float from ``[FileInfo]``, raising ValueError if absent."""
    value = _optional_float(fileinfo, key)
    if value is None:
        raise ValueError(f"[FileInfo] is missing or has an unreadable {key!r}")
    return value


def _optional_float(fileinfo: Mapping[str, str], key: str) -> float | None:
    """Read an optional float from ``[FileInfo]``, returning None if unusable."""
    raw = fileinfo.get(key)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        logger.warning("[FileInfo] %s is not a number: %r", key, raw)
        return None


def _optional_int(fileinfo: Mapping[str, str], key: str) -> int | None:
    """Read an optional int from ``[FileInfo]``, returning None if unusable."""
    value = _optional_float(fileinfo, key)
    return None if value is None else int(value)
