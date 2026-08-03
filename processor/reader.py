"""Reading Persyst sample data out of the ``.dat`` binary.

The ``.dat`` file has no header beyond ``HeaderLength``, holds little-endian
integers, and interleaves them by sample: every channel's value at time 0, then
every channel's value at time 1, and so on. A reshape to ``(samples, channels)``
needs no transpose, which is the orientation NWB uses.
"""

import logging
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import numpy.typing as npt

from processor.constants import MICROVOLTS_PER_VOLT, SampleDtype
from processor.layout import Comment, Layout, read_layout
from processor.timebase import (
    has_gaps,
    resolve_session_start,
    segment_spans,
    timestamps_seconds,
    timestamps_window,
)

logger = logging.getLogger(__name__)

_REF_SUFFIXES: tuple[str, ...] = ("-ref",)
"""Lower-cased channel-label suffixes ``strip_ref_suffix`` removes."""


def resolve_dat_path(lay_path: Path, dat_name: str) -> Path:
    r"""Locate the ``.dat`` file belonging to a ``.lay`` header.

    ``[FileInfo] File`` is often a Windows absolute path such as
    ``D:\\Archive\\...\\x.dat``, so the basename is taken by splitting on both
    separators rather than trusting the host's. Matching is case-insensitive
    because real files pair a lower-case ``File=`` with an upper-case ``.DAT`` on
    disk.

    If the named file is absent, the only permitted substitute is a sibling with
    the same stem as the ``.lay`` file. The reader decodes whatever file it gets
    at the channel count, the dtype, and the rate of this header, so an unrelated
    ``.dat`` file yields output that looks correct and holds the wrong recording.
    A directory that pairs the two wrongly has to be an error.

    Raise FileNotFoundError when nothing matches and ValueError when several
    siblings are equally plausible.
    """
    directory = lay_path.parent
    basename = dat_name.replace("\\", "/").rsplit("/", 1)[-1].strip()

    if basename:
        wanted = basename.lower()
        for candidate in sorted(directory.iterdir()):
            if candidate.is_file() and candidate.name.lower() == wanted:
                return candidate
        logger.warning(
            "lay file names %r but it is not present; falling back to a "
            "sibling that matches the lay stem",
            basename,
        )

    siblings = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".dat"
    )
    if not siblings:
        raise FileNotFoundError(f"no .dat file alongside {lay_path.name!r}")

    by_stem = [
        path for path in siblings if path.stem.lower() == lay_path.stem.lower()
    ]
    if len(by_stem) == 1:
        return by_stem[0]
    if not by_stem:
        raise FileNotFoundError(
            f"no .dat file for {lay_path.name!r}: the header names "
            f"{basename!r}, which is absent, and no sibling shares the "
            f"header's stem; found {[path.name for path in siblings]!r}"
        )
    raise ValueError(
        f"cannot choose a .dat for {lay_path.name!r}: "
        f"{[path.name for path in by_stem]!r}"
    )


class PersystReader:
    """Windowed access to a Persyst recording's samples and metadata.

    Presents one surface to the NWB writer whether the recording is contiguous or
    stitched from segments. The samples stay memory-mapped, so the whole ``.dat``
    never loads into memory.
    """

    def __init__(
        self,
        lay_path: Path,
        *,
        timezone: ZoneInfo,
        strip_ref_suffix: bool = False,
    ) -> None:
        """Open the recording described by the ``.lay`` file at ``lay_path``.

        Raise ValueError when the header and the ``.dat`` size disagree so badly
        that no whole sample can be read.
        """
        self.lay_path = lay_path
        self.layout: Layout = read_layout(lay_path)
        self.dat_path = resolve_dat_path(
            lay_path, self.layout.file_info.dat_name
        )
        self._strip_ref_suffix = strip_ref_suffix

        info = self.layout.file_info
        self.dtype: SampleDtype = info.dtype
        self.n_channels = self.layout.n_channels
        self.n_samples = self._compute_n_samples()

        self._data = np.memmap(
            self.dat_path,
            dtype=self.dtype,
            mode="r",
            offset=info.header_length,
            shape=(self.n_samples, self.n_channels),
        )

        self.session_start_time, self.session_start_source = (
            resolve_session_start(self.layout, timezone)
        )
        self.spans = segment_spans(
            self.layout.segments, self.n_samples, info.sampling_rate
        )
        logger.info(
            "%s: %d channels, %d samples @ %g Hz, %d segment(s), start %s (%s)",
            lay_path.name,
            self.n_channels,
            self.n_samples,
            self.sampling_rate,
            len(self.spans),
            self.session_start_time.isoformat(),
            self.session_start_source,
        )

    @property
    def sampling_rate(self) -> float:
        """Samples per second, shared by every channel."""
        return self.layout.file_info.sampling_rate

    @property
    def conversion(self) -> float:
        """Factor turning a stored count into volts.

        Persyst's calibration yields microvolts, and ``ElectricalSeries.unit`` is
        fixed to volts by the NWB schema, so the two combine here. Recording this
        alongside the raw counts, rather than pre-scaling the samples, is lossless
        and roughly halves the file against float64.
        """
        return self.layout.file_info.calibration / MICROVOLTS_PER_VOLT

    @property
    def channel_names(self) -> tuple[str, ...]:
        """Channel labels in column order.

        Case and inner spacing survive, so ``Fp1-Ref`` stays as written. The
        trailing reference suffix comes off only on request, because the label as
        recorded is the one clinicians recognize.
        """
        names = self.layout.channel_names
        if not self._strip_ref_suffix:
            return names
        return tuple(_strip_ref(name) for name in names)

    @property
    def comments(self) -> tuple[Comment, ...]:
        """Annotations parsed from the header."""
        return self.layout.comments

    @property
    def duration_s(self) -> float:
        """Seconds of samples the recording holds, ignoring gaps."""
        return self.n_samples / self.sampling_rate

    @property
    def recording_end_s(self) -> float:
        """Seconds from the session start to the last sample, gaps included.

        Annotations use this scale. ``duration_s`` counts only the samples. A gap
        in a segmented recording can be hours or days, so ``duration_s`` reports
        a shorter time than the recording spans.
        """
        last = self.spans[-1]
        return last.offset_s + last.n_samples / self.sampling_rate

    def has_gaps(self) -> bool:
        """Whether ``[SampleTimes]`` reports a real break between segments."""
        return has_gaps(self.spans, self.sampling_rate)

    def timestamps_seconds(self) -> npt.NDArray[np.float64]:
        """One timestamp per sample, in seconds from the session start.

        This method builds the array for the whole recording. The writer streams
        the timestamps with ``timestamps_window`` instead.
        """
        return timestamps_seconds(self.spans, self.sampling_rate)

    def timestamps_window(
        self, start: int, stop: int
    ) -> npt.NDArray[np.float64]:
        """Timestamps for samples ``[start, stop)``, seconds from session start."""
        return timestamps_window(self.spans, self.sampling_rate, start, stop)

    def read_window(
        self, start: int, stop: int
    ) -> npt.NDArray[np.signedinteger]:
        """Return samples ``[start, stop)`` for every channel, unscaled.

        The values are the raw integers from disk; scaling is expressed through
        ``conversion`` on the NWB dataset instead of being applied here.
        """
        return np.asarray(self._data[start:stop, :])

    def subject_fields(self) -> dict[str, str]:
        """Non-empty ``[Patient]`` identity fields, lower-cased keys.

        Empty values are dropped rather than written through, because every real
        fixture leaves most of this section blank and pynwb rejects an empty
        date of birth.
        """
        return {
            key: value.strip()
            for key, value in self.layout.patient.items()
            if value and value.strip()
        }

    def _compute_n_samples(self) -> int:
        """Derive the sample count from the ``.dat`` size.

        The header's own count is not trusted; file size is authoritative. Raise
        ValueError when not even one whole sample is present: an empty recording
        is a truncated or mispaired file, not something worth converting.
        """
        info = self.layout.file_info
        payload = self.dat_path.stat().st_size - info.header_length
        stride = self.n_channels * self.dtype.itemsize

        if payload <= 0:
            raise ValueError(
                f"{self.dat_path.name!r} holds no data after a "
                f"{info.header_length}-byte header"
            )

        n_samples, remainder = divmod(payload, stride)
        if remainder:
            logger.warning(
                "%s has %d trailing byte(s) beyond a whole sample; truncating",
                self.dat_path.name,
                remainder,
            )
        if n_samples == 0:
            raise ValueError(
                f"{self.dat_path.name!r} is smaller than one "
                f"{self.n_channels}-channel sample"
            )
        return int(n_samples)


def _strip_ref(name: str) -> str:
    """Remove a trailing reference suffix from a channel label."""
    lowered = name.lower()
    for suffix in _REF_SUFFIXES:
        if lowered.endswith(suffix):
            return name[: -len(suffix)]
    return name
