"""Assembly of the NWB file.

The output follows the conventional shape for extracellular electrophysiology, so
any NWB reader can consume it: one ``ElectricalSeries`` in ``acquisition`` with
data shaped ``(samples, channels)``, an electrodes table with one row per channel
and a region covering all of them, a ``channel_name`` column carrying the
recording's own labels, a timezone-aware session start, and discontinuities
expressed as jumps in ``timestamps`` rather than a constant rate.
"""

import logging
import uuid
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from hdmf.backends.hdf5 import H5DataIO
from hdmf.common import DynamicTableRegion
from hdmf.data_utils import GenericDataChunkIterator
from pynwb import NWBHDF5IO, NWBFile
from pynwb.ecephys import ElectricalSeries
from pynwb.epoch import TimeIntervals
from pynwb.file import Subject

from processor.constants import (
    COMMENTS_TABLE_NAME,
    DEFAULT_COMPRESSION_LEVEL,
    DEVICE_NAME,
    ELECTRICAL_SERIES_NAME,
    ELECTRODE_GROUP_NAME,
    UNKNOWN_LOCATION,
)
from processor.layout import Comment
from processor.reader import PersystReader
from processor.sizing import chunk_samples
from processor.timebase import parse_test_datetime

logger = logging.getLogger(__name__)

_SEX_CODES: dict[str, str] = {
    "m": "M",
    "male": "M",
    "f": "F",
    "female": "F",
    "o": "O",
    "other": "O",
    "u": "U",
    "unknown": "U",
}
"""``[Patient] Sex`` values mapped to the four codes NWB documents."""


class _RawDataIterator(GenericDataChunkIterator):  # type: ignore[misc]
    # hdmf ships no type stubs, so the base class resolves to Any.
    """Streams raw samples so the whole recording never sits in memory."""

    def __init__(self, reader: PersystReader, samples_per_chunk: int) -> None:
        self._reader = reader
        self._shape = (reader.n_samples, reader.n_channels)
        shape = (samples_per_chunk, reader.n_channels)
        super().__init__(buffer_shape=shape, chunk_shape=shape)

    def _get_data(self, selection: tuple[slice, ...]) -> npt.NDArray[Any]:
        """Return the samples covered by ``selection``.

        hdmf invokes this with ``selection=`` as a keyword, so the parameter name
        is part of the contract.
        """
        rows = selection[0]
        return self._reader.read_window(rows.start, rows.stop)

    def _get_maxshape(self) -> tuple[int, int]:
        """Full dataset shape, samples first."""
        return self._shape

    def _get_dtype(self) -> np.dtype[Any]:
        """Return the Persyst integer dtype, which is stored without upcast."""
        return self._reader.dtype


class _TimestampIterator(GenericDataChunkIterator):  # type: ignore[misc]
    # hdmf ships no type stubs, so the base class resolves to Any.
    """Streams per-sample timestamps, so the full array stays out of memory.

    One float64 timestamp uses 8 bytes per sample. For a 4-channel int16
    recording, the full array is as large as the ``.dat`` file. Such an array
    cancels the benefit that ``_RawDataIterator`` gives.
    """

    def __init__(self, reader: PersystReader, samples_per_chunk: int) -> None:
        self._reader = reader
        self._shape = (reader.n_samples,)
        shape = (min(samples_per_chunk, reader.n_samples),)
        super().__init__(buffer_shape=shape, chunk_shape=shape)

    def _get_data(self, selection: tuple[slice, ...]) -> npt.NDArray[Any]:
        """Return the timestamps covered by ``selection``."""
        rows = selection[0]
        return self._reader.timestamps_window(rows.start, rows.stop)

    def _get_maxshape(self) -> tuple[int]:
        """Full dataset shape: one timestamp per sample."""
        return self._shape

    def _get_dtype(self) -> np.dtype[Any]:
        """NWB timestamps are float64 seconds."""
        return np.dtype("float64")


def write_nwb(
    reader: PersystReader,
    output_path: Path,
    *,
    compression_level: int = DEFAULT_COMPRESSION_LEVEL,
    target_chunk_bytes: int | None = None,
    write_comments: bool = True,
    write_subject_metadata: bool = True,
    identifier: str | None = None,
) -> Path:
    """Convert an open Persyst recording into an NWB file.

    ``identifier`` defaults to a random value; pass one to make output
    reproducible. Return the path written.

    ``output_path`` holds a complete file, or no file. HDF5 empties the target
    file when it opens it. A write to the target itself therefore leaves a corrupt
    ``.nwb`` file if the conversion fails before the end.
    """
    samples_per_chunk = _resolve_chunk(reader, target_chunk_bytes)
    nwb = _build_nwb_file(reader, identifier, write_subject_metadata)
    electrodes = _add_electrodes(nwb, reader)
    nwb.add_acquisition(
        _build_series(reader, electrodes, samples_per_chunk, compression_level)
    )

    if write_comments:
        _add_comments(nwb, reader)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_atomically(nwb, output_path)

    logger.info(
        "wrote %s (%.1f MiB)",
        output_path,
        output_path.stat().st_size / 2**20,
    )
    return output_path


def _write_atomically(nwb: NWBFile, output_path: Path) -> None:
    """Write ``nwb`` so that ``output_path`` holds only a complete file.

    The temporary file is in the same directory as the target, which keeps the
    rename atomic. Its name does not end in ``.nwb``. If the process stops
    without warning, the file that remains is therefore one that no reader, and
    no later stage of the pipeline, accepts as output. pynwb recommends the
    ``.nwb`` suffix, so this function hides that one warning for the temporary
    name.
    """
    partial = output_path.with_name(output_path.name + ".partial")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The file path provided",
                category=UserWarning,
                module="pynwb",
            )
            with NWBHDF5IO(str(partial), mode="w") as io:
                io.write(nwb)
        partial.replace(output_path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def _resolve_chunk(
    reader: PersystReader, target_chunk_bytes: int | None
) -> int:
    """Pick the samples-per-chunk for both datasets."""
    if target_chunk_bytes is None:
        return chunk_samples(
            reader.n_samples, reader.n_channels, reader.dtype.itemsize
        )
    return chunk_samples(
        reader.n_samples,
        reader.n_channels,
        reader.dtype.itemsize,
        target_chunk_bytes,
    )


def _build_nwb_file(
    reader: PersystReader, identifier: str | None, write_subject: bool
) -> NWBFile:
    """Create the NWBFile shell, with subject metadata if it is available.

    ``write_subject`` controls the block that comes from ``[Patient]``. That block
    holds the patient identifier and the date of birth. A date of birth is a HIPAA
    Safe Harbor identifier, so a workflow that publishes outside the PHI boundary
    must be able to omit the block.
    """
    nwb = NWBFile(
        session_description=(
            f"Persyst recording {reader.lay_path.name} "
            f"(start from {reader.session_start_source})"
        ),
        identifier=identifier or f"persyst_{uuid.uuid4().hex[:8]}",
        session_start_time=reader.session_start_time,
        session_id=reader.lay_path.stem,
    )
    if not write_subject:
        logger.info("WRITE_SUBJECT_METADATA is off; omitting the Subject block")
        return nwb

    subject = _build_subject(reader)
    if subject is not None:
        nwb.subject = subject
    return nwb


def _build_subject(reader: PersystReader) -> Subject | None:
    """Build a Subject from ``[Patient]``, or None when nothing usable is there.

    Only non-empty fields are passed through; every real fixture leaves most of
    the section blank, and an empty date of birth fails validation outright.
    """
    fields = reader.subject_fields()
    kwargs: dict[str, Any] = {}

    if "id" in fields:
        kwargs["subject_id"] = fields["id"]

    sex = _SEX_CODES.get(fields.get("sex", "").lower())
    if sex is not None:
        kwargs["sex"] = sex

    birth = parse_test_datetime(
        fields.get("birthdate"),
        "00:00:00",
        reader.session_start_time.tzinfo,  # type: ignore[arg-type]
    )
    if birth is not None and birth >= reader.session_start_time:
        # Persyst writes a two-digit year, and strptime reads 68 or less as 20xx.
        # A patient born before 1969 therefore gets a date 100 years late:
        # 01/02/40 gives 2040. A date of birth at or after the recording is not
        # possible, so this test finds such a date without a fixed cutoff year.
        logger.warning(
            "[Patient] BirthDate %r reads as %s, not before the recording "
            "start %s; omitting date_of_birth",
            fields.get("birthdate"),
            birth.date().isoformat(),
            reader.session_start_time.date().isoformat(),
        )
        birth = None
    if birth is not None:
        kwargs["date_of_birth"] = birth

    if not kwargs:
        return None
    return Subject(**kwargs)


def _add_electrodes(nwb: NWBFile, reader: PersystReader) -> DynamicTableRegion:
    """Populate the electrodes table and return a region covering all of it.

    Persyst records one flat list of channels with no montage, so the table has
    exactly one row per data column and the region spans all of them; readers
    generally pair data columns with table rows positionally.

    ``location`` and ``group`` are the only electrode fields pynwb requires. The
    optional coordinate and impedance columns are omitted rather than filled with
    placeholder zeros, which would imply the recording located its electrodes.
    """
    device = nwb.create_device(
        name=DEVICE_NAME, description="Persyst EEG recording system"
    )
    group = nwb.create_electrode_group(
        name=ELECTRODE_GROUP_NAME,
        description="Channels listed in the Persyst [ChannelMap]",
        location=UNKNOWN_LOCATION,
        device=device,
    )
    nwb.add_electrode_column(
        name="channel_name",
        description="Channel label from the Persyst [ChannelMap]",
    )
    for name in reader.channel_names:
        nwb.add_electrode(
            location=UNKNOWN_LOCATION, group=group, channel_name=name
        )
    return nwb.create_electrode_table_region(
        region=list(range(reader.n_channels)),
        description="All Persyst channels",
    )


def _build_series(
    reader: PersystReader,
    electrodes: DynamicTableRegion,
    samples_per_chunk: int,
    compression_level: int,
) -> ElectricalSeries:
    """Build the sole ElectricalSeries.

    A gapless recording gets ``rate`` and ``starting_time``, which is both smaller
    and the stronger statement: it asserts uniform sampling. A gapped one gets
    explicit ``timestamps``, since a rate cannot express a discontinuity. pynwb
    rejects setting both.
    """
    data = H5DataIO(
        data=_RawDataIterator(reader, samples_per_chunk),
        **_compression(compression_level),
    )
    shared: dict[str, Any] = {
        "name": ELECTRICAL_SERIES_NAME,
        "description": "Persyst EEG samples, stored as recorded counts",
        "data": data,
        "electrodes": electrodes,
        "conversion": reader.conversion,
        "offset": 0.0,
    }

    if not reader.has_gaps():
        logger.info(
            "recording is contiguous; writing rate %g Hz", reader.sampling_rate
        )
        return ElectricalSeries(
            **shared, rate=reader.sampling_rate, starting_time=0.0
        )

    logger.info(
        "recording has %d segment(s) with gaps; writing %d explicit timestamps",
        len(reader.spans),
        reader.n_samples,
    )
    return ElectricalSeries(
        **shared,
        timestamps=H5DataIO(
            data=_TimestampIterator(reader, samples_per_chunk),
            **_compression(compression_level),
        ),
    )


def _compression(level: int) -> dict[str, Any]:
    """Gzip settings for a dataset, or nothing when compression is disabled.

    The shuffle filter is worth more than a higher gzip level on integer EEG:
    byte-transposing before deflate takes int16 samples to roughly 0.56 of raw
    against 0.79 for gzip alone.
    """
    if level <= 0:
        return {}
    return {"compression": "gzip", "compression_opts": level, "shuffle": True}


def _add_comments(nwb: NWBFile, reader: PersystReader) -> None:
    """Record ``[Comments]`` annotations as a TimeIntervals table.

    Skipped entirely when there are no comments: a table carrying a custom column
    and zero rows cannot resolve that column's dtype and fails at write time.
    Onsets outside the recording are kept rather than dropped -- they are real
    provenance, and Persyst headers legitimately describe events before the first
    sample or beyond a clipped ``.dat`` -- but they are counted in the log.
    """
    comments = reader.comments
    if not comments:
        logger.info(
            "no annotations in the lay header; omitting %s", COMMENTS_TABLE_NAME
        )
        return

    table = TimeIntervals(
        name=COMMENTS_TABLE_NAME,
        description="Annotations from the Persyst [Comments] section",
    )
    table.add_column(name="label", description="Annotation text as recorded")

    for comment in comments:
        table.add_interval(
            start_time=float(comment.onset_s),
            stop_time=float(comment.onset_s + comment.duration_s),
            label=comment.text,
        )

    outside = _count_outside(comments, reader.recording_end_s)
    if outside:
        logger.warning(
            "%d of %d annotation(s) fall outside the %.1f s the recording "
            "spans; keeping them",
            outside,
            len(comments),
            reader.recording_end_s,
        )

    nwb.add_time_intervals(table)
    logger.info(
        "wrote %d annotation(s) to %s", len(comments), COMMENTS_TABLE_NAME
    )


def _count_outside(comments: tuple[Comment, ...], end_s: float) -> int:
    """Count annotations that start before 0 or after the last sample.

    ``end_s`` must include the gaps. A segmented recording spans much more time
    than its sample count gives. A count against the samples alone therefore
    reports annotations as outside the recording when they are inside it.
    """
    return sum(1 for c in comments if c.onset_s < 0.0 or c.onset_s > end_s)
