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


def write_nwb(
    reader: PersystReader,
    output_path: Path,
    *,
    compression_level: int = DEFAULT_COMPRESSION_LEVEL,
    target_chunk_bytes: int | None = None,
    write_comments: bool = True,
    identifier: str | None = None,
) -> Path:
    """Convert an open Persyst recording into an NWB file.

    ``identifier`` defaults to a random value; pass one to make output
    reproducible. Return the path written.
    """
    samples_per_chunk = _resolve_chunk(reader, target_chunk_bytes)
    nwb = _build_nwb_file(reader, identifier)
    electrodes = _add_electrodes(nwb, reader)
    nwb.add_acquisition(
        _build_series(reader, electrodes, samples_per_chunk, compression_level)
    )

    if write_comments:
        _add_comments(nwb, reader)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with NWBHDF5IO(str(output_path), mode="w") as io:
        io.write(nwb)

    logger.info(
        "wrote %s (%.1f MiB)",
        output_path,
        output_path.stat().st_size / 2**20,
    )
    return output_path


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


def _build_nwb_file(reader: PersystReader, identifier: str | None) -> NWBFile:
    """Create the NWBFile shell, including subject metadata when available."""
    nwb = NWBFile(
        session_description=(
            f"Persyst recording {reader.lay_path.name} "
            f"(start from {reader.session_start_source})"
        ),
        identifier=identifier or f"persyst_{uuid.uuid4().hex[:8]}",
        session_start_time=reader.session_start_time,
        session_id=reader.lay_path.stem,
    )
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

    timestamps = reader.timestamps_seconds()
    logger.info(
        "recording has %d segment(s) with gaps; writing %d explicit timestamps",
        len(reader.spans),
        timestamps.size,
    )
    return ElectricalSeries(
        **shared,
        timestamps=H5DataIO(
            data=timestamps,
            chunks=(min(samples_per_chunk, timestamps.size),),
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

    outside = _count_outside(comments, reader.duration_s)
    if outside:
        logger.warning(
            "%d of %d annotation(s) fall outside the %.1f s of samples; keeping them",
            outside,
            len(comments),
            reader.duration_s,
        )

    nwb.add_time_intervals(table)
    logger.info(
        "wrote %d annotation(s) to %s", len(comments), COMMENTS_TABLE_NAME
    )


def _count_outside(comments: tuple[Comment, ...], duration_s: float) -> int:
    """Count annotations starting before 0 or after the last sample."""
    return sum(1 for c in comments if c.onset_s < 0.0 or c.onset_s > duration_s)
