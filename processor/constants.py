"""Format constants for Persyst decoding and NWB emission."""

from collections.abc import Mapping
from typing import Any, Final

import numpy as np

type SampleDtype = np.dtype[np.signedinteger[Any]]
"""Dtype of one stored Persyst sample: a little-endian signed integer."""

DATATYPE_TO_DTYPE: Final[Mapping[int, SampleDtype]] = {
    0: np.dtype("<i2"),
    7: np.dtype("<i4"),
}
"""Persyst ``[FileInfo] DataType`` to sample dtype; Persyst is a Windows format."""

MICROVOLTS_PER_VOLT: Final = 1e6
"""Divisor turning the Persyst calibration (counts per uV) into volts per count."""

FILETIME_EPOCH_OFFSET_S: Final = 11_644_473_600
"""Seconds between the Windows FILETIME epoch (1601-01-01) and the Unix epoch."""

FILETIME_TICKS_PER_SECOND: Final = 10_000_000
"""FILETIME resolution: 100 ns ticks per second."""

GAP_THRESHOLD_PERIODS: Final = 2
"""Sample periods a timestamp jump must exceed to count as a discontinuity.

Two periods is the conventional tolerance for electrophysiology timeseries: it
admits one period of ordinary jitter while still catching a genuinely dropped
sample.
"""

TARGET_CHUNK_BYTES: Final = 4 * 2**20
"""Byte budget for one HDF5 chunk (~4 MiB)."""

MAX_CHUNK_SAMPLES: Final = 2**17
"""Cap on samples per chunk, whatever the byte budget allows.

Bounds how much a reader must decompress to reach a single sample when a
recording has few channels, where the byte budget alone would permit a chunk
spanning millions of samples.
"""

DEFAULT_COMPRESSION_LEVEL: Final = 4
"""gzip level for sample and timestamp datasets.

With the shuffle filter enabled, level 4 reaches ~0.56 of raw on int16 EEG and
level 9 only reaches ~0.55, so the extra CPU buys nothing.
"""

MAX_COMPRESSION_LEVEL: Final = 9
"""Highest gzip level h5py accepts; 0 disables compression entirely."""

MIN_SEGMENTS_FOR_SPANS: Final = 2
"""Usable ``[SampleTimes]`` entries needed before segmentation is meaningful."""

ELECTRICAL_SERIES_NAME: Final = "ElectricalSeries"
"""Name of the sole acquisition holding the samples."""

COMMENTS_TABLE_NAME: Final = "persyst_comments"
"""Name of the ``TimeIntervals`` table holding ``[Comments]`` annotations."""

DEVICE_NAME: Final = "Persyst"
"""NWB Device name."""

ELECTRODE_GROUP_NAME: Final = "PersystElectrodes"
"""NWB ElectrodeGroup name; pynwb copies it into the required ``group_name``."""

UNKNOWN_LOCATION: Final = "unknown"
"""Placeholder for the required electrode/group location Persyst never records."""
