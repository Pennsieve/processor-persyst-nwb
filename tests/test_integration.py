"""End-to-end conversion checked through a plain NWB reader.

``NwbTimeseriesReader`` below is an ordinary consumer of an NWB
``ElectricalSeries``: it pairs data columns with electrode rows, derives a rate
from either ``rate`` or ``timestamps``, splits on discontinuities, and scales to
microvolts via ``conversion`` and ``unit``. Nothing in it is specific to any one
pipeline -- it is the reading half of the format contract, written out so a change
that would break a real consumer fails here first.

This also happens to mirror how Pennsieve's ``processor-post-timeseries`` reads
the file, which is why the segment-splitting threshold matches.
"""

from zoneinfo import ZoneInfo

import numpy as np
import pytest
from pynwb import NWBHDF5IO
from pynwb.ecephys import ElectricalSeries

from processor.main import main
from tests.conftest import ramp_samples

UTC_ZONE = ZoneInfo("UTC")

UNIT_TO_UV = {
    "volts": 1e6,
    "v": 1e6,
    "millivolts": 1e3,
    "mv": 1e3,
    "microvolts": 1,
    "uv": 1,
}


def infer_sampling_rate(timestamps):
    """Derive a rate from the median spacing of the first few timestamps."""
    return 1 / np.median(np.diff(timestamps[:10]))


class NwbTimeseriesReader:
    """A plain reader of an NWB ElectricalSeries."""

    def __init__(self, series, session_start_time):
        self.series = series
        self.session_start_time_secs = session_start_time.timestamp()
        self.num_samples, self.num_channels = series.data.shape

        assert self.num_samples > 0, "Electrical series has no sample data"
        assert len(series.electrodes.table) == self.num_channels, (
            "Electrode channels do not align with data shape"
        )

        if self.has_explicit_timestamps:
            assert self.num_samples == len(series.timestamps), (
                "Differing number of sample and timestamp value"
            )

        self.sampling_rate = self._compute_sampling_rate()

    @property
    def has_explicit_timestamps(self):
        return self.series.timestamps is not None

    def _compute_sampling_rate(self):
        if self.series.rate is None and not self.has_explicit_timestamps:
            raise AssertionError(
                "electrical series has no defined sampling rate or timestamp values"
            )
        if self.series.rate:
            return self.series.rate
        sample = self.get_timestamps(0, min(10000, self.num_samples))
        return round(infer_sampling_rate(sample))

    def get_timestamps(self, start, end):
        if self.has_explicit_timestamps:
            timestamps = np.array(self.series.timestamps[start:end])
        else:
            timestamps = np.linspace(
                start / self.sampling_rate,
                end / self.sampling_rate,
                end - start,
                endpoint=False,
            )
        return timestamps + self.session_start_time_secs

    def channel_names(self):
        table = self.series.electrodes.table
        return list(table["channel_name"][:])

    def contiguous_chunks(self):
        if not self.has_explicit_timestamps:
            yield 0, self.num_samples
            return
        threshold = (1.0 / self.sampling_rate) * 2
        timestamps = self.get_timestamps(0, self.num_samples)
        boundaries = [0]
        gaps = np.where(np.diff(timestamps) > threshold)[0]
        boundaries.extend(int(index) + 1 for index in gaps)
        boundaries.append(self.num_samples)
        for index in range(len(boundaries) - 1):
            yield boundaries[index], boundaries[index + 1]

    def get_chunk(self, start=None, end=None):
        """Return per-channel arrays scaled to microvolts."""
        all_data = self.series.data[start:end, :]
        scaled = all_data * self.series.conversion + self.series.offset
        unit = getattr(self.series, "unit", "volts").lower()
        if unit not in UNIT_TO_UV:
            raise ValueError(f"Unknown unit '{unit}'")
        scaled = scaled * UNIT_TO_UV[unit]
        return [scaled[:, i] for i in range(self.num_channels)]


def convert_and_open(tmp_path, persyst_pair, **pair_kwargs):
    """Run the conversion and return the output opened through the reader."""
    persyst_pair(tmp_path, stem="rec", **pair_kwargs)
    out_dir = tmp_path / "out"
    env = {
        "INPUT_DIR": str(tmp_path),
        "OUTPUT_DIR": str(out_dir),
        "PERSYST_TIMEZONE": "UTC",
    }
    assert main([], env) == 0

    produced = [p for p in out_dir.iterdir() if p.suffix.lower() == ".nwb"]
    assert len(produced) == 1

    io = NWBHDF5IO(str(produced[0]), mode="r")
    nwb = io.read()
    series = [
        acq
        for acq in nwb.acquisition.values()
        if isinstance(acq, ElectricalSeries)
    ]
    assert len(series) == 1
    return io, NwbTimeseriesReader(series[0], nwb.session_start_time)


def test_contiguous_recording_reads_as_one_chunk(tmp_path, persyst_pair):
    io, reader = convert_and_open(
        tmp_path, persyst_pair, n_samples=1000, rate=250.0, segments=None
    )
    assert reader.num_samples == 1000
    assert reader.num_channels == 4
    assert reader.sampling_rate == 250.0
    assert list(reader.contiguous_chunks()) == [(0, 1000)]
    io.close()


def test_gapped_recording_splits_into_segments(tmp_path, persyst_pair):
    # 4 segments of 250 samples, each starting 500 s apart -- 3 real gaps.
    segments = [(0, 0.0), (250, 500.0), (500, 1000.0), (750, 1500.0)]
    io, reader = convert_and_open(
        tmp_path, persyst_pair, n_samples=1000, rate=250.0, segments=segments
    )
    assert reader.sampling_rate == 250
    assert list(reader.contiguous_chunks()) == [
        (0, 250),
        (250, 500),
        (500, 750),
        (750, 1000),
    ]
    io.close()


def test_channel_labels_survive_the_round_trip(tmp_path, persyst_pair):
    channels = ("Fp1-Ref", "Lhip1 - Lhip2", "Sin 20Hz", "Ch.1")
    io, reader = convert_and_open(tmp_path, persyst_pair, channels=channels)
    assert reader.channel_names() == list(channels)
    io.close()


def test_scaled_chunk_is_microvolts(tmp_path, persyst_pair):
    calibration = 0.2
    io, reader = convert_and_open(
        tmp_path, persyst_pair, n_samples=200, calibration=calibration
    )
    raw = ramp_samples(200, 4, np.dtype("<i2")).astype(np.float64)
    for index, channel in enumerate(reader.get_chunk()):
        np.testing.assert_allclose(
            channel, raw[:, index] * calibration, rtol=1e-12
        )
    io.close()


def test_absolute_time_of_first_sample(tmp_path, persyst_pair):
    io, reader = convert_and_open(
        tmp_path,
        persyst_pair,
        n_samples=500,
        rate=250.0,
        patient={"TestDate": "02/06/2000", "TestTime": "04:59:59.964000"},
    )
    first = reader.get_timestamps(0, 1)[0]
    assert first == pytest.approx(949813199.964)
    io.close()


def test_sine_waves_reconstruct_through_the_pipeline(tmp_path, persyst_pair):
    """The legacy harness checked 400*sin(2*pi*f*t); so does this."""
    rate, n = 800.0, 12000
    t = np.arange(n) / rate
    expected = np.column_stack(
        [400 * np.sin(2 * np.pi * 20 * t), 400 * np.sin(2 * np.pi * 10 * t)]
    )

    io, reader = convert_and_open(
        tmp_path,
        persyst_pair,
        data=expected.astype("<i2"),
        rate=rate,
        calibration=1.0,
        channels=("Sin 20Hz", "Sin 10Hz"),
    )
    assert reader.channel_names() == ["Sin 20Hz", "Sin 10Hz"]
    for index, channel in enumerate(reader.get_chunk()):
        np.testing.assert_allclose(
            channel, expected[:, index], rtol=0.01, atol=1.0
        )
    io.close()


def test_neuropace_style_recording_converts(tmp_path, persyst_pair):
    """A file with no [Patient] and a Windows File= must still convert.

    This is the shape MNE cannot open at all: it resolves the Windows path
    against the wrong directory and crashes on the missing [Patient] section.
    """
    io, reader = convert_and_open(
        tmp_path,
        persyst_pair,
        n_samples=7587,
        rate=250.0,
        calibration=0.3,
        dat_suffix=".DAT",
        dat_name=r"D:\Archive\VDhServer2\191036\rec.dat",
        patient=None,
        np_file_info={"ECoGTimeStampAsUTC": "130998625655000000"},
        np_comments=[
            "DEVICE,PROG_MARKER_MAGNET_APPLIED,CHANNEL_0_0,20.132,0,0"
        ],
    )
    assert reader.num_samples == 7587
    # 2016-02-13T18:42:45.5Z, recovered from the FILETIME.
    assert reader.get_timestamps(0, 1)[0] == pytest.approx(1455388965.5)
    io.close()
