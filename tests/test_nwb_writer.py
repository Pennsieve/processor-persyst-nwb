"""Tests asserting the NWB output is well formed and self-consistent.

These cover the properties an NWB reader relies on: data shaped
``(samples, channels)``, an electrodes table aligned with the data columns, either
a rate or per-sample timestamps, labels on a ``channel_name`` column, a tz-aware
session start, and a ``conversion`` that recovers the recorded microvolts.
"""

from zoneinfo import ZoneInfo

import h5py
import numpy as np
import pytest
from pynwb import NWBHDF5IO
from pynwb.ecephys import ElectricalSeries

from processor.nwb_writer import write_nwb
from processor.reader import PersystReader

UTC_ZONE = ZoneInfo("UTC")
GAP_SEGMENTS = [(0, 0.0), (250, 500.0)]


def convert(
    tmp_path, persyst_pair, *, name="out.nwb", writer=None, **pair_kwargs
):
    """Write a synthetic recording to NWB and return the output path."""
    lay, _ = persyst_pair(tmp_path, **pair_kwargs)
    reader = PersystReader(lay, timezone=UTC_ZONE)
    out = tmp_path / name
    write_nwb(reader, out, identifier="persyst_test", **(writer or {}))
    return out


def read_series(path):
    """Open the file and return (io, nwb, the sole ElectricalSeries)."""
    io = NWBHDF5IO(str(path), mode="r")
    nwb = io.read()
    series = [
        acq
        for acq in nwb.acquisition.values()
        if isinstance(acq, ElectricalSeries)
    ]
    assert len(series) == 1, (
        "exactly one series keeps channel indices unambiguous"
    )
    return io, nwb, series[0]


def test_exactly_one_electrical_series(tmp_path, persyst_pair):
    out = convert(tmp_path, persyst_pair)
    io, nwb, _ = read_series(out)
    assert list(nwb.acquisition) == ["ElectricalSeries"]
    io.close()


def test_data_is_samples_by_channels(tmp_path, persyst_pair):
    # Time-major: readers unpack `n_samples, n_channels = data.shape`.
    out = convert(
        tmp_path, persyst_pair, n_samples=300, channels=("A", "B", "C")
    )
    io, _, series = read_series(out)
    assert series.data.shape == (300, 3)
    io.close()


@pytest.mark.parametrize(
    ("datatype", "expected"), [(0, np.int16), (7, np.int32)]
)
def test_stored_dtype_is_not_upcast(tmp_path, persyst_pair, datatype, expected):
    # Raw counts plus a conversion factor is lossless and about half the size of
    # pre-scaled float64.
    out = convert(tmp_path, persyst_pair, datatype=datatype)
    with h5py.File(out) as f:
        assert f["acquisition/ElectricalSeries/data"].dtype == expected


def test_electrode_table_has_one_row_per_channel(tmp_path, persyst_pair):
    # The whole table, not just the region, must align with the data columns.
    channels = ("Fp1-Ref", "Fp2-Ref", "C3-Ref")
    out = convert(tmp_path, persyst_pair, channels=channels)
    io, _, series = read_series(out)
    assert len(series.electrodes.table) == 3
    assert list(series.electrodes.data) == [0, 1, 2]
    io.close()


def test_channel_name_column_preserves_labels_in_order(tmp_path, persyst_pair):
    channels = ("Fp1-Ref", "Lhip1 - Lhip2", "Sin 20Hz", "Ch.1")
    out = convert(tmp_path, persyst_pair, channels=channels)
    io, nwb, _ = read_series(out)
    assert list(nwb.electrodes["channel_name"][:]) == list(channels)
    io.close()


def test_group_name_column_present(tmp_path, persyst_pair):
    # pynwb derives group_name from the electrode group; readers expect it.
    out = convert(tmp_path, persyst_pair)
    io, nwb, _ = read_series(out)
    assert set(nwb.electrodes["group_name"][:]) == {"PersystElectrodes"}
    io.close()


def test_unit_is_volts(tmp_path, persyst_pair):
    # Fixed to volts by the NWB schema for ElectricalSeries.
    out = convert(tmp_path, persyst_pair)
    io, _, series = read_series(out)
    assert series.unit == "volts"
    io.close()


def test_conversion_is_calibration_in_volts(tmp_path, persyst_pair):
    out = convert(tmp_path, persyst_pair, calibration=0.2)
    io, _, series = read_series(out)
    assert series.conversion == pytest.approx(0.2e-6)
    assert series.offset == 0.0
    io.close()


def test_conversion_stored_as_float64(tmp_path, persyst_pair):
    # The schema types conversion as float32; hdmf upcasts a float64 value. If
    # that ever regresses, the gain picks up ~6e-8 of relative error.
    out = convert(tmp_path, persyst_pair, calibration=0.2)
    with h5py.File(out) as f:
        assert f["acquisition/ElectricalSeries/data"].attrs[
            "conversion"
        ].dtype == (np.float64)


def test_scaled_values_match_legacy_microvolts(tmp_path, persyst_pair):
    """data * conversion * 1e6 must equal the legacy raw * Calibration."""
    calibration = 0.2
    rng = np.random.default_rng(0)
    raw = rng.integers(-3000, 3000, size=(400, 4)).astype("<i2")
    out = convert(tmp_path, persyst_pair, data=raw, calibration=calibration)

    io, _, series = read_series(out)
    # The NWB-specified scaling: data * conversion + offset, volts -> uV.
    scaled = np.asarray(series.data[:, :]) * series.conversion + series.offset
    np.testing.assert_allclose(
        scaled * 1e6, raw.astype(np.float64) * calibration, rtol=1e-12
    )
    io.close()


def test_contiguous_recording_uses_rate(tmp_path, persyst_pair):
    out = convert(
        tmp_path, persyst_pair, n_samples=500, rate=250.0, segments=None
    )
    io, _, series = read_series(out)
    assert series.rate == 250.0
    assert series.starting_time == 0.0
    assert series.timestamps is None
    io.close()


def test_gapped_recording_uses_timestamps(tmp_path, persyst_pair):
    # A constant rate cannot express a discontinuity; timestamps can.
    out = convert(
        tmp_path, persyst_pair, n_samples=500, rate=250.0, segments=GAP_SEGMENTS
    )
    io, _, series = read_series(out)
    assert series.rate is None
    assert series.timestamps is not None
    assert len(series.timestamps) == 500
    io.close()


def test_timestamps_are_monotonic_and_reveal_one_gap(tmp_path, persyst_pair):
    out = convert(
        tmp_path, persyst_pair, n_samples=500, rate=250.0, segments=GAP_SEGMENTS
    )
    io, _, series = read_series(out)
    ts = np.asarray(series.timestamps[:])
    diffs = np.diff(ts)
    assert np.all(diffs > 0)
    assert int((diffs > 2 / 250.0).sum()) == 1
    io.close()


def test_session_start_time_is_timezone_aware(tmp_path, persyst_pair):
    out = convert(
        tmp_path,
        persyst_pair,
        patient={"TestDate": "02/06/2000", "TestTime": "04:59:59.964000"},
    )
    io, nwb, _ = read_series(out)
    assert nwb.session_start_time.tzinfo is not None
    assert nwb.session_start_time.timestamp() == pytest.approx(949813199.964)
    io.close()


def test_compression_and_chunking_applied(tmp_path, persyst_pair):
    out = convert(tmp_path, persyst_pair, n_samples=5000)
    with h5py.File(out) as f:
        data = f["acquisition/ElectricalSeries/data"]
        assert data.compression == "gzip"
        assert data.compression_opts == 4
        assert data.shuffle is True
        assert data.chunks[1] == 4


def test_compression_disabled_at_level_zero(tmp_path, persyst_pair):
    out = convert(tmp_path, persyst_pair, writer={"compression_level": 0})
    with h5py.File(out) as f:
        assert f["acquisition/ElectricalSeries/data"].compression is None


def test_chunk_clamped_to_short_recording(tmp_path, persyst_pair):
    # A chunk longer than the dataset trips an assertion inside hdmf.
    out = convert(tmp_path, persyst_pair, n_samples=64)
    with h5py.File(out) as f:
        assert f["acquisition/ElectricalSeries/data"].chunks[0] == 64


def test_timestamps_chunked_and_compressed(tmp_path, persyst_pair):
    out = convert(
        tmp_path, persyst_pair, n_samples=500, rate=250.0, segments=GAP_SEGMENTS
    )
    with h5py.File(out) as f:
        ts = f["acquisition/ElectricalSeries/timestamps"]
        assert ts.dtype == np.float64
        assert ts.compression == "gzip"
        assert ts.chunks is not None


def test_timestamps_are_written_in_chunks_not_all_at_once(
    tmp_path, persyst_pair, monkeypatch
):
    """The writer must stream the timestamps, as it streams the samples.

    One float64 timestamp uses 8 bytes per sample. For a 4-channel int16
    recording, the full array is as large as the .dat file. This test records the
    largest window that the writer requests, which is a repeatable measure of the
    memory that the writer needs.
    """
    requested = []
    original = PersystReader.timestamps_window

    def spy(self, start, stop):
        requested.append(stop - start)
        return original(self, start, stop)

    monkeypatch.setattr(PersystReader, "timestamps_window", spy)
    out = convert(
        tmp_path,
        persyst_pair,
        n_samples=5000,
        rate=250.0,
        segments=GAP_SEGMENTS,
        writer={"target_chunk_bytes": 4096},
    )

    assert requested, "the writer never asked for a timestamp window"
    # 4096 bytes / (4 channels x 2 bytes) = 512 samples per chunk.
    assert max(requested) == 512
    with h5py.File(out) as f:
        ts = f["acquisition/ElectricalSeries/timestamps"]
        assert ts.shape == (5000,)
        assert ts.chunks == (512,)


def test_streamed_timestamps_match_a_single_pass(tmp_path, persyst_pair):
    """A write in chunks must give the same array as one pass."""
    lay, _ = persyst_pair(
        tmp_path, n_samples=5000, rate=250.0, segments=GAP_SEGMENTS
    )
    reader = PersystReader(lay, timezone=UTC_ZONE)
    out = tmp_path / "out.nwb"
    write_nwb(reader, out, identifier="persyst_test", target_chunk_bytes=4096)

    io, _, series = read_series(out)
    np.testing.assert_array_equal(
        np.asarray(series.timestamps[:]), reader.timestamps_seconds()
    )
    io.close()


def test_comments_written_as_time_intervals(tmp_path, persyst_pair):
    comments = [
        (16479.035, 1579.058, "Impedance Test On"),
        (11029.92, 0.0, "Type1: bolusing, then EKG"),
    ]
    out = convert(tmp_path, persyst_pair, n_samples=500_000, comments=comments)
    io, nwb, _ = read_series(out)
    table = nwb.intervals["persyst_comments"]
    assert list(table["start_time"][:]) == [16479.035, 11029.92]
    assert table["stop_time"][0] == pytest.approx(16479.035 + 1579.058)
    assert table["stop_time"][1] == pytest.approx(11029.92)
    assert list(table["label"][:]) == [
        "Impedance Test On",
        "Type1: bolusing, then EKG",
    ]
    io.close()


def test_no_comments_omits_the_table(tmp_path, persyst_pair):
    """An empty [Comments] section must not produce an unwritable table.

    A TimeIntervals carrying a custom column and zero rows cannot resolve that
    column's dtype and fails the whole write. HUP1234 has a present-but-empty
    [Comments], so this is a real path.
    """
    out = convert(tmp_path, persyst_pair, comments=[])
    io, nwb, _ = read_series(out)
    assert "persyst_comments" not in nwb.intervals
    io.close()


def test_absent_comments_section_omits_the_table(tmp_path, persyst_pair):
    out = convert(tmp_path, persyst_pair, comments=None)
    io, nwb, _ = read_series(out)
    assert "persyst_comments" not in nwb.intervals
    io.close()


def test_write_comments_disabled(tmp_path, persyst_pair):
    out = convert(
        tmp_path,
        persyst_pair,
        comments=[(1.0, 0.0, "kept out")],
        writer={"write_comments": False},
    )
    io, nwb, _ = read_series(out)
    assert "persyst_comments" not in nwb.intervals
    io.close()


def test_long_xml_label_round_trips(tmp_path, persyst_pair):
    label = (
        "<RevealProtocol>" + "<Channel Name='F7'/>" * 250 + "</RevealProtocol>"
    )
    out = convert(tmp_path, persyst_pair, comments=[(0.0, 0.0, label)])
    io, nwb, _ = read_series(out)
    assert nwb.intervals["persyst_comments"]["label"][0] == label
    io.close()


def test_out_of_range_comments_kept_and_counted(tmp_path, persyst_pair, caplog):
    comments = [(-63881.0, 0.0, "Filter Change"), (188452.446, 0.0, "suction")]
    out = convert(tmp_path, persyst_pair, n_samples=500, comments=comments)
    io, nwb, _ = read_series(out)
    assert len(nwb.intervals["persyst_comments"]) == 2
    assert "fall outside" in caplog.text
    io.close()


def test_subject_omitted_when_patient_is_blank(tmp_path, persyst_pair):
    out = convert(tmp_path, persyst_pair, patient={"ID": "", "Sex": ""})
    io, nwb, _ = read_series(out)
    assert nwb.subject is None
    io.close()


PATIENT_2000 = {
    "ID": "HUP1234",
    "Sex": "m",
    "BirthDate": "01/02/80",
    "TestDate": "02/06/2000",
    "TestTime": "04:59:59.964000",
}
"""A [Patient] section with a recording date, to validate a date of birth."""


def test_subject_populated_from_patient(tmp_path, persyst_pair):
    out = convert(tmp_path, persyst_pair, patient=PATIENT_2000)
    io, nwb, _ = read_series(out)
    assert nwb.subject.subject_id == "HUP1234"
    assert nwb.subject.sex == "M"
    assert nwb.subject.date_of_birth.year == 1980
    io.close()


def test_subject_metadata_can_be_switched_off(tmp_path, persyst_pair):
    """A date of birth is a HIPAA Safe Harbor identifier, so make it optional.

    The default is on. A workflow that publishes outside the PHI boundary sets
    WRITE_SUBJECT_METADATA to false.
    """
    out = convert(
        tmp_path,
        persyst_pair,
        patient=PATIENT_2000,
        writer={"write_subject_metadata": False},
    )
    io, nwb, _ = read_series(out)
    assert nwb.subject is None
    io.close()


def test_subject_metadata_written_by_default(tmp_path, persyst_pair):
    out = convert(tmp_path, persyst_pair, patient=PATIENT_2000)
    io, nwb, _ = read_series(out)
    assert nwb.subject is not None
    io.close()


@pytest.mark.parametrize("birthdate", ["01/02/40", "01/02/68", "01/02/2055"])
def test_birth_date_after_the_recording_is_dropped(
    tmp_path, persyst_pair, caplog, birthdate
):
    """strptime reads 68 or less as 20xx, so 01/02/40 gives 2040, not 1940.

    A date of birth at or after the recording is not possible. The writer
    therefore omits that field, and does not write a date in the future. It keeps
    the other subject fields.
    """
    out = convert(
        tmp_path,
        persyst_pair,
        patient={**PATIENT_2000, "BirthDate": birthdate},
    )
    io, nwb, _ = read_series(out)
    assert nwb.subject.subject_id == "HUP1234"
    assert nwb.subject.date_of_birth is None
    assert "omitting date_of_birth" in caplog.text
    io.close()


def test_session_description_records_start_source(tmp_path, persyst_pair):
    out = convert(
        tmp_path,
        persyst_pair,
        patient={"TestDate": "01/19/12", "TestTime": "10:50:22"},
    )
    io, nwb, _ = read_series(out)
    assert "TestDate" in nwb.session_description
    io.close()


def test_failed_write_leaves_no_file_behind(
    tmp_path, persyst_pair, monkeypatch
):
    """HDF5 empties the target on open, so a write in place corrupts the file.

    A non-zero exit code must mean that the converter produced nothing usable. It
    must not leave an incomplete .nwb file in OUTPUT_DIR for the next stage.
    """
    lay, _ = persyst_pair(tmp_path)
    reader = PersystReader(lay, timezone=UTC_ZONE)
    out = tmp_path / "out.nwb"

    def boom(self, *args, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(NWBHDF5IO, "write", boom)
    with pytest.raises(OSError, match="no space left"):
        write_nwb(reader, out, identifier="persyst_test")

    assert not out.exists()
    assert list(tmp_path.glob("*.partial")) == []


def test_out_of_range_count_accounts_for_gaps(tmp_path, persyst_pair, caplog):
    """An annotation in a gap is inside the recording, and not outside it.

    At 250 Hz, 500 samples give 2 s of data. With the gap, the recording spans
    501 s. The annotation at 300 s is therefore in range.
    """
    out = convert(
        tmp_path,
        persyst_pair,
        n_samples=500,
        rate=250.0,
        segments=[(0, 0.0), (250, 500.0)],
        comments=[(300.0, 0.0, "inside the gap")],
    )
    io, nwb, _ = read_series(out)
    assert len(nwb.intervals["persyst_comments"]) == 1
    assert "fall outside" not in caplog.text
    io.close()


def test_output_directory_created(tmp_path, persyst_pair):
    lay, _ = persyst_pair(tmp_path)
    reader = PersystReader(lay, timezone=UTC_ZONE)
    out = tmp_path / "nested" / "deeper" / "out.nwb"
    write_nwb(reader, out, identifier="persyst_test")
    assert out.is_file()
