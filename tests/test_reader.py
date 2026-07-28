from zoneinfo import ZoneInfo

import numpy as np
import pytest

from processor.reader import PersystReader, resolve_dat_path
from tests.conftest import build_lay_text, ramp_samples

UTC_ZONE = ZoneInfo("UTC")


def open_reader(lay_path, **kwargs):
    return PersystReader(lay_path, timezone=UTC_ZONE, **kwargs)


@pytest.mark.parametrize("datatype", [0, 7])
def test_deinterleaves_samples_exactly(tmp_path, persyst_pair, datatype):
    """Channel c sample i must read back as c * 1000 + i.

    An interleave or channel-order error shows up as an obviously wrong value
    rather than as plausible-looking noise.
    """
    channels = ("A", "B", "C", "D", "E")
    lay, _ = persyst_pair(
        tmp_path, n_samples=97, datatype=datatype, channels=channels
    )
    reader = open_reader(lay)

    assert reader.n_channels == 5
    assert reader.n_samples == 97
    expected = ramp_samples(97, 5, reader.dtype)
    assert np.array_equal(reader.read_window(0, 97), expected)


@pytest.mark.parametrize("datatype", [0, 7])
def test_dtype_follows_datatype(tmp_path, persyst_pair, datatype):
    lay, _ = persyst_pair(tmp_path, datatype=datatype)
    expected = np.dtype("<i4") if datatype == 7 else np.dtype("<i2")
    assert open_reader(lay).dtype == expected


def test_recovers_known_sine_frequencies(tmp_path):
    """A 2-channel file whose header claims 4 waveforms must decode as 2.

    This mirrors wave_sin, whose .dat divides evenly at both 2 and 4 channels --
    so only the recovered frequencies can prove which width is right.
    """
    rate, n = 800.0, 12000
    t = np.arange(n) / rate
    data = np.column_stack(
        [400 * np.sin(2 * np.pi * 20 * t), 400 * np.sin(2 * np.pi * 10 * t)]
    ).astype("<i2")

    (tmp_path / "w.dat").write_bytes(data.tobytes())
    (tmp_path / "w.lay").write_text(
        build_lay_text(
            dat_name="w.dat",
            rate=rate,
            calibration=1.0,
            waveform_count=4,
            channels=("Sin 20Hz", "Sin 10Hz"),
        )
    )

    reader = open_reader(tmp_path / "w.lay")
    assert reader.n_channels == 2
    assert reader.n_samples == 12000

    got = reader.read_window(0, n).astype(np.float64)
    freqs = np.fft.rfftfreq(n, 1 / rate)
    peaks = [freqs[np.abs(np.fft.rfft(got[:, c])).argmax()] for c in range(2)]
    assert peaks == [20.0, 10.0]


def test_conversion_is_calibration_in_volts(tmp_path, persyst_pair):
    lay, _ = persyst_pair(tmp_path, calibration=0.2)
    assert open_reader(lay).conversion == pytest.approx(0.2e-6)


def test_header_length_offsets_the_memmap(tmp_path, persyst_pair):
    # Every real fixture has HeaderLength=0, so this path needs synthetic data.
    lay, _ = persyst_pair(tmp_path, n_samples=50, header_length=512)
    reader = open_reader(lay)
    assert reader.n_samples == 50
    assert np.array_equal(
        reader.read_window(0, 50), ramp_samples(50, 4, reader.dtype)
    )


def test_trailing_partial_sample_truncated(tmp_path, persyst_pair, caplog):
    lay, dat = persyst_pair(tmp_path, n_samples=50)
    dat.write_bytes(dat.read_bytes() + b"\x01\x02\x03")
    reader = open_reader(lay)
    assert reader.n_samples == 50
    assert "trailing byte" in caplog.text


def test_empty_dat_raises(tmp_path, persyst_pair):
    lay, dat = persyst_pair(tmp_path, n_samples=10)
    dat.write_bytes(b"")
    with pytest.raises(ValueError, match="no data"):
        open_reader(lay)


def test_dat_smaller_than_one_sample_raises(tmp_path, persyst_pair):
    lay, dat = persyst_pair(tmp_path, n_samples=10)
    dat.write_bytes(b"\x00\x00")
    with pytest.raises(ValueError, match="smaller than one"):
        open_reader(lay)


def test_read_window_matches_slice_of_full_read(tmp_path, persyst_pair):
    rng = np.random.default_rng(0)
    data = rng.integers(-3000, 3000, size=(500, 4)).astype("<i2")
    lay, _ = persyst_pair(tmp_path, data=data)
    reader = open_reader(lay)
    assert np.array_equal(reader.read_window(120, 340), data[120:340])
    assert np.array_equal(reader.read_window(0, 500), data)


def test_windows_dat_path_resolved_to_sibling(tmp_path, persyst_pair):
    # 130998627754330000.lay names D:\Archive\...\x.dat and ships x.DAT.
    lay, _ = persyst_pair(
        tmp_path,
        stem="130998627754330000",
        dat_suffix=".DAT",
        dat_name=r"D:\Archive\VDhServer2\191036\130998627754330000.dat",
    )
    reader = open_reader(lay)
    assert reader.dat_path.name == "130998627754330000.DAT"
    assert reader.n_samples == 64


def test_uppercase_dat_extension_matched_case_insensitively(
    tmp_path, persyst_pair
):
    lay, _ = persyst_pair(tmp_path, dat_suffix=".DAT", dat_name="rec.dat")
    assert open_reader(lay).dat_path.suffix == ".DAT"


def test_missing_named_dat_falls_back_to_sole_sibling(
    tmp_path, persyst_pair, caplog
):
    lay, _ = persyst_pair(tmp_path, dat_name="does-not-exist.dat")
    assert open_reader(lay).dat_path.name == "rec.dat"
    assert "falling back" in caplog.text


def test_no_dat_raises(tmp_path, persyst_pair):
    lay, dat = persyst_pair(tmp_path)
    dat.unlink()
    with pytest.raises(FileNotFoundError, match="no .dat file"):
        open_reader(lay)


def test_ambiguous_dat_siblings_raise(tmp_path, persyst_pair):
    lay, _ = persyst_pair(tmp_path, stem="rec", dat_name="missing.dat")
    (tmp_path / "other.dat").write_bytes(b"\x00" * 64)
    (tmp_path / "third.dat").write_bytes(b"\x00" * 64)
    (tmp_path / "rec.dat").unlink()
    with pytest.raises(ValueError, match="cannot choose a .dat"):
        resolve_dat_path(lay, "missing.dat")


def test_sibling_matching_lay_stem_preferred(tmp_path, persyst_pair):
    lay, _ = persyst_pair(tmp_path, stem="rec", dat_name="missing.dat")
    (tmp_path / "other.dat").write_bytes(b"\x00" * 64)
    assert resolve_dat_path(lay, "missing.dat").name == "rec.dat"


def test_channel_names_preserved_by_default(tmp_path, persyst_pair):
    lay, _ = persyst_pair(tmp_path, channels=("Fp1-Ref", "C3-Ref"))
    assert open_reader(lay).channel_names == ("Fp1-Ref", "C3-Ref")


def test_strip_ref_suffix_when_requested(tmp_path, persyst_pair):
    lay, _ = persyst_pair(tmp_path, channels=("Fp1-Ref", "C3-Ref"))
    reader = open_reader(lay, strip_ref_suffix=True)
    assert reader.channel_names == ("Fp1", "C3")


def test_strip_ref_leaves_other_names_alone(tmp_path, persyst_pair):
    lay, _ = persyst_pair(tmp_path, channels=("Lhip1 - Lhip2", "Ch.1"))
    reader = open_reader(lay, strip_ref_suffix=True)
    assert reader.channel_names == ("Lhip1 - Lhip2", "Ch.1")


def test_duration_reflects_sample_count(tmp_path, persyst_pair):
    lay, _ = persyst_pair(tmp_path, n_samples=500, rate=250.0)
    assert open_reader(lay).duration_s == pytest.approx(2.0)


def test_subject_fields_drop_empty_values(tmp_path, persyst_pair):
    lay, _ = persyst_pair(
        tmp_path, patient={"ID": "HUP1234", "Sex": "", "BirthDate": ""}
    )
    assert open_reader(lay).subject_fields() == {"id": "HUP1234"}


def test_gapped_recording_detected(tmp_path, persyst_pair):
    lay, _ = persyst_pair(
        tmp_path, n_samples=500, rate=250.0, segments=[(0, 0.0), (250, 500.0)]
    )
    reader = open_reader(lay)
    assert reader.has_gaps()
    assert reader.timestamps_seconds().size == 500


def test_contiguous_recording_has_no_gaps(tmp_path, persyst_pair):
    lay, _ = persyst_pair(tmp_path, n_samples=500, segments=None)
    assert not open_reader(lay).has_gaps()


def test_reader_matches_mne_on_a_lay_mne_accepts(tmp_path):
    """Cross-check the decode against MNE's independent implementation.

    MNE cannot read any of the four real Persyst fixtures: it rejects two-digit
    years, fractional TestTime, Windows paths in File=, and a missing [Patient]
    section, and additionally requires both a Hand field and a parseable
    BirthDate. So the header below is written specifically to suit it -- which is
    also why MNE is not the runtime reader. It still makes a good oracle for the
    parts it does handle: dtype, interleaving and calibration.
    """
    mne = pytest.importorskip("mne")
    mne.set_log_level("ERROR")

    rate, n, cal = 256.0, 300, 0.2
    channels = ("FP1", "FP2", "C3", "C4")
    rng = np.random.default_rng(1)
    data = rng.integers(-2000, 2000, size=(n, len(channels))).astype("<i2")

    (tmp_path / "m.dat").write_bytes(data.tobytes())
    (tmp_path / "m.lay").write_text(
        build_lay_text(
            dat_name="m.dat",
            rate=rate,
            calibration=cal,
            channels=channels,
            patient={
                "TestDate": "01/19/2012",
                "TestTime": "10:50:22",
                "BirthDate": "01/02/80",
                "Sex": "m",
                "Hand": "r",
                "ID": "ORACLE1",
            },
        )
    )

    reader = open_reader(tmp_path / "m.lay")
    raw = mne.io.read_raw_persyst(str(tmp_path / "m.lay"), preload=True)

    assert list(reader.channel_names) == raw.ch_names
    assert reader.sampling_rate == raw.info["sfreq"]
    # MNE returns volts; we store counts plus a conversion factor.
    ours = reader.read_window(0, n).astype(np.float64) * reader.conversion
    np.testing.assert_allclose(ours, raw.get_data().T, rtol=1e-9, atol=0)
