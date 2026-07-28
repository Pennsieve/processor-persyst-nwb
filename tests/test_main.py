import h5py
import pytest

from processor.main import find_layout_file, main

BASE_ENV = {"PERSYST_TIMEZONE": "UTC"}


def env_for(tmp_path, output_dir=None, **extra):
    return {
        **BASE_ENV,
        "INPUT_DIR": str(tmp_path),
        "OUTPUT_DIR": str(output_dir or tmp_path / "out"),
        **extra,
    }


def test_find_layout_file(tmp_path, persyst_pair):
    lay, _ = persyst_pair(tmp_path)
    assert find_layout_file(tmp_path) == lay


def test_find_layout_file_is_case_insensitive(tmp_path, persyst_pair):
    lay, _ = persyst_pair(tmp_path)
    upper = lay.with_suffix(".LAY")
    lay.rename(upper)
    assert find_layout_file(tmp_path) == upper


def test_find_layout_file_none_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="found none"):
        find_layout_file(tmp_path)


def test_find_layout_file_multiple_raises(tmp_path, persyst_pair):
    persyst_pair(tmp_path, stem="a")
    persyst_pair(tmp_path, stem="b")
    with pytest.raises(ValueError, match="expected one .lay file"):
        find_layout_file(tmp_path)


def test_missing_input_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        find_layout_file(tmp_path / "nope")


def test_returns_zero_and_writes_one_nwb(tmp_path, persyst_pair):
    persyst_pair(tmp_path, stem="rec", n_samples=300)
    out_dir = tmp_path / "out"
    assert main([], env_for(tmp_path, out_dir)) == 0

    # One recording in, exactly one NWB file out.
    produced = sorted(out_dir.glob("*.nwb"))
    assert [p.name for p in produced] == ["rec.nwb"]
    with h5py.File(produced[0]) as f:
        assert f["acquisition/ElectricalSeries/data"].shape == (300, 4)


def test_output_filename_override(tmp_path, persyst_pair):
    persyst_pair(tmp_path, stem="rec")
    out_dir = tmp_path / "out"
    env = env_for(tmp_path, out_dir, OUTPUT_FILENAME="custom.nwb")
    assert main([], env) == 0
    assert (out_dir / "custom.nwb").is_file()


def test_positional_arguments_override_env(tmp_path, persyst_pair):
    lay, _ = persyst_pair(tmp_path, stem="rec")
    target = tmp_path / "elsewhere" / "explicit.nwb"
    assert main([str(lay), str(target)], BASE_ENV) == 0
    assert target.is_file()


def test_output_directory_created(tmp_path, persyst_pair):
    persyst_pair(tmp_path, stem="rec")
    out_dir = tmp_path / "deeply" / "nested"
    assert main([], env_for(tmp_path, out_dir)) == 0
    assert (out_dir / "rec.nwb").is_file()


def test_no_lay_file_returns_one(tmp_path):
    # Non-zero so a container orchestrator sees the failure.
    assert main([], env_for(tmp_path)) == 1


def test_two_lay_files_return_one(tmp_path, persyst_pair):
    persyst_pair(tmp_path, stem="a")
    persyst_pair(tmp_path, stem="b")
    assert main([], env_for(tmp_path)) == 1


def test_missing_dat_returns_one(tmp_path, persyst_pair):
    _, dat = persyst_pair(tmp_path, stem="rec")
    dat.unlink()
    assert main([], env_for(tmp_path)) == 1


def test_bad_configuration_returns_one(tmp_path, persyst_pair):
    persyst_pair(tmp_path, stem="rec")
    env = env_for(tmp_path, PERSYST_TIMEZONE="Nowhere/Special")
    assert main([], env) == 1


def test_failure_is_logged(tmp_path, caplog):
    assert main([], env_for(tmp_path)) == 1
    assert "conversion failed" in caplog.text
