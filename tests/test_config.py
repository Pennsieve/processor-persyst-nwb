from pathlib import Path

import pytest

from processor.config import Config
from processor.constants import DEFAULT_COMPRESSION_LEVEL, TARGET_CHUNK_BYTES


def test_defaults_match_the_container_mounts():
    config = Config.from_env({})
    assert config.input_dir == Path("/data/input")
    assert config.output_dir == Path("/data/output")
    assert config.output_filename is None
    assert config.timezone.key == "UTC"
    assert config.target_chunk_bytes == TARGET_CHUNK_BYTES
    assert config.compression_level == DEFAULT_COMPRESSION_LEVEL
    assert config.strip_ref_suffix is False
    assert config.write_comments is True


def test_all_settings_overridable():
    # Every setting is retunable without a rebuild.
    config = Config.from_env(
        {
            "INPUT_DIR": "/in",
            "OUTPUT_DIR": "/out",
            "OUTPUT_FILENAME": "rec.nwb",
            "PERSYST_TIMEZONE": "America/New_York",
            "CHUNK_TARGET_BYTES": "1048576",
            "COMPRESSION_LEVEL": "9",
            "STRIP_REF_SUFFIX": "true",
            "WRITE_COMMENTS": "false",
        }
    )
    assert config.input_dir == Path("/in")
    assert config.output_dir == Path("/out")
    assert config.output_filename == "rec.nwb"
    assert config.timezone.key == "America/New_York"
    assert config.target_chunk_bytes == 1048576
    assert config.compression_level == 9
    assert config.strip_ref_suffix is True
    assert config.write_comments is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("true", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("0", False),
        ("no", False),
        ("", False),
    ],
)
def test_boolean_parsing(value, expected):
    # An unset variable keeps the default; an empty one does too.
    default_when_blank = value == ""
    config = Config.from_env({"STRIP_REF_SUFFIX": value})
    assert config.strip_ref_suffix is (
        False if default_when_blank else expected
    )


def test_empty_output_filename_treated_as_unset():
    assert Config.from_env({"OUTPUT_FILENAME": ""}).output_filename is None


def test_unknown_timezone_raises():
    with pytest.raises(ValueError, match="unknown PERSYST_TIMEZONE"):
        Config.from_env({"PERSYST_TIMEZONE": "Mars/Olympus_Mons"})


def test_compression_level_zero_allowed():
    assert Config.from_env({"COMPRESSION_LEVEL": "0"}).compression_level == 0


@pytest.mark.parametrize("value", ["-1", "10", "42"])
def test_compression_level_out_of_range_raises(value):
    with pytest.raises(ValueError, match="between 0 and 9"):
        Config.from_env({"COMPRESSION_LEVEL": value})


def test_non_numeric_compression_level_raises():
    with pytest.raises(ValueError, match="not an integer"):
        Config.from_env({"COMPRESSION_LEVEL": "high"})


@pytest.mark.parametrize("value", ["0", "-4096"])
def test_non_positive_chunk_target_raises(value):
    with pytest.raises(ValueError, match="must be positive"):
        Config.from_env({"CHUNK_TARGET_BYTES": value})


def test_non_numeric_chunk_target_raises():
    with pytest.raises(ValueError, match="not an integer"):
        Config.from_env({"CHUNK_TARGET_BYTES": "4MiB"})
