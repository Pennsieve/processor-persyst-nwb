"""Shared pytest fixtures.

No real Persyst recordings are committed, so everything here synthesises them.
The builders mirror the quirks found in real files -- Windows paths in ``File=``,
two-digit years, fractional seconds, comma-laden comment text, NeuroPace
sections -- so those variants stay covered without shipping patient data.
"""

import numpy as np
import pytest

DEFAULT_CHANNELS = ("Fp1-Ref", "Fp2-Ref", "C3-Ref", "C4-Ref")
DEFAULT_RATE = 250.0
DEFAULT_CALIBRATION = 0.2


def build_lay_text(
    *,
    dat_name="rec.dat",
    file_type="Interleaved",
    rate=DEFAULT_RATE,
    header_length=0,
    calibration=DEFAULT_CALIBRATION,
    waveform_count=None,
    datatype=0,
    channels=DEFAULT_CHANNELS,
    patient=None,
    segments=None,
    comments=None,
    np_file_info=None,
    np_comments=None,
    newline="\n",
    extra_sections="",
):
    """Assemble ``.lay`` text from parts.

    ``waveform_count`` defaults to the channel count; pass a different value to
    reproduce the real disagreement between ``WaveformCount`` and
    ``[ChannelMap]``. ``patient`` of None omits the section entirely, whereas an
    empty dict emits it with blank values -- both occur in real files.
    """
    count = len(channels) if waveform_count is None else waveform_count
    lines = ["[FileInfo]", f"File={dat_name}", f"FileType={file_type}"]
    lines += [f"SamplingRate={rate:g}", f"HeaderLength={header_length}"]
    lines += [f"Calibration={calibration:g}", f"WaveformCount={count}"]
    if datatype is not None:
        lines.append(f"DataType={datatype}")

    if np_file_info is not None:
        lines += ["", "[NP_FileInfo]"]
        lines += [f"{k}={v}" for k, v in np_file_info.items()]

    if np_comments is not None:
        lines += ["", "[NP_Comments]"]
        lines += [f"Annotations={row}" for row in np_comments]

    if patient is not None:
        lines += ["", "[Patient]"]
        lines += [f"{k}={v}" for k, v in patient.items()]

    lines += ["", "[ChannelMap]"]
    lines += [f"{name}={i + 1}" for i, name in enumerate(channels)]

    if segments is not None:
        lines += ["", "[SampleTimes]"]
        lines += [f"{start}={time:.3f}" for start, time in segments]

    if comments is not None:
        lines += ["", "[Comments]"]
        lines += [
            f"{onset:.3f},{duration:.3f},0,100,{text}"
            for onset, duration, text in comments
        ]

    if extra_sections:
        lines += ["", extra_sections]

    return newline.join(lines) + newline


def ramp_samples(n_samples, n_channels, dtype):
    """Build samples where channel c at sample i is ``c * 1000 + i``.

    Any channel mix-up or interleave error shows up as an obviously wrong value
    rather than as plausible noise.
    """
    channel = np.arange(n_channels, dtype=np.int64) * 1000
    index = np.arange(n_samples, dtype=np.int64)[:, None]
    return (channel + index).astype(dtype)


def write_pair(
    tmp_path,
    *,
    data=None,
    n_samples=64,
    datatype=0,
    header_length=0,
    stem="rec",
    dat_name=None,
    dat_suffix=".dat",
    **lay_kwargs,
):
    """Write a matching ``.lay``/``.dat`` pair and return both paths.

    ``data`` defaults to a ramp sized to the channel list. ``header_length`` bytes
    of filler are prepended to the ``.dat`` so the offset path can be exercised --
    every real fixture has ``HeaderLength=0``.
    """
    channels = lay_kwargs.get("channels", DEFAULT_CHANNELS)
    dtype = np.dtype("<i4") if datatype == 7 else np.dtype("<i2")

    if data is None:
        data = ramp_samples(n_samples, len(channels), dtype)
    data = np.asarray(data, dtype=dtype)

    dat_path = tmp_path / f"{stem}{dat_suffix}"
    dat_path.write_bytes(b"\x00" * header_length + data.tobytes())

    lay_path = tmp_path / f"{stem}.lay"
    lay_path.write_text(
        build_lay_text(
            dat_name=dat_name if dat_name is not None else dat_path.name,
            datatype=datatype,
            header_length=header_length,
            file_type="32BitInterleaved" if datatype == 7 else "Interleaved",
            **lay_kwargs,
        )
    )
    return lay_path, dat_path


@pytest.fixture
def lay_text():
    return build_lay_text


@pytest.fixture
def persyst_pair():
    return write_pair


@pytest.fixture
def samples():
    return ramp_samples
