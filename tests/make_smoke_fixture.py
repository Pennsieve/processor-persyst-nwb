"""Write a small synthetic Persyst pair for the container smoke test.

python3 tests/make_smoke_fixture.py <input_dir> [stem]
"""

import struct
import sys
from pathlib import Path

CHANNELS = ("Fp1-Ref", "Fp2-Ref")
RATE = 200.0
N_SAMPLES = 800


def ramp_bytes(n_samples, n_channels):
    """Interleave int16 samples where channel c at sample i is c * 1000 + i.

    A channel mix-up or an interleave error then shows up as an obviously wrong
    value rather than as plausible noise. This mirrors ``ramp_samples`` in
    conftest.py; the ``<`` keeps the bytes little-endian on any runner.
    """
    values = [
        channel * 1000 + index
        for index in range(n_samples)
        for channel in range(n_channels)
    ]
    return struct.pack(f"<{len(values)}h", *values)


def lay_text(dat_name):
    """Build the minimal header the reader needs: no patient, no annotations."""
    lines = [
        "[FileInfo]",
        f"File={dat_name}",
        "FileType=Interleaved",
        f"SamplingRate={RATE:g}",
        "HeaderLength=0",
        "Calibration=1",
        f"WaveformCount={len(CHANNELS)}",
        "DataType=0",
        "",
        "[ChannelMap]",
        *(f"{name}={index + 1}" for index, name in enumerate(CHANNELS)),
    ]
    return "\n".join(lines) + "\n"


def main(argv):
    target = Path(argv[0]) if argv else Path("data/input")
    stem = argv[1] if len(argv) > 1 else "rec"
    target.mkdir(parents=True, exist_ok=True)

    dat_path = target / f"{stem}.dat"
    lay_path = target / f"{stem}.lay"
    dat_path.write_bytes(ramp_bytes(N_SAMPLES, len(CHANNELS)))
    lay_path.write_text(lay_text(dat_path.name))

    print(f"wrote {lay_path} and {dat_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
