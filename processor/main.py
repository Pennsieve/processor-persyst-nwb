"""Command-line shell around the conversion.

Takes an explicit path to the Persyst `.lay` input and an explicit path for the
`.nwb` output. Given no paths, it finds the single `.lay` file in `INPUT_DIR` and
writes the result to `OUTPUT_DIR`.
"""

import logging
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from processor.config import Config
from processor.nwb_writer import write_nwb
from processor.reader import PersystReader

logger = logging.getLogger(__name__)

_LAY_SUFFIX = ".lay"
"""Extension of the Persyst header; the ``.dat`` it names sits alongside it."""


def find_layout_file(input_dir: Path) -> Path:
    """Find the single ``.lay`` file in ``input_dir``.

    Matching ignores case because Persyst recordings arrive from Windows. A
    Persyst package holds exactly one header, so raise FileNotFoundError when
    there is none and ValueError when there are several.
    """
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")

    candidates = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == _LAY_SUFFIX
    )
    if not candidates:
        raise FileNotFoundError(
            f"expected one .lay file in {input_dir}, found none"
        )
    if len(candidates) > 1:
        raise ValueError(
            f"expected one .lay file in {input_dir}, found "
            f"{[path.name for path in candidates]!r}"
        )
    return candidates[0]


def convert(lay_path: Path, output_path: Path, config: Config) -> Path:
    """Convert one Persyst recording to NWB at ``output_path``."""
    reader = PersystReader(
        lay_path,
        timezone=config.timezone,
        strip_ref_suffix=config.strip_ref_suffix,
    )
    return write_nwb(
        reader,
        output_path,
        compression_level=config.compression_level,
        target_chunk_bytes=config.target_chunk_bytes,
        write_comments=config.write_comments,
        write_subject_metadata=config.write_subject_metadata,
    )


def main(argv: Sequence[str], env: Mapping[str, str]) -> int:
    """Run the conversion and return a shell exit code.

    The code is 0 on success and 1 on any failure.
    """
    try:
        config = Config.from_env(env)
        lay_path = Path(argv[0]) if argv else find_layout_file(config.input_dir)
        output_path = _output_path(argv, config, lay_path)

        logger.info("converting %s -> %s", lay_path, output_path)
        convert(lay_path, output_path, config)
    except (OSError, ValueError):
        logger.exception("conversion failed")
        return 1

    logger.info("conversion complete")
    return 0


def _output_path(argv: Sequence[str], config: Config, lay_path: Path) -> Path:
    """Decide where the NWB file goes.

    A second positional argument wins, then ``OUTPUT_FILENAME``, then the
    header's own stem, which keeps the recording identifiable after conversion.
    """
    if len(argv) > 1:
        return Path(argv[1])
    name = config.output_filename or f"{lay_path.stem}.nwb"
    return config.output_dir / name


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    raise SystemExit(main(sys.argv[1:], os.environ))
