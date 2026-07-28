# processor-persyst-nwb

Converts a Persyst EEG recording into a single NWB file. A recording is a `.lay` header file
and an interleaved `.dat` binary file.

Point the converter at the `.lay` file. It writes an NWB file that any NWB reader can open.
The converter preserves the channel labels, the absolute start time, the gaps between
segments, and the annotations.

The project also builds a container image for the Pennsieve analysis platform, where the
converter runs as the first stage of a timeseries workflow.

## Usage

Give the converter an input path and an output path. It finds the `.dat` file that the
`.lay` header names in the same directory.

```bash
python -m processor.main <input.lay> <output.nwb>
```

If you supply no arguments, the converter reads the single `.lay` file in `INPUT_DIR` and
writes the result to `OUTPUT_DIR`. Containerized runs use this form.

To run the container locally, copy a recording into `data/input` and then start it. Git
ignores that directory and `data/output`.

```bash
cp <recording>.lay <recording>.dat data/input/
make run
```

Exit code 0 means the conversion succeeded. Any other exit code means it failed.

## Configuration

Set these environment variables to change the defaults. All are optional.
On Pennsieve, the workflow `params` arrive as upper-case environment variables,
so you can set these per workflow.

| Variable | Default | Description |
|---|---|---|
| `INPUT_DIR` | `/data/input` | Directory that holds one `.lay` file and its `.dat` file |
| `OUTPUT_DIR` | `/data/output` | Directory to write the NWB file to |
| `OUTPUT_FILENAME` | `<lay stem>.nwb` | Name of the output file |
| `PERSYST_TIMEZONE` | `UTC` | Time zone for `TestDate` and `TestTime` |
| `CHUNK_TARGET_BYTES` | `4194304` | Target size of one HDF5 chunk |
| `COMPRESSION_LEVEL` | `4` | gzip level. Set to `0` to disable compression |
| `STRIP_REF_SUFFIX` | `false` | Write channel `Fp1-Ref` as `Fp1` |
| `WRITE_COMMENTS` | `true` | Write `[Comments]` as a `TimeIntervals` table |

## Output format

The output file holds one `ElectricalSeries` in `/acquisition`.

**Samples.** The `data` dataset has the shape `(samples, channels)`. It holds the recorded
integers, not scaled values, with `conversion` set to `Calibration x 1e-6` and `offset` set
to `0.0`. The NWB schema fixes `ElectricalSeries.unit` to volts, so `data x conversion x 1e6`
gives the microvolts that the recording encodes. Integers are lossless and use about half the
space of scaled float64 values.

**Compression.** Samples and timestamps use gzip level 4 with the shuffle filter. Chunks have
the shape `(n, channels)`, where `n` comes from `CHUNK_TARGET_BYTES`. On integer EEG data, the
shuffle filter reduces the samples to about 0.56 of their original size, against 0.79 for
gzip alone.

**Timing.** A continuous recording gets a sample `rate` and `starting_time`.
A recording with gaps gets one timestamp per sample instead.
`pynwb` rejects a series that has both.
A constant rate asserts uniform sampling and cannot express a gap,
so a recording with gaps must carry timestamps.

**Channels.** The electrodes table has one row for each data column. The `channel_name`
column holds the label from the header, unchanged. The `group_name` column comes from the
electrode group, which `pynwb` fills in.

**Start time.** `session_start_time` always has a time zone. All timestamps are relative to
it. The converter selects the start time from the first of these sources that is available,
and records the source in `session_description`:

1. A NeuroPace FILETIME value, which is an unambiguous UTC instant.
2. `TestDate` and `TestTime`, read in `PERSYST_TIMEZONE`.
3. The first `[SampleTimes]` value, if it is a plausible Unix timestamp.
4. The Unix epoch. The converter logs a warning.

**Annotations.** The `[Comments]` entries become a `TimeIntervals` table named
`persyst_comments` in `/intervals`, with `start_time`, `stop_time`, and `label` columns. The
table sits outside `/acquisition`, so a reader that scans for signal data cannot mistake it
for a recording. The converter keeps annotations that fall outside the recording and logs how
many there were. If a recording has no comments, the converter omits the table. A
`TimeIntervals` table that has a custom column and no rows cannot resolve the column type,
and the write fails.

## Persyst format

The `[FileInfo]` section is the only section that the converter requires.

| Key | Meaning | Notes |
|---|---|---|
| `File` | Name of the `.dat` file | Can be an absolute Windows path. The case can differ from the file on disk |
| `FileType` | `Interleaved` or `32BitInterleaved` | Used only if `DataType` is absent |
| `SamplingRate` | Sample rate in Hz | One rate applies to all channels |
| `HeaderLength` | Byte offset into the `.dat` file | Usually 0. The converter applies it |
| `Calibration` | Counts for each microvolt | Becomes the NWB `conversion` value |
| `WaveformCount` | Number of channels | Unreliable. See below |
| `DataType` | `0` for int16, `7` for int32 | Any other value is an error |

The `.dat` file has no header. It holds little-endian integers, interleaved by sample:
`s0c0, s0c1, ... s0cN, s1c0`, and so on. A reshape to `(samples, channels)` needs no
transpose, which is the orientation that NWB uses.

**Channel count.** The converter counts the entries in `[ChannelMap]`. It does not use
`WaveformCount`. Some recordings declare more waveforms than the channel map lists. If you
read such a file at the declared width, each column holds every second sample of one channel.
The recovered frequencies then double. The file size cannot show which width is correct,
because these files divide evenly at both widths. A test converts known sine waves and checks
the frequencies that the converter recovers.

**Segments.** The `[SampleTimes]` section maps a start sample index to a segment start time.
The epoch of that time differs between recordings. In some files it is a Unix timestamp that
matches `TestDate` and `TestTime`. In others it is neither a Unix timestamp nor a count of
seconds since midnight. Only the differences between entries are reliable. The converter
therefore computes segment offsets relative to the first entry, and takes the absolute start
time from another field.

Persyst also rounds these times to milliseconds, an error of up to 500 microseconds.
Above about 1024 Hz, that error is larger than the gap threshold.
The converter therefore snaps a segment boundary to the exact time when the
boundary is within tolerance of where continuous sampling would place it.
Without this step, a continuous 2048 Hz recording reports false gaps,
and can produce timestamps that do not increase.

**Comments.** Each `[Comments]` row has the form `onset,duration,state,var_type,text`, not
`key=value`. The text can contain commas, colons, and backslashes. Some recordings store
several kilobytes of XML in one comment. An onset can be negative, or later than the end of
the samples.

**NeuroPace recordings.** Some recordings come from NeuroPace RNS devices. These files add
`[NP_*]` sections, omit `[Patient]` and `[SampleTimes]`, and repeat the `Annotations` key. A
Windows FILETIME value in `[NP_FileInfo]` is their only absolute time.

## Architecture

The conversion runs in the following order: parse the header from the `.lay` file,
resolve the timebase, map the samples into memory, then write the `.nwb` file.

Most modules hold no I/O, so you can test the complex logic without a file.

`layout.py` parses the text of a `.lay` file into frozen dataclasses. It reads nothing else,
so a test can supply a header as a string.

`timebase.py` holds the date formats, the FILETIME conversion, the order of
start-time sources, and the segment, gap, and timestamp arithmetic.

`sizing.py` computes the chunk shape.

`constants.py` holds static Persyst format information as well as static NWB output values.

`reader.py` opens the Persyst recording. It locates the `.dat` file, maps it into
memory, and returns windows of recorded integers.

`nwb_writer.py` builds the NWB file.
It streams the samples through a chunk iterator, so a long recording never loads into memory.

`config.py` converts a mapping of environment variables into a frozen `Config`. It takes the
mapping as an argument instead of reading `os.environ`, which keeps it free of I/O.

`main.py` locates the input, returns an exit code, and is the only module that calls
`logging.basicConfig`.

### Why not use an off-the-shelf Persyst reader?

`neuroconv`, `neo`, and `spikeinterface` currently have no Persyst support.

MNE is the only popular neurodata library that reads Persyst recordings, through
`mne.io.read_raw_persyst`. However, it failed to open every recording used while developing
this converter, for these reasons:

- It accepts a four-digit year in `TestDate` and rejects a two-digit year.
- It rejects a `TestTime` value that includes fractional seconds.
- It cannot locate the `.dat` file when `File` holds an absolute Windows path. It splits the
  path with the separator of the host operating system.
- It matches the `.dat` filename case-sensitively, and Persyst writes a lower-case name for a
  file that is often upper-case on disk.
- It fails when the header has no `[Patient]` section.
- Recent versions also require a `Hand` field and a `BirthDate` value that they can parse.
- It fails when `WaveformCount` and `[ChannelMap]` disagree.

Three further problems affect the data that MNE returns:

- It ignores `[SampleTimes]`, so it reports a segmented recording as continuous. Some
  recordings join many sessions, with gaps of hours or days between them. MNE discards those
  gaps.
- It changes the channel labels. It converts them to upper case and removes a `-Ref` suffix.
- It reads `TestDate` and `TestTime` as UTC, although Persyst stores no time zone.

The test suite still uses MNE as an oracle. One test writes a header that MNE accepts and
compares the samples that both readers produce. That test confirms the sample type, the
interleaving, and the calibration independently.

## Development

The code targets Python 3.12. `mypy --strict` checks it, and `ruff` lints it with a strict
rule set that includes pydocstyle. Each module in `processor/` has a matching test file in
`tests/`.

```bash
make venv        # create the virtual environment and install dependencies
make test        # run the tests
make test-cov    # run the tests and report coverage
make typecheck   # run mypy --strict
make lint        # run ruff check --fix and ruff format
make check       # run ruff check, ruff format --check, mypy, and the tests
make pre-commit  # install the pre-commit hook
```

The repository contains no Persyst recordings. The tests write `.lay` and `.dat` files into a
temporary directory, and reproduce each format problem described above.

`tests/test_integration.py` converts a recording and reads the result back through a plain
NWB reader. That reader pairs electrodes with data columns, derives the sample rate, splits
the samples at gaps, and scales the values to microvolts. A change that breaks a real reader
fails this test first.

## Dependencies

The converter requires `numpy`, `pynwb`, `hdmf`, and `tzdata`. Development and testing also
require `pytest`, `pytest-cov`, `mypy`, `ruff`, `pre-commit`, and `mne`.
