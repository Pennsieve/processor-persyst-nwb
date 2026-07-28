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
| `OUTPUT_FILENAME` | `<lay stem>.nwb` | Name of the output file. Must be a bare filename |
| `PERSYST_TIMEZONE` | `UTC` | Time zone for `TestDate` and `TestTime` |
| `CHUNK_TARGET_BYTES` | `4194304` | Target size of one HDF5 chunk |
| `COMPRESSION_LEVEL` | `4` | gzip level. Set to `0` to disable compression |
| `STRIP_REF_SUFFIX` | `false` | Write channel `Fp1-Ref` as `Fp1` |
| `WRITE_COMMENTS` | `true` | Write `[Comments]` as a `TimeIntervals` table |
| `WRITE_SUBJECT_METADATA` | `true` | Write the `[Patient]` subject block. See below |

A boolean setting accepts `true` or `false`, `1` or `0`, `yes` or `no`, `on` or `off`. Any
other value is an error. A typo in `WRITE_COMMENTS` therefore stops the run. It does not
discard the annotations.

`OUTPUT_FILENAME` must be a bare filename. The converter rejects an absolute path, and a path
that contains `..`, because such a value writes the file outside `OUTPUT_DIR`.

### Patient identifiers

`WRITE_SUBJECT_METADATA` controls the NWB `Subject` block. That block holds `[Patient] ID` as
`subject_id`, and `[Patient] BirthDate` as `date_of_birth`. The default is `true`. A date of
birth is a HIPAA Safe Harbor identifier. Set this variable to `false` if the output of your
workflow goes outside the PHI boundary.

Persyst writes a two-digit year. Python reads a two-digit year of 69 or more as 19xx, and a
year of 68 or less as 20xx. The value `01/02/40` therefore gives 2040, not 1940. A date of
birth at or after the recording is not possible. The converter discards such a date and writes
a warning. `TestDate` has the same ambiguity. No other field contradicts it, so the converter
keeps that value.

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

The converter writes the timestamps in chunks, as it writes the samples. One `float64`
timestamp uses 8 bytes per sample. For a 4-channel int16 recording, the full array of
timestamps is as large as the `.dat` file. The converter therefore does not build that array
in memory.

The output path holds a complete file, or no file. HDF5 empties the target file when it opens
it. The converter therefore writes to a temporary file in the same directory, then renames it.
A failed run leaves no file for the next stage to read.

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
for a recording. The converter keeps annotations that fall outside the recording, and logs how
many there were. It counts them against the time the recording spans, which includes the gaps,
and not against the number of samples. If a recording has no comments, the converter omits the
table. A `TimeIntervals` table that has a custom column and no rows cannot resolve the column
type, and the write fails.

The converter writes each onset as the header states it. The Persyst comment epoch does not
always agree with the samples in the `.dat` file. In `test-persyst.lay`, `[SampleTimes]` starts
at 102903 s and the comments run from -63881 s to 188452 s. The samples cover 3796 s. That
`.dat` file is an extract of a longer session, but the annotations describe all of the session.
No field in the header gives the offset between the two epochs. The converter therefore does
not apply an offset, because a wrong offset moves an annotation to a wrong time. A clipped
recording gives the out-of-range warning.

## Persyst format

The `[FileInfo]` section is the only section that the converter requires.

| Key | Meaning | Notes |
|---|---|---|
| `File` | Name of the `.dat` file | Can be an absolute Windows path. The case can differ from the file on disk. If that file is absent, the converter accepts only a sibling `.dat` file with the same stem as the `.lay` file |
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

**Channel order.** Each `[ChannelMap]` entry has the form `Label=Index`. The index is the
1-based position of the label in the interleaved `.dat` file. The converter sorts the channels
by this index. It does not use the order of the lines in the file. If the two orders differ,
the converter writes a warning. Most files list the entries in index order. If the converter
used the line order, it could attach each label to the wrong column. The indices must be a
permutation of 1..n. A sparse map is an error, because the interleave width is then ambiguous.

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

NWB requires timestamps that increase. Two more cases can break this rule.

First, `[SampleTimes]` does not always start at sample 0. The converter adds a span for the
samples before the first entry. It calculates the time of that span backward from the first
entry at the sample rate.

Second, the entry times can be in order and still describe an overlap. At 250 Hz, 250 samples
fill one second. An entry 0.5 s after the previous entry therefore covers samples that the
previous segment also covers. The converter moves such a boundary forward to the end of the
previous segment, and writes a warning.

The spans always cover the recording in order, and their offsets never decrease. The converter
checks these two properties before it writes the file.

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
make lock        # regenerate the dependency locks
```

The repository contains no Persyst recordings. The tests write `.lay` and `.dat` files into a
temporary directory, and reproduce each format problem described above.

`tests/test_integration.py` converts a recording and reads the result back through a plain
NWB reader. That reader pairs electrodes with data columns, derives the sample rate, splits
the samples at gaps, and scales the values to microvolts. A change that breaks a real reader
fails this test first.

The tests also check each conversion with the `pynwb` schema validator and with
`nwbinspector`, at the `BEST_PRACTICE_VIOLATION` level and above. This level is necessary,
because `check_timestamps_ascending` is below the `CRITICAL` level. A filter that used
`CRITICAL` only cannot find timestamps that do not increase. `EXPECTED_DEVIATIONS` lists the
checks that Persyst data cannot satisfy, such as the species and the usually blank `[Patient]`
section. Each entry gives the reason.

## Dependencies

The converter requires `numpy`, `pynwb`, `hdmf`, and `tzdata`. Development and testing also
require `pytest`, `pytest-cov`, `nwbinspector`, `mypy`, `ruff`, `pre-commit`, and `mne`.

`processor/requirements.txt` and `requirements-test.txt` hold the version ranges.
`processor/requirements.lock` and `requirements-test.lock` pin each transitive version with a
hash. All installs use these lock files with `--require-hashes`. The container, the CI
workflow, and `make venv` use the same lock files. The tests therefore run against the same
versions that the image contains. A new release of `pynwb` or `hdmf` cannot change the output
until you update a lock file.

To update the lock files, run `make lock`. This target resolves the versions in the base image
of the Dockerfile, which is pinned by digest. The pins therefore match the container, and not
your local machine. `pytest.ini` does not filter `DeprecationWarning`. An upstream deprecation
is therefore visible when you update a lock file.

The image runs as an unprivileged user, and it works with any UID. The `import pynwb` statement
writes a schema cache before the converter starts. The build creates this cache and keeps it
writable. The CI workflow converts a recording with an arbitrary UID to test this.
