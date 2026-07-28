import logging
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from processor.layout import Segment, parse_layout
from processor.timebase import (
    SegmentSpan,
    filetime_to_datetime,
    has_gaps,
    parse_test_datetime,
    resolve_session_start,
    segment_spans,
    timestamps_seconds,
    timestamps_window,
)

UTC_ZONE = ZoneInfo("UTC")
EASTERN = ZoneInfo("America/New_York")


@pytest.mark.parametrize(
    ("testdate", "testtime", "expected"),
    [
        # Both spellings appear in real files; MNE supports only the first and
        # crashes on the second.
        (
            "02/06/2000",
            "04:59:59.964000",
            datetime(2000, 2, 6, 4, 59, 59, 964000),
        ),
        ("01/19/12", "10:50:22", datetime(2012, 1, 19, 10, 50, 22)),
        ("2016.02.13", "10:42:45", datetime(2016, 2, 13, 10, 42, 45)),
        ("06-02-2000", "04:59", datetime(2000, 2, 6, 4, 59)),
    ],
)
def test_parse_test_datetime_formats(testdate, testtime, expected):
    parsed = parse_test_datetime(testdate, testtime, UTC_ZONE)
    assert parsed == expected.replace(tzinfo=UTC_ZONE)


@pytest.mark.parametrize(
    ("testdate", "testtime"),
    [
        ("", "10:50:22"),
        ("01/19/12", ""),
        (None, "10:50:22"),
        ("01/19/12", None),
    ],
)
def test_parse_test_datetime_missing_returns_none(testdate, testtime):
    assert parse_test_datetime(testdate, testtime, UTC_ZONE) is None


@pytest.mark.parametrize(
    ("testdate", "testtime"),
    [("19 January 2012", "10:50:22"), ("01/19/12", "half past ten")],
)
def test_parse_test_datetime_unparseable_returns_none(testdate, testtime):
    assert parse_test_datetime(testdate, testtime, UTC_ZONE) is None


def test_parse_test_datetime_applies_timezone():
    parsed = parse_test_datetime("02/06/2000", "04:59:59", EASTERN)
    assert parsed.tzinfo is EASTERN
    assert parsed.utcoffset().total_seconds() == -5 * 3600


def test_filetime_to_datetime_keeps_subsecond():
    # The real NeuroPace stamp carries a half-second fraction.
    assert filetime_to_datetime(130998625655000000) == datetime(
        2016, 2, 13, 18, 42, 45, 500000, tzinfo=UTC
    )


def _layout(lay_text, **kwargs):
    return parse_layout(lay_text(**kwargs))


def test_session_start_prefers_filetime_over_patient(lay_text):
    # FILETIME is an unambiguous UTC instant; TestDate/TestTime carry no zone.
    layout = _layout(
        lay_text,
        np_file_info={"ECoGTimeStampAsUTC": "130998625655000000"},
        patient={"TestDate": "01/19/12", "TestTime": "10:50:22"},
    )
    start, source = resolve_session_start(layout, UTC_ZONE)
    assert start == datetime(2016, 2, 13, 18, 42, 45, 500000, tzinfo=UTC)
    assert "ecogtimestampasutc" in source


def test_session_start_falls_back_to_layout_filetime(lay_text):
    layout = _layout(
        lay_text,
        np_file_info={"LayoutFileTimeStampAsUTC": "130998627755100000"},
    )
    _, source = resolve_session_start(layout, UTC_ZONE)
    assert "layoutfiletimestampasutc" in source


def test_session_start_from_patient(lay_text):
    layout = _layout(
        lay_text,
        patient={"TestDate": "02/06/2000", "TestTime": "04:59:59.964000"},
    )
    start, source = resolve_session_start(layout, UTC_ZONE)
    assert start == datetime(2000, 2, 6, 4, 59, 59, 964000, tzinfo=UTC)
    assert "TestDate" in source


def test_session_start_from_plausible_epoch_sample_time(lay_text):
    layout = _layout(
        lay_text, segments=[(0, 949813199.964), (22542, 949856397.964)]
    )
    start, source = resolve_session_start(layout, UTC_ZONE)
    assert start == datetime(2000, 2, 6, 4, 59, 59, 964000, tzinfo=UTC)
    assert "Unix seconds" in source


def test_session_start_ignores_implausible_sample_time(lay_text):
    # test-persyst.lay's 102903.000 is neither an epoch nor seconds-past-midnight.
    layout = _layout(lay_text, segments=[(0, 102903.0), (7680, 103093.0)])
    start, source = resolve_session_start(layout, UTC_ZONE)
    assert start == datetime(1970, 1, 1, tzinfo=UTC)
    assert "fallback" in source


def test_session_start_fallback_warns(lay_text, caplog):
    resolve_session_start(_layout(lay_text), UTC_ZONE)
    assert "no usable start time" in caplog.text


def test_session_start_always_timezone_aware(lay_text):
    # A naive value would be read in whatever zone the reader happens to have.
    for kwargs in (
        {},
        {"patient": {"TestDate": "01/19/12", "TestTime": "10:50:22"}},
        {"np_file_info": {"ECoGTimeStampAsUTC": "130998625655000000"}},
    ):
        start, _ = resolve_session_start(_layout(lay_text, **kwargs), UTC_ZONE)
        assert start.tzinfo is not None


def test_segment_spans_without_sample_times():
    spans = segment_spans((), 1000, 250.0)
    assert spans == (SegmentSpan(0, 1000, 0.0),)


def test_segment_spans_single_entry_is_contiguous():
    spans = segment_spans((Segment(0, 102903.0),), 1000, 250.0)
    assert spans == (SegmentSpan(0, 1000, 0.0),)


def test_segment_spans_tile_range_exactly():
    segments = (Segment(0, 0.0), Segment(100, 1000.0), Segment(250, 2000.0))
    spans = segment_spans(segments, 400, 250.0)
    assert [(s.start_sample, s.stop_sample) for s in spans] == [
        (0, 100),
        (100, 250),
        (250, 400),
    ]
    assert sum(s.n_samples for s in spans) == 400


def test_segment_spans_offsets_are_relative_to_first():
    # HUP1234's values are Unix epoch seconds; only the differences matter.
    segments = (Segment(0, 949813199.964), Segment(22542, 949856397.964))
    spans = segment_spans(segments, 325438, 250.0)
    assert spans[0].offset_s == 0.0
    assert spans[1].offset_s == pytest.approx(43198.0)


def test_segment_spans_drops_entries_beyond_data(caplog):
    segments = (Segment(0, 0.0), Segment(500, 100.0), Segment(9999, 200.0))
    spans = segment_spans(segments, 1000, 250.0)
    assert len(spans) == 2
    assert "at or beyond sample 1000" in caplog.text


def test_segment_spans_sorts_unordered_entries():
    segments = (Segment(500, 100.0), Segment(0, 0.0))
    spans = segment_spans(segments, 1000, 250.0)
    assert [s.start_sample for s in spans] == [0, 500]


def test_segment_spans_non_monotonic_times_fall_back(caplog):
    segments = (Segment(0, 500.0), Segment(500, 100.0))
    spans = segment_spans(segments, 1000, 250.0)
    assert spans == (SegmentSpan(0, 1000, 0.0),)
    assert "not monotonic" in caplog.text


def test_segment_spans_rejects_empty_recording():
    with pytest.raises(ValueError, match="no samples"):
        segment_spans((), 0, 250.0)


def test_segment_spans_allows_a_very_short_leading_span():
    segments = (Segment(0, 0.0), Segment(3, 500.0))
    spans = segment_spans(segments, 1000, 250.0)
    assert spans[0].n_samples == 3
    assert sum(s.n_samples for s in spans) == 1000


def test_hup1234_gap_reproduced():
    # 22542 samples at 250 Hz is 90.168 s, but the next segment starts 43198 s
    # later, so the real gap is 43107.832 s.
    segments = (Segment(0, 949813199.964), Segment(22542, 949856397.964))
    spans = segment_spans(segments, 45084, 250.0)
    gap = spans[1].offset_s - (spans[0].offset_s + spans[0].n_samples / 250.0)
    assert spans[0].n_samples == 22542
    assert gap == pytest.approx(43107.832, abs=1e-6)
    assert has_gaps(spans, 250.0)


def test_has_gaps_false_for_contiguous_segments():
    # Boundary times exactly match gapless sampling.
    segments = (Segment(0, 0.0), Segment(250, 1.0), Segment(500, 2.0))
    spans = segment_spans(segments, 750, 250.0)
    assert not has_gaps(spans, 250.0)


def test_has_gaps_true_just_past_threshold():
    rate = 250.0
    # Segment 1 covers 1.0 s; start it 1.0 s + 3 sample periods later.
    segments = (Segment(0, 0.0), Segment(250, 1.0 + 3 / rate))
    spans = segment_spans(segments, 500, rate)
    assert has_gaps(spans, rate)


def test_has_gaps_false_for_single_span():
    assert not has_gaps((SegmentSpan(0, 100, 0.0),), 250.0)


def test_timestamps_length_matches_samples():
    # NWB requires one timestamp per sample.
    spans = segment_spans((Segment(0, 0.0), Segment(250, 100.0)), 500, 250.0)
    assert timestamps_seconds(spans, 250.0).size == 500


def test_timestamps_strictly_increasing_across_gap():
    spans = segment_spans((Segment(0, 0.0), Segment(250, 100.0)), 500, 250.0)
    ts = timestamps_seconds(spans, 250.0)
    assert np.all(np.diff(ts) > 0)


def test_timestamps_start_at_zero():
    spans = segment_spans((Segment(0, 949813199.964),), 100, 250.0)
    assert timestamps_seconds(spans, 250.0)[0] == 0.0


def test_timestamps_contiguous_within_span():
    rate = 256.0
    spans = (SegmentSpan(0, 1000, 0.0),)
    ts = timestamps_seconds(spans, rate)
    assert np.allclose(np.diff(ts), 1 / rate)


@pytest.mark.parametrize("rate", [250.0, 256.0, 1024.0, 2000.0, 2048.0, 4096.0])
def test_millisecond_quantised_boundaries_do_not_fake_gaps(rate):
    """Contiguous segments whose stored times are ms-rounded must stay gapless.

    Persyst writes [SampleTimes] to three decimals, an error of up to 500 us.
    Above ~1024 Hz that exceeds the 2/rate gap threshold, so without snapping a
    perfectly contiguous 2048 Hz recording reports gaps and can even produce
    non-monotonic timestamps.
    """
    rng = np.random.default_rng(0)
    lengths = rng.integers(int(rate * 0.7), int(rate * 1.3), size=40)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    segments = tuple(
        Segment(int(start), round(float(start) / rate, 3)) for start in starts
    )
    n_samples = int(starts[-1] + lengths[-1])

    spans = segment_spans(segments, n_samples, rate)
    assert not has_gaps(spans, rate)

    ts = timestamps_seconds(spans, rate)
    diffs = np.diff(ts)
    assert np.all(diffs > 0), "timestamps must be strictly increasing"
    assert np.all(diffs <= 2 / rate), "no diff may cross the gap threshold"


def test_real_gap_survives_snapping():
    # Snapping must not erase a genuine discontinuity.
    rate = 2048.0
    segments = (Segment(0, 0.0), Segment(2048, 60.0))
    spans = segment_spans(segments, 4096, rate)
    assert spans[1].offset_s == 60.0
    assert has_gaps(spans, rate)


def test_snapping_logged(caplog):
    caplog.set_level(logging.INFO)
    rate = 2048.0
    segments = (Segment(0, 0.0), Segment(1000, round(1000 / rate, 3)))
    segment_spans(segments, 2000, rate)
    assert "snapped" in caplog.text


@pytest.mark.parametrize("ticks", ["0", "-5", "1"])
def test_implausible_filetime_falls_through(lay_text, caplog, ticks):
    """A FILETIME of 0 converts to 1601 and gives no error, so check the range.

    Without the check, such a value takes precedence over a usable
    TestDate/TestTime, and the recording gets a date in the 17th century.
    """
    layout = _layout(
        lay_text,
        np_file_info={"ECoGTimeStampAsUTC": ticks},
        patient={"TestDate": "01/19/12", "TestTime": "10:50:22"},
    )
    start, source = resolve_session_start(layout, UTC_ZONE)
    assert start == datetime(2012, 1, 19, 10, 50, 22, tzinfo=UTC_ZONE)
    assert "implausible FILETIME" in caplog.text
    assert "TestDate" in source


def test_unreadable_filetime_falls_through(lay_text, caplog):
    layout = _layout(
        lay_text,
        np_file_info={"ECoGTimeStampAsUTC": "not-a-filetime"},
        patient={"TestDate": "01/19/12", "TestTime": "10:50:22"},
    )
    start, source = resolve_session_start(layout, UTC_ZONE)
    assert start == datetime(2012, 1, 19, 10, 50, 22, tzinfo=UTC_ZONE)
    assert "unreadable FILETIME" in caplog.text
    assert "TestDate" in source


@pytest.mark.parametrize(
    ("start", "stop"),
    [
        (0, 500),
        (0, 1),
        (100, 200),
        (240, 260),  # straddles the segment boundary at 250
        (249, 251),
        (250, 500),
        (499, 500),
        (0, 0),
        (500, 500),
    ],
)
def test_timestamps_window_matches_the_full_array(start, stop):
    """A window must equal the same slice of the whole recording.

    The writer streams the timestamps one chunk at a time. Each window boundary
    must therefore give the same values as one pass over the recording. This
    applies to a boundary inside a segment, and to a boundary on a gap.
    """
    spans = segment_spans((Segment(0, 0.0), Segment(250, 100.0)), 500, 250.0)
    whole = timestamps_seconds(spans, 250.0)
    window = timestamps_window(spans, 250.0, start, stop)
    assert np.array_equal(window, whole[start:stop])


def test_timestamps_window_dtype_is_float64():
    spans = segment_spans((Segment(0, 0.0), Segment(250, 100.0)), 500, 250.0)
    assert timestamps_window(spans, 250.0, 0, 10).dtype == np.float64


def test_timestamps_of_empty_spans_is_empty():
    assert timestamps_seconds((SegmentSpan(0, 0, 0.0),), 250.0).size == 0


def test_segment_spans_deduplicates_repeated_start_samples():
    segments = (Segment(0, 0.0), Segment(500, 100.0), Segment(500, 200.0))
    spans = segment_spans(segments, 1000, 250.0)
    assert [s.start_sample for s in spans] == [0, 500]


def test_segment_spans_prepends_when_first_key_is_not_zero(caplog):
    segments = (Segment(100, 0.0), Segment(500, 100.0))
    spans = segment_spans(segments, 1000, 250.0)
    assert spans[0].start_sample == 0
    assert "not 0; prepending" in caplog.text
    assert sum(s.n_samples for s in spans) == 1000
    # The converter must calculate the time of the added span backward from the
    # first entry. The time of that entry leaves both spans at offset 0.0, and
    # the timestamps then decrease at sample 100.
    assert spans[1].offset_s == pytest.approx(100 / 250.0)
    assert np.all(np.diff(timestamps_seconds(spans, 250.0)) > 0)


def test_overlapping_segment_times_are_clamped(caplog):
    """A second span must not start before the first span ends.

    The [SampleTimes] entries can be in order and still describe an overlap. At
    250 Hz, 250 samples fill 1.0 s, so an entry 0.5 s later describes samples
    that the first span also holds. Such an entry gives timestamps that decrease.
    """
    segments = (Segment(0, 0.0), Segment(250, 0.5), Segment(400, 1000.0))
    spans = segment_spans(segments, 600, 250.0)
    assert [s.offset_s for s in spans] == [0.0, 1.0, 1000.0]
    assert "overlap" in caplog.text
    assert np.all(np.diff(timestamps_seconds(spans, 250.0)) > 0)
    # Clamping must not erase the genuine discontinuity that follows.
    assert has_gaps(spans, 250.0)


def test_overlap_without_a_later_gap_becomes_contiguous(caplog):
    # has_gaps examines only forward jumps. An overlap that the converter does
    # not move therefore reaches the constant-rate branch, and the segment
    # information is lost.
    segments = (Segment(0, 0.0), Segment(250, 0.5))
    spans = segment_spans(segments, 500, 250.0)
    assert [s.offset_s for s in spans] == [0.0, 1.0]
    assert "overlap" in caplog.text
    assert not has_gaps(spans, 250.0)
    assert np.all(np.diff(timestamps_seconds(spans, 250.0)) > 0)


@pytest.mark.parametrize(
    "segments",
    [
        (Segment(100, 0.0), Segment(500, 100.0)),
        (Segment(0, 0.0), Segment(250, 0.5), Segment(400, 1000.0)),
        (Segment(0, 0.0), Segment(250, 0.5)),
        (Segment(0, 949813199.964), Segment(22542, 949856397.964)),
        (Segment(0, 0.0), Segment(100, 1000.0), Segment(250, 2000.0)),
        (Segment(500, 100.0), Segment(0, 0.0)),
        (Segment(0, 0.0), Segment(500, 100.0), Segment(500, 200.0)),
    ],
)
@pytest.mark.parametrize("rate", [250.0, 2048.0])
def test_spans_always_tile_and_advance(segments, rate):
    """The two rules that each caller needs, for every branch above.

    The spans tile [0, n_samples) exactly, and no span starts before the previous
    span ends. The per-sample timestamps therefore always increase.
    """
    n_samples = 1000
    spans = segment_spans(segments, n_samples, rate)

    assert spans[0].start_sample == 0
    assert spans[-1].stop_sample == n_samples
    assert sum(s.n_samples for s in spans) == n_samples
    for previous, current in zip(spans, spans[1:], strict=False):
        assert current.start_sample == previous.stop_sample
        assert current.offset_s >= previous.offset_s + previous.n_samples / rate
    assert np.all(np.diff(timestamps_seconds(spans, rate)) > 0)
