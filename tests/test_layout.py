import numpy as np
import pytest

from processor.layout import parse_layout

_XML_COMMENT = (
    '<RevealProtocol Name="DefaultScalp"><Detector Name="Spike">'
    + '<Channel Name="F7-aF7" Definition="F7-aF7" Process="1"/>' * 60
    + "</Detector></RevealProtocol>"
)


def test_parses_fileinfo(lay_text):
    info = parse_layout(lay_text()).file_info
    assert info.dat_name == "rec.dat"
    assert info.sampling_rate == 250.0
    assert info.calibration == 0.2
    assert info.header_length == 0
    assert info.dtype == np.dtype("<i2")


@pytest.mark.parametrize(
    ("datatype", "file_type", "expected"),
    [
        (0, "Interleaved", np.dtype("<i2")),
        (7, "32BitInterleaved", np.dtype("<i4")),
    ],
)
def test_datatype_maps_to_dtype(lay_text, datatype, file_type, expected):
    text = lay_text(datatype=datatype, file_type=file_type)
    assert parse_layout(text).file_info.dtype == expected


@pytest.mark.parametrize(
    ("file_type", "expected"),
    [("32BitInterleaved", np.dtype("<i4")), ("Interleaved", np.dtype("<i2"))],
)
def test_missing_datatype_inferred_from_file_type(
    lay_text, file_type, expected
):
    text = lay_text(datatype=None, file_type=file_type)
    assert parse_layout(text).file_info.dtype == expected


@pytest.mark.parametrize("datatype", [1, 3, 99])
def test_unknown_datatype_raises(lay_text, datatype):
    with pytest.raises(ValueError, match="unsupported DataType"):
        parse_layout(lay_text(datatype=datatype))


def test_missing_fileinfo_raises():
    with pytest.raises(ValueError, match=r"no \[FileInfo\] section"):
        parse_layout("[ChannelMap]\nA=1\n")


def test_missing_sampling_rate_raises(lay_text):
    text = lay_text().replace("SamplingRate=250\n", "")
    with pytest.raises(ValueError, match="samplingrate"):
        parse_layout(text)


def test_channel_names_preserve_case_and_spacing(lay_text):
    names = ("Fp1-Ref", "Lhip1 - Lhip2", "Sin 20Hz", "Ch.1")
    parsed = parse_layout(lay_text(channels=names)).channel_names
    assert parsed == names


def test_n_channels_comes_from_channelmap_not_waveform_count(lay_text, caplog):
    # wave_sin.lay really does declare 4 waveforms for 2 mapped channels.
    text = lay_text(channels=("Sin 20Hz", "Sin 10Hz"), waveform_count=4)
    layout = parse_layout(text)
    assert layout.n_channels == 2
    assert "trusting [ChannelMap]" in caplog.text


def test_empty_channelmap_raises(lay_text):
    text = lay_text().split("[ChannelMap]")[0] + "[ChannelMap]\n"
    with pytest.raises(ValueError, match=r"no \[ChannelMap\] entries"):
        parse_layout(text)


def test_channel_names_ordered_by_index_not_file_order(caplog):
    """The [ChannelMap] value is the interleave position, so it must be honored.

    Returning labels in file order while merely *validating* the indices puts
    every label on the wrong data column -- a silent channel swap.
    """
    text = (
        "[FileInfo]\nFile=a.dat\nSamplingRate=250\nCalibration=1\nDataType=0\n"
        "[ChannelMap]\nFp1=2\nFp2=1\nC3=4\nC4=3\n"
    )
    assert parse_layout(text).channel_names == ("Fp2", "Fp1", "C4", "C3")
    assert "not in index order" in caplog.text


def test_channel_names_in_index_order_are_not_reordered(lay_text, caplog):
    layout = parse_layout(lay_text(channels=("A", "B", "C")))
    assert layout.channel_names == ("A", "B", "C")
    assert "not in index order" not in caplog.text


def test_non_sequential_channel_indices_raise():
    # A sparse map leaves the interleave width ambiguous, and file size cannot
    # disambiguate it, so guessing would silently mis-decode.
    text = (
        "[FileInfo]\nFile=a.dat\nSamplingRate=250\nCalibration=1\nDataType=0\n"
        "[ChannelMap]\nA=1\nB=3\n"
    )
    with pytest.raises(ValueError, match="non-sequential"):
        parse_layout(text)


def test_windows_dat_path_preserved_verbatim(lay_text):
    windows = r"D:\Archive\VDhServer2\191036\130998627754330000.dat"
    assert (
        parse_layout(lay_text(dat_name=windows)).file_info.dat_name == windows
    )


def test_absent_patient_section_yields_empty_mapping(lay_text):
    assert parse_layout(lay_text(patient=None)).patient == {}


def test_empty_patient_values_are_empty_strings(lay_text):
    text = lay_text(patient={"BirthDate": "", "Sex": "", "ID": "HUP1234"})
    patient = parse_layout(text).patient
    assert patient["birthdate"] == ""
    assert patient["id"] == "HUP1234"


def test_section_header_without_preceding_blank_line():
    # wave_sin.lay puts [ChannelMap] directly after DataType=0.
    text = (
        "[FileInfo]\nFile=a.dat\nSamplingRate=800\nCalibration=1\nDataType=0\n"
        "[ChannelMap]\nSin 20Hz=1\nSin 10Hz=2\n"
    )
    assert parse_layout(text).n_channels == 2


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_crlf_and_lf_both_parse(lay_text, newline):
    layout = parse_layout(lay_text(newline=newline))
    assert layout.n_channels == 4
    assert layout.channel_names[0] == "Fp1-Ref"


def test_sample_times_parsed_in_order(lay_text):
    text = lay_text(segments=[(0, 102903.0), (7680, 103093.0)])
    segments = parse_layout(text).segments
    assert [s.start_sample for s in segments] == [0, 7680]
    assert [s.start_time_s for s in segments] == [102903.0, 103093.0]


def test_absent_sample_times_yields_no_segments(lay_text):
    assert parse_layout(lay_text(segments=None)).segments == ()


def test_empty_comments_section_yields_no_comments(lay_text):
    # HUP1234_2000_02.lay has a [Comments] section that is present but empty.
    assert parse_layout(lay_text(comments=[])).comments == ()


def test_comment_text_keeps_commas_and_colons(lay_text):
    text = lay_text(
        comments=[
            (11029.92, 0.0, "Type1: bolusing with 10 of versed, then EKG")
        ]
    )
    comment = parse_layout(text).comments[0]
    assert comment.text == "Type1: bolusing with 10 of versed, then EKG"
    assert comment.onset_s == 11029.92


def test_comment_with_embedded_equals_and_xml_survives(lay_text):
    # Comment text contains '=' inside XML attributes, so an '='-sniffing
    # parser would shred it.
    text = lay_text(comments=[(0.0, 0.0, _XML_COMMENT)])
    comment = parse_layout(text).comments[0]
    assert comment.text == _XML_COMMENT
    assert len(comment.text) > 3000


def test_negative_and_out_of_range_onsets_retained(lay_text):
    text = lay_text(
        comments=[
            (-63881.0, 0.0, "Low Pass Filter Change: 70"),
            (188452.446, 0.0, "Type1: suction"),
        ]
    )
    onsets = [c.onset_s for c in parse_layout(text).comments]
    assert onsets == [-63881.0, 188452.446]


def test_comment_duration_retained(lay_text):
    text = lay_text(comments=[(16479.035, 1579.058, "Impedance Test On")])
    assert parse_layout(text).comments[0].duration_s == 1579.058


def test_duplicate_comment_texts_all_retained(lay_text):
    text = lay_text(
        comments=[(1.0, 0.0, "Calibration On"), (2.0, 0.0, "Calibration On")]
    )
    assert len(parse_layout(text).comments) == 2


def test_malformed_comment_row_skipped(lay_text, caplog):
    text = lay_text(comments=[(1.0, 0.0, "good")]).replace(
        "1.000,0.000,0,100,good", "not,a,valid,row,here\n1.000,0.000,0,100,good"
    )
    comments = parse_layout(text).comments
    assert [c.text for c in comments] == ["good"]
    assert "malformed [Comments]" in caplog.text


def test_duplicate_np_annotation_keys_both_retained(lay_text):
    # configparser would raise DuplicateOptionError here.
    text = lay_text(
        np_comments=[
            "DEVICE,PROG_MARKER_MAGNET_APPLIED,CHANNEL_0_0,20.132,0,0",
            "DEVICE,PROG_MARKER_MAGNET_REMOVED,CHANNEL_0_0,20.892,0,0",
        ]
    )
    comments = parse_layout(text).comments
    assert [c.text for c in comments] == [
        "PROG_MARKER_MAGNET_APPLIED",
        "PROG_MARKER_MAGNET_REMOVED",
    ]
    assert [c.onset_s for c in comments] == [20.132, 20.892]


def test_np_file_info_captured(lay_text):
    text = lay_text(np_file_info={"ECoGTimeStampAsUTC": "130998625655000000"})
    assert (
        parse_layout(text).np_file_info["ecogtimestampasutc"]
        == "130998625655000000"
    )


def test_line_without_equals_is_skipped_not_fatal(lay_text):
    text = lay_text(
        extra_sections="[NP_Parameters]\nFirstAmplifierChannel 1  Lhip1"
    )
    assert parse_layout(text).n_channels == 4


def test_extra_unknown_sections_ignored(lay_text):
    text = lay_text(extra_sections="[UserEvents]\n")
    assert parse_layout(text).n_channels == 4


def test_non_numeric_datatype_raises(lay_text):
    text = lay_text().replace("DataType=0", "DataType=int16")
    with pytest.raises(ValueError, match="DataType is not an integer"):
        parse_layout(text)


def test_non_numeric_channel_index_raises():
    text = (
        "[FileInfo]\nFile=a.dat\nSamplingRate=250\nCalibration=1\nDataType=0\n"
        "[ChannelMap]\nA=one\nB=2\n"
    )
    with pytest.raises(ValueError, match="non-sequential"):
        parse_layout(text)


def test_malformed_sample_times_row_skipped(lay_text, caplog):
    text = lay_text(segments=[(0, 0.0)]).replace(
        "[SampleTimes]", "[SampleTimes]\nnotanumber=alsonot"
    )
    layout = parse_layout(text)
    assert [s.start_sample for s in layout.segments] == [0]
    assert "malformed [SampleTimes]" in caplog.text


def test_comment_row_with_too_few_fields_skipped(lay_text, caplog):
    text = lay_text(comments=[(1.0, 0.0, "good")]).replace(
        "[Comments]", "[Comments]\n1.0,2.0,onlythree"
    )
    layout = parse_layout(text)
    assert [c.text for c in layout.comments] == ["good"]
    assert "malformed [Comments]" in caplog.text


def test_np_annotation_with_too_few_fields_skipped(lay_text, caplog):
    text = lay_text(np_comments=["DEVICE,LABEL"])
    assert parse_layout(text).comments == ()
    assert "malformed NP annotation" in caplog.text


def test_np_annotation_with_bad_onset_skipped(lay_text, caplog):
    text = lay_text(np_comments=["DEVICE,LABEL,CHANNEL_0_0,notanumber,0,0"])
    assert parse_layout(text).comments == ()
    assert "bad onset" in caplog.text


def test_np_comments_non_annotation_keys_ignored(lay_text):
    text = lay_text(np_comments=["DEVICE,LABEL,CH,1.0,0,0"]).replace(
        "[NP_Comments]", "[NP_Comments]\nTriggerReason=ECOG_MAGNET_CATEGORY"
    )
    assert len(parse_layout(text).comments) == 1


def test_non_numeric_optional_field_warns_and_defaults(lay_text, caplog):
    text = lay_text().replace("HeaderLength=0", "HeaderLength=none")
    assert parse_layout(text).file_info.header_length == 0
    assert "not a number" in caplog.text


def test_non_numeric_waveform_count_ignored(lay_text, caplog):
    text = lay_text().replace("WaveformCount=4", "WaveformCount=many")
    layout = parse_layout(text)
    assert layout.file_info.waveform_count is None
    assert layout.n_channels == 4
