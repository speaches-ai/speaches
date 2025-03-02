from speaches.text_utils import (
    srt_format_timestamp,
    vtt_format_timestamp,
)


def test_srt_format_timestamp() -> None:
    assert srt_format_timestamp(0.0) == "00:00:00,000"
    assert srt_format_timestamp(1.0) == "00:00:01,000"
    assert srt_format_timestamp(1.234) == "00:00:01,234"
    assert srt_format_timestamp(60.0) == "00:01:00,000"
    assert srt_format_timestamp(61.0) == "00:01:01,000"
    assert srt_format_timestamp(61.234) == "00:01:01,234"
    assert srt_format_timestamp(3600.0) == "01:00:00,000"
    assert srt_format_timestamp(3601.0) == "01:00:01,000"
    assert srt_format_timestamp(3601.234) == "01:00:01,234"
    assert srt_format_timestamp(23423.4234) == "06:30:23,423"


def test_vtt_format_timestamp() -> None:
    assert vtt_format_timestamp(0.0) == "00:00:00.000"
    assert vtt_format_timestamp(1.0) == "00:00:01.000"
    assert vtt_format_timestamp(1.234) == "00:00:01.234"
    assert vtt_format_timestamp(60.0) == "00:01:00.000"
    assert vtt_format_timestamp(61.0) == "00:01:01.000"
    assert vtt_format_timestamp(61.234) == "00:01:01.234"
    assert vtt_format_timestamp(3600.0) == "01:00:00.000"
    assert vtt_format_timestamp(3601.0) == "01:00:01.000"
    assert vtt_format_timestamp(3601.234) == "01:00:01.234"
    assert vtt_format_timestamp(23423.4234) == "06:30:23.423"
