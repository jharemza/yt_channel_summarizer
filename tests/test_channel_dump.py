from unittest.mock import MagicMock

import pytest

import channel_dump


def test_nested_channel_entries_are_flattened_and_deduplicated(monkeypatch):
    ydl = MagicMock()
    ydl.__enter__.return_value = ydl
    ydl.extract_info.return_value = {
        "entries": [
            None,
            {
                "id": "channel",
                "entries": [
                    {"id": "first", "title": None},
                    {"id": "first"},
                    {"id": "second", "title": "Second"},
                ],
            },
            {"id": "empty-tab", "_type": "playlist"},
        ]
    }
    monkeypatch.setattr(channel_dump, "_ydl", lambda: ydl)
    videos = channel_dump.list_channel_videos("channel", max_videos=2)
    assert [video.id for video in videos] == ["first", "second"]
    assert videos[0].title == ""
    assert videos[1].url == "https://www.youtube.com/watch?v=second"
    ydl.__exit__.assert_called_once()


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_video_limits(limit):
    with pytest.raises(ValueError, match="max_videos"):
        channel_dump.list_channel_videos("channel", max_videos=limit)


@pytest.mark.parametrize("delay", [-1, float("nan"), float("inf")])
def test_invalid_delay_is_rejected_before_creating_output(tmp_path, delay):
    with pytest.raises(ValueError, match="delay_s"):
        channel_dump.fetch_and_store("channel", tmp_path, ["en"], delay_s=delay)
    assert not (tmp_path / "data").exists()


def test_failed_transcripts_are_rate_limited_and_existing_records_skipped(tmp_path, monkeypatch):
    videos = [
        channel_dump.VideoMeta(vid, vid, "url", None, None, None, None)
        for vid in ["cached", "missing", "available"]
    ]
    paths = channel_dump.ensure_dirs(tmp_path)
    (paths["tx_dir"] / "cached.json").write_text("{}")
    monkeypatch.setattr(channel_dump, "list_channel_videos", lambda *a, **kw: videos)
    fetch = MagicMock(
        side_effect=[
            None,
            {
                "lang": "en",
                "is_generated": False,
                "segments": [{"text": "Hello"}],
            },
        ]
    )
    sleep = MagicMock()
    monkeypatch.setattr(channel_dump, "pick_transcript_variant", fetch)
    monkeypatch.setattr(channel_dump.time, "sleep", sleep)
    channel_dump.fetch_and_store("channel", tmp_path, ["en"], delay_s=0.5)
    assert [call.args[0] for call in fetch.call_args_list] == ["missing", "available"]
    sleep.assert_called_once_with(0.5)
    assert '"text": "Hello"' in paths["transcripts_jsonl"].read_text()
