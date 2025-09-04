from channel_dump import ensure_dirs as cd_ensure_dirs
from summarize import ensure_dirs as sum_ensure_dirs


def test_channel_dump_ensure_dirs(tmp_path):
    paths = cd_ensure_dirs(tmp_path)
    assert (tmp_path / "data" / "raw").is_dir()
    assert (tmp_path / "data" / "raw" / "transcripts").is_dir()
    assert paths["manifest_csv"] == tmp_path / "data" / "raw" / "manifest.csv"
    assert paths["transcripts_jsonl"] == tmp_path / "data" / "raw" / "transcripts.jsonl"


def test_summarize_ensure_dirs(tmp_path):
    target = tmp_path / "subdir" / "file.txt"
    assert not target.parent.exists()
    sum_ensure_dirs(target)
    assert target.parent.exists()
