import json
import subprocess
import sys
from pathlib import Path


def run_script(json_path: Path, cwd: Path) -> subprocess.CompletedProcess:
    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "scripts" / "extract_transcript_text.py"
    return subprocess.run(
        [sys.executable, str(script), str(json_path)],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


def test_extract_transcript_text_success(tmp_path: Path) -> None:
    json_file = tmp_path / "sample.json"
    json_file.write_text(json.dumps({"text": "hello world"}))

    result = run_script(json_file, tmp_path)
    assert result.returncode == 0, result.stderr

    out_file = tmp_path / "data" / "transcript_text" / "sample.txt"
    assert out_file.read_text() == "hello world"


def test_extract_transcript_text_missing_text(tmp_path: Path) -> None:
    json_file = tmp_path / "missing.json"
    json_file.write_text(json.dumps({"nope": "data"}))

    result = run_script(json_file, tmp_path)
    assert result.returncode != 0
    assert "Missing 'text' field" in result.stderr


def test_extract_transcript_text_malformed_json(tmp_path: Path) -> None:
    json_file = tmp_path / "bad.json"
    json_file.write_text("{not json}")

    result = run_script(json_file, tmp_path)
    assert result.returncode != 0
    assert "Malformed JSON" in result.stderr
