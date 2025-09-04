#!/usr/bin/env python3
"""Convert a transcript JSON file into plain text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_OUT_DIR = Path("data") / "transcript_text"


def extract_text(json_file: Path, out_dir: Path | None = None) -> Path:
    """Extract transcript text from *json_file* into *out_dir*.

    Returns the path to the written text file.
    Raises ValueError if the JSON is malformed or missing ``text``.
    """
    out_dir = out_dir or DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        data = json.loads(json_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - message tested via CLI
        raise ValueError(f"Malformed JSON: {exc}") from exc

    text = data.get("text")
    if not isinstance(text, str):
        raise ValueError("Missing 'text' field in transcript JSON")

    out_file = out_dir / (json_file.stem + ".txt")
    out_file.write_text(text, encoding="utf-8")
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract transcript text from a JSON file")
    parser.add_argument("json_file", type=Path, help="Path to transcript JSON file")
    args = parser.parse_args()

    try:
        out_path = extract_text(args.json_file)
    except Exception as exc:  # pragma: no cover - exercised via CLI tests
        parser.exit(1, f"Error: {exc}\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
