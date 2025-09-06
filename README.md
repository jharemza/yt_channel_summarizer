# YouTube Channel Summarizer

Scripts to download YouTube channel metadata and transcripts, then generate AI summaries.

## Requirements

- Python 3.12+
- An OpenAI API key for summarization

## Installation

```bash
git clone <repository-url>
cd yt_channel_summarizer
python -m venv .venv && source .venv/bin/activate  # optional
pip install -r requirements.txt
```

## Usage

### 1. Fetch channel data

`channel_dump.py` lists videos from a channel or playlist, stores a manifest of basic metadata, and saves available transcripts.

```bash
python channel_dump.py CHANNEL_URL --out . --langs en,en-US --max 50
```

Outputs are written under `data/raw/`:

- `manifest.csv` – video metadata
- `transcripts.jsonl` – combined transcripts
- `transcripts/{id}.json` – per‑video records

### 2. Summarize transcripts

`summarize.py` reads `data/raw/transcripts.jsonl` and produces CSV and Markdown summaries via the OpenAI API.

Set your API key and run:

```bash
export OPENAI_API_KEY=your_key
python summarize.py --config config/summarizer.yaml
```

Summary files are written to `data/processed/`.

### 3. Optional: extract plain transcript text

Convert a transcript JSON file to plain text:

```bash
python scripts/extract_transcript_text.py data/raw/transcripts/{id}.json
```

Text files default to `data/transcript_text/`.

## Configuration

Edit `config/summarizer.yaml` to adjust model settings, prompts, chunk sizes, and output paths.
`MODEL` and `TEMPERATURE` environment variables override those values at runtime.

## Development

Run linters and tests before committing:

```bash
pre-commit run --files README.md
pytest
```

## License

MIT


