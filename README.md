# YouTube Channel Summarizer

Scripts to download YouTube channel metadata and transcripts and generate AI summaries.

## Installation

1. Clone the repository and switch into it.
2. (Optional) Create a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
pip install openai tiktoken orjson pyyaml python-dotenv tenacity tqdm
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

`summarize.py` reads `data/raw/transcripts.jsonl` and produces CSV/Markdown summaries via the OpenAI API.

```bash
export OPENAI_API_KEY=your_key
python summarize.py --config config/summarizer.yaml
```

Summary files are written to `data/processed/`.

## Configuration

Adjust `config/summarizer.yaml` to change model settings, prompts, chunk sizes, or output paths.

## License

MIT

