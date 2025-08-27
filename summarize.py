#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# deps
import orjson
import yaml
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

import tiktoken

try:
    from openai import OpenAI
except Exception as e:  # pragma: no cover
    print("OpenAI client not installed. Run: pip install -U openai", file=sys.stderr)
    raise

# ---------- Config & models ----------

@dataclass
class Config:
    provider: str
    model: str
    temperature: float
    max_output_tokens: int
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    input_jsonl: str
    output_csv: str
    output_md: str
    concurrency: int
    rate_limit_rps: float
    map_prompt: str
    reduce_prompt: str

def load_config(path: Path) -> Config:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    # allow env overrides for simple swaps
    model = os.getenv("MODEL", data["model"])
    temperature = float(os.getenv("TEMPERATURE", data["temperature"]))
    return Config(
        provider=data["provider"],
        model=model,
        temperature=temperature,
        max_output_tokens=int(data["max_output_tokens"]),
        chunk_size_tokens=int(data["chunk_size_tokens"]),
        chunk_overlap_tokens=int(data["chunk_overlap_tokens"]),
        input_jsonl=data["input_jsonl"],
        output_csv=data["output_csv"],
        output_md=data["output_md"],
        concurrency=int(data["concurrency"]),
        rate_limit_rps=float(data["rate_limit_rps"]),
        map_prompt=data["map_prompt"],
        reduce_prompt=data["reduce_prompt"],
    )

# ---------- IO helpers ----------

def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield orjson.loads(line)

def ensure_dirs(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

# ---------- Tokenization & chunking ----------

def get_encoder_for_model(model: str):
    # try model-specific; fallback to cl100k_base
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")

def chunk_text_by_tokens(text: str, chunk_size: int, overlap: int, enc) -> List[str]:
    if not text:
        return []
    toks = enc.encode(text)
    n = len(toks)
    chunks = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunk_ids = toks[start:end]
        chunks.append(enc.decode(chunk_ids))
        if end == n:
            break
        start = max(end - overlap, 0)
    return chunks

# ---------- OpenAI calls ----------

class RateLimiter:
    """Very simple client-side throttle."""
    def __init__(self, rps: float):
        self.min_interval = 1.0 / max(rps, 1e-6)
        self._last = 0.0
    def wait(self):
        now = time.perf_counter()
        delta = now - self._last
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self._last = time.perf_counter()

def build_client() -> OpenAI:
    # OpenAI client uses OPENAI_API_KEY from env
    return OpenAI()

class TransientLLMError(Exception):
    pass

@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type(TransientLLMError),
)
def chat_complete(client: OpenAI, model: str, system: str, user: str, temperature: float, max_output_tokens: int) -> str:
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_output_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        # treat as transient to retry
        raise TransientLLMError(str(e))

# ---------- Summarization pipeline ----------

SYSTEM_MAP = "You are a precise technical summarizer. Output only the requested bullets; no preamble."
SYSTEM_REDUCE = "You are a careful editor. Merge bullets into one coherent summary per instructions."

def summarize_one_video(
    record: dict,
    cfg: Config,
    client: OpenAI,
    enc,
    throttle: RateLimiter,
) -> Tuple[str, str]:
    """
    Returns (tldr, bullets_markdown) for a single transcript record.
    """
    meta = record.get("meta", {})
    text = (record.get("text") or "").strip()
    if not text:
        return "", ""

    # 1) Chunk
    chunks = chunk_text_by_tokens(
        text=text,
        chunk_size=cfg.chunk_size_tokens,
        overlap=cfg.chunk_overlap_tokens,
        enc=enc,
    )

    # 2) Map: summarize each chunk
    map_bullets = []
    for ch in chunks:
        throttle.wait()
        user_prompt = cfg.map_prompt.replace("{chunk_text}", ch)
        out = chat_complete(
            client=client,
            model=cfg.model,
            system=SYSTEM_MAP,
            user=user_prompt,
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_output_tokens,
        )
        map_bullets.append(out)

    # 3) Reduce: merge chunk summaries
    merged_input = "\n\n".join(map_bullets)
    throttle.wait()
    # user_reduce = cfg.reduce_prompt.replace("{chunk_text}", "").replace("{bullets}", merged_input)
    user_reduce = f"""{cfg.reduce_prompt}

                   === BULLETS START ===
                   {merged_input}
                   === BULLETS END ===
                   """
    reduce_out = chat_complete(
        client=client,
        model=cfg.model,
        system=SYSTEM_REDUCE,
        user=user_reduce,
        temperature=cfg.temperature,
        max_output_tokens=cfg.max_output_tokens,
    )

    # Parse TL;DR if present (first line starting with TL;DR)
    tldr = ""
    lines = [ln.strip() for ln in reduce_out.splitlines() if ln.strip()]
    if lines and lines[0].lower().startswith("tl;dr"):
        tldr = lines[0]
        bullets_md = "\n".join(lines[1:])
    else:
        bullets_md = "\n".join(lines)

    return tldr, bullets_md

# ---------- CSV/MD writers ----------

def append_csv(path: Path, header: List[str], rows: List[List[str]]) -> None:
    ensure_dirs(path)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(header)
        for r in rows:
            w.writerow(r)

def append_md(path: Path, block: str) -> None:
    ensure_dirs(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(block)
        if not block.endswith("\n"):
            f.write("\n")

def already_done_ids(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    out = set()
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.add(row.get("id", ""))
    return out

# ---------- Main ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize transcripts.jsonl into CSV/Markdown.")
    p.add_argument("--config", default="config/summarizer.yaml", help="Path to YAML config.")
    p.add_argument("--limit", type=int, default=None, help="Only process first N transcripts.")
    p.add_argument("--resume", action="store_true", help="Skip videos already present in output CSV.")
    p.add_argument("--ids", help="Comma-separated video IDs to summarize (filters transcripts.jsonl)")
    return p.parse_args()

def main():
    load_dotenv()
    args = parse_args()
    cfg = load_config(Path(args.config))

    in_path = Path(cfg.input_jsonl)
    out_csv = Path(cfg.output_csv)
    out_md = Path(cfg.output_md)

    if not in_path.exists():
        print(f"Input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    client = build_client()
    enc = get_encoder_for_model(cfg.model)
    throttle = RateLimiter(cfg.rate_limit_rps)

    # Resume support
    done = already_done_ids(out_csv) if args.resume else set()

    header = [
        "id",
        "title",
        "url",
        "upload_date",
        "duration_sec",
        "language",
        "is_generated",
        "tldr",
        "summary_bullets_md",
    ]

    count = 0
    rows_batch: List[List[str]] = []

    records = list(iter_jsonl(in_path))
    if args.limit:
        records = records[: args.limit]
    if args.ids:
        target = {s.strip() for s in args.ids.split(",") if s.strip()}
        records = [r for r in records if r.get("meta", {}).get("id", "") in target]

    for rec in tqdm(records, desc="Summarizing"):
        vid = rec.get("meta", {}).get("id", "")
        if not vid:
            continue
        if args.resume and vid in done:
            continue

        tldr, bullets_md = summarize_one_video(rec, cfg, client, enc, throttle)

        meta = rec.get("meta", {})
        rows_batch.append([
            meta.get("id", ""),
            meta.get("title", ""),
            meta.get("url", ""),
            meta.get("upload_date", ""),
            str(meta.get("duration", "")),
            rec.get("language", ""),
            str(rec.get("is_generated", "")),
            tldr,
            bullets_md,
        ])

        # Append to MD file as a section
        md_block = f"""# {meta.get('title','(untitled)')}
**URL:** {meta.get('url','')}
**Uploaded:** {meta.get('upload_date','')}
**Language:** {rec.get('language','')} | **Auto-captions:** {rec.get('is_generated','')}

{tldr if tldr else ''}

{bullets_md}

---
"""
        append_md(out_md, md_block)

        count += 1
        # flush CSV every few items to be robust
        if len(rows_batch) >= 5:
            append_csv(out_csv, header, rows_batch)
            rows_batch = []

    if rows_batch:
        append_csv(out_csv, header, rows_batch)

    print(f"Done. Wrote: {out_csv} and {out_md}")

if __name__ == "__main__":
    main()
