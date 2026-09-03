"""Grade survey generations for instruction-following with a Haiku judge.

Reads a phase2 records jsonl (the artifact format), grades each non-excluded
generation's prompt + decoded text with claude-haiku-4-5, and writes one
judgment per record to a sidecar jsonl (same order, keyed by line index), so
raw artifacts stay untouched and compliant-conditioned rates are a join away:

    ANTHROPIC_API_KEY=... uv run python -m retok.phase2_judge \\
        data/retok/english_v2/<model>.jsonl \\
        --out data/retok/english_v2/judge_<model>.jsonl

Default transport is the **Message Batches API** (half price, no rate-limit
management; one batch per input file, polled until done). ``--mode sync``
uses direct threaded calls instead — handy for smoke tests with ``--limit``.

Judgment fields: ``followed`` in {full, partial, no}, ``language`` (dominant
language of the output), ``coherent`` (bool — degenerate/looping/word-salad
output is False even when on-topic). The judge sees only decoded TEXT — it
knows nothing about tokenization, so grading cannot leak the outcome
variable. For harmony-format models (gpt-oss) only the final channel is
graded; long outputs are truncated head+tail so a late answer is never cut
away.

Failures are recorded as ``{"error": ...}`` for that line and treated as
ungraded (never compliant) by the analysis join.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.anthropic.com/v1"
JUDGE_MODEL = "claude-haiku-4-5"

JUDGE_PROMPT = """\
You are grading whether a language model's output followed a writing task.

<task>
{prompt}
</task>

<output>
{text}
</output>

Grade ONLY the output above. Reply with a single JSON object, no other text:
{{"followed": "full" | "partial" | "no",
  "language": "<dominant language of the output, e.g. English>",
  "coherent": true | false}}

- "full": the output does what the task asked (format and topic), even if
  it is cut off mid-sentence at the end.
- "partial": clearly attempts the task but misses the format or drifts.
- "no": ignores the task, or is unrelated text.
- "coherent" is false for degenerate output: looping, word salad, or text
  that no fluent reader could follow — regardless of topic.
"""


def _judge_view(text: str) -> str:
    """What the judge reads: final-channel-only for harmony models, head+tail
    truncation for long outputs (so a late final answer is never cut away)."""
    marker = "<|channel|>final<|message|>"
    if marker in text:
        text = text.split(marker)[-1]
    if len(text) > 6000:
        text = text[:3000] + "\n...[middle truncated for grading]...\n" + text[-3000:]
    return text


def _params(prompt: str, text: str) -> dict:
    return {
        "model": JUDGE_MODEL,
        "max_tokens": 200,
        "messages": [
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(prompt=prompt, text=_judge_view(text)),
            }
        ],
    }


def _request(
    method: str, url: str, api_key: str, body: dict | None = None, *, raw: bool = False
) -> dict:
    req = urllib.request.Request(
        url,
        method=method,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        data=json.dumps(body).encode() if body is not None else None,
    )
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                text = r.read().decode()
            return {"_raw": text} if raw else json.loads(text)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529) and attempt < 5:
                time.sleep(2**attempt)
                continue
            raise
        except (urllib.error.URLError, http.client.HTTPException, OSError):
            if attempt < 5:
                time.sleep(2**attempt)
                continue
            raise
    raise RuntimeError("unreachable")


def _parse_verdict(message: dict) -> dict:
    raw = message["content"][0]["text"].strip()
    if raw.startswith("```"):
        raw = raw.strip("`").removeprefix("json").strip()
    # Haiku sometimes appends commentary after the JSON object; take the
    # first object and ignore trailing text rather than failing the line.
    start = raw.find("{")
    if start < 0:
        raise ValueError(f"no JSON object: {raw[:80]}")
    verdict, _ = json.JSONDecoder().raw_decode(raw[start:])
    if verdict.get("followed") not in ("full", "partial", "no"):
        raise ValueError(f"bad verdict: {raw[:100]}")
    return {
        "followed": verdict["followed"],
        "language": str(verdict.get("language", "?")),
        "coherent": bool(verdict.get("coherent", False)),
    }


def grade_batch(rows: list[dict], api_key: str) -> dict[int, dict]:
    """One Message Batch covering every non-excluded row; poll until done."""
    requests = [
        {"custom_id": f"idx-{i}", "params": _params(r["prompt"], r.get("text", ""))}
        for i, r in enumerate(rows)
        if not r.get("excluded")
    ]
    out: dict[int, dict] = {}
    if not requests:
        return out
    batch = _request("POST", f"{API}/messages/batches", api_key, {"requests": requests})
    batch_id = batch["id"]
    print(f"submitted batch {batch_id} ({len(requests)} requests)", flush=True)
    while batch["processing_status"] != "ended":
        time.sleep(20)
        batch = _request("GET", f"{API}/messages/batches/{batch_id}", api_key)
        print(
            f"  {batch['processing_status']} {batch.get('request_counts')}", flush=True
        )
    results_raw = _request("GET", batch["results_url"], api_key, raw=True)["_raw"]
    for line in results_raw.splitlines():
        res = json.loads(line)
        i = int(res["custom_id"].removeprefix("idx-"))
        result = res["result"]
        if result["type"] == "succeeded":
            try:
                out[i] = _parse_verdict(result["message"])
            except (ValueError, KeyError, IndexError, json.JSONDecodeError) as e:
                out[i] = {"error": f"parse:{e}"}
        else:
            out[i] = {"error": f"batch:{result['type']}"}
    return out


def grade_sync(rows: list[dict], api_key: str, workers: int) -> dict[int, dict]:
    def one(item: tuple[int, dict]) -> tuple[int, dict]:
        i, r = item
        try:
            resp = _request(
                "POST",
                f"{API}/messages",
                api_key,
                _params(r["prompt"], r.get("text", "")),
            )
            return i, _parse_verdict(resp)
        except Exception as e:
            return i, {"error": f"{type(e).__name__}:{e}"}

    todo = [(i, r) for i, r in enumerate(rows) if not r.get("excluded")]
    out: dict[int, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for i, verdict in pool.map(one, todo):
            out[i] = verdict
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("records", help="phase2 records jsonl to grade")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None, help="grade first N only")
    p.add_argument("--mode", choices=["batch", "sync"], default="batch")
    p.add_argument("--workers", type=int, default=12, help="sync-mode concurrency")
    args = p.parse_args()
    api_key = os.environ["ANTHROPIC_API_KEY"]

    with open(args.records) as fh:
        rows = [json.loads(line) for line in fh]
    if args.limit is not None:
        rows = rows[: args.limit]

    if args.mode == "batch":
        verdicts = grade_batch(rows, api_key)
    else:
        verdicts = grade_sync(rows, api_key, args.workers)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for i, r in enumerate(rows):
            if r.get("excluded"):
                j: dict = {"skipped": "excluded"}
            else:
                j = verdicts.get(i, {"error": "missing-from-batch"})
            j["idx"] = i
            j["model"] = r.get("model")
            j["domain"] = r.get("domain")
            f.write(json.dumps(j) + "\n")
    print(f"wrote {len(rows)} judgments to {out_path}")


if __name__ == "__main__":
    main()
