"""Grade survey generations for instruction-following with a Haiku judge.

Reads a phase2 records jsonl (the artifact format), sends each non-excluded
generation's prompt + decoded text to claude-haiku-4-5, and writes one
judgment per record to a sidecar jsonl (same order, keyed by line index), so
raw artifacts stay untouched and compliant-conditioned rates are a join away:

    ANTHROPIC_API_KEY=... uv run python -m retok.phase2_judge \\
        data/retok/english_v2/<model>.jsonl \\
        --out data/retok/english_v2/judge_<model>.jsonl

Judgment fields: ``followed`` in {full, partial, no}, ``language`` (dominant
language of the output), ``coherent`` (bool — degenerate/looping/word-salad
output is False even when on-topic). The judge sees only decoded TEXT — it
knows nothing about tokenization, so grading cannot leak the outcome variable.

Failures (API errors after retries) are recorded as ``{"error": ...}`` for
that line and skipped by the analysis join.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

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


def _judge(prompt: str, text: str, api_key: str) -> dict:
    body = {
        "model": JUDGE_MODEL,
        "max_tokens": 200,
        "messages": [
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(prompt=prompt, text=_judge_view(text)),
            }
        ],
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        data=json.dumps(body).encode(),
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                resp = json.load(r)
            raw = resp["content"][0]["text"].strip()
            if raw.startswith("```"):
                raw = raw.strip("`").removeprefix("json").strip()
            verdict = json.loads(raw)
            if verdict.get("followed") not in ("full", "partial", "no"):
                raise ValueError(f"bad verdict: {raw[:100]}")
            return {
                "followed": verdict["followed"],
                "language": str(verdict.get("language", "?")),
                "coherent": bool(verdict.get("coherent", False)),
            }
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529) and attempt < 4:
                time.sleep(2**attempt)
                continue
            return {"error": f"http:{e.code}"}
        except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
            # transient network failure (DNS blip, reset, incomplete read):
            # retry with backoff rather than poisoning the sidecar
            if attempt < 4:
                time.sleep(2**attempt)
                continue
            return {"error": f"net:{e}"}
        except (ValueError, KeyError, IndexError, json.JSONDecodeError) as e:
            if attempt < 4:
                continue  # re-ask; judge output was malformed
            return {"error": f"parse:{e}"}
    return {"error": "unreachable"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("records", help="phase2 records jsonl to grade")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None, help="grade first N only")
    args = p.parse_args()
    api_key = os.environ["ANTHROPIC_API_KEY"]

    with open(args.records) as fh:
        rows = [json.loads(line) for line in fh]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with out_path.open("w") as f:
        for i, r in enumerate(rows):
            if args.limit is not None and i >= args.limit:
                break
            if r.get("excluded"):
                j: dict = {"skipped": "excluded"}
            else:
                j = _judge(r["prompt"], r.get("text", ""), api_key)
            j["idx"] = i
            j["model"] = r.get("model")
            j["domain"] = r.get("domain")
            f.write(json.dumps(j) + "\n")
            done += 1
            if done % 50 == 0:
                print(f"graded {done}/{len(rows)}")
    print(f"wrote {done} judgments to {out_path}")


if __name__ == "__main__":
    main()
