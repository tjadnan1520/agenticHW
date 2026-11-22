"""Recover measured prompt-cache hits from an Ollama server log (needs OLLAMA_DEBUG=1).

The /api/chat response does NOT say how many prompt tokens came from the prefix cache. With
debug logging the server prints, for every call:
    new prompt, n_ctx_slot = 4096, ... task.n_tokens = N
    cached n_tokens = C
We take the last len(records) such calls, check every N equals the record's prompt_tokens
(otherwise we refuse to guess), and write runs/prompt_cache.jsonl.

    python cache_from_logs.py ollama_server.log
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NUM_CTX = 4096
ORDER = [(m, k) for k in (1, 2, 3) for m in ("prompt", "grammar")]  # extract.py loop order


def parse_log(lines):
    """Return [{'n_tokens': N, 'cached_tokens': C}] for each call logged at NUM_CTX."""
    calls, cur = [], None
    for line in lines:
        m = re.search(r"new prompt, n_ctx_slot = (\d+).*task\.n_tokens = (\d+)", line)
        if m:
            cur = {"n_ctx": int(m.group(1)), "n_tokens": int(m.group(2))}
            continue
        m = re.search(r"cached n_tokens = (\d+)", line)
        if m and cur:
            calls.append({**cur, "cached_tokens": int(m.group(1))})
            cur = None
    return [c for c in calls if c["n_ctx"] == NUM_CTX]


def main(log_path, runs_dir=HERE / "runs"):
    calls = parse_log(Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines())
    recs = [json.loads(l) for m, k in ORDER
            for l in (runs_dir / f"{m}_run{k}.jsonl").read_text(encoding="utf-8").splitlines() if l]
    if len(calls) < len(recs):
        sys.exit(f"log has {len(calls)} calls, runs have {len(recs)}: cannot match")
    calls = calls[-len(recs):]
    bad = [r["id"] for r, c in zip(recs, calls) if r["prompt_tokens"] != c["n_tokens"]]
    if bad:
        sys.exit(f"log and runs disagree on prompt size for {len(bad)} calls, e.g. {bad[:5]}")
    with (runs_dir / "prompt_cache.jsonl").open("w", encoding="utf-8") as f:
        for r, c in zip(recs, calls):
            f.write(json.dumps({"method": r["method"], "run": r["run"], "id": r["id"],
                                "prompt_tokens": r["prompt_tokens"],
                                "cached_tokens": c["cached_tokens"]}) + "\n")
    total_p = sum(r["prompt_tokens"] for r in recs)
    total_c = sum(c["cached_tokens"] for c in calls)
    print(f"matched {len(recs)} calls; cached {total_c:,} of {total_p:,} prompt tokens "
          f"({total_c / total_p:.1%})")


if __name__ == "__main__":
    main(sys.argv[1])
