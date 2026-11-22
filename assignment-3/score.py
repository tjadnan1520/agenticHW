"""Score runs/*.jsonl against data/answer_key.json.

Per run we report:
  json_valid_rate    output parses as a JSON object
  schema_valid_rate  parses AND passes schema v1 AND was not cut off (done_reason != "length")
  invalid_rate       1 - schema_valid_rate
  field_accuracy     correct fields / (calls x 7); an invalid output scores 0 on every field
  per-field and per-notice-type accuracy
Fields scored: the 7 in FIELDS (schema_version is a constant, not scored). Extra keys are ignored
for accuracy (the schema allows them) but counted. Nothing is retried or repaired, except that
a surrounding ``` code fence is stripped.

    python score.py [--runs runs] [--out results]
"""
import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from jsonschema import Draft202012Validator

from llm import notional_cost

HERE = Path(__file__).resolve().parent
SCHEMA = json.loads((HERE / "schema" / "notice_v1.schema.json").read_text(encoding="utf-8"))
KEY = json.loads((HERE / "data" / "answer_key.json").read_text(encoding="utf-8"))
FIELDS = ["domain", "notice_type", "expires_on", "registrar", "amount_due", "currency",
          "action_required"]
VALIDATOR = Draft202012Validator(SCHEMA)
FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def parse(raw):
    """Return the parsed JSON object (dict) or None."""
    text = raw.strip()
    m = FENCE.match(text)
    if m:
        text = m.group(1)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def field_equal(name, got, want):
    if want is None or got is None:
        return got is None and want is None
    if name == "amount_due":
        return isinstance(got, (int, float)) and not isinstance(got, bool) and abs(got - want) < 0.005
    if name == "action_required":
        return got is want
    if name == "registrar":
        norm = lambda s: " ".join(str(s).casefold().split())  # noqa: E731
        return norm(got) == norm(want)
    if name == "domain":
        return str(got).strip().lower() == want
    return got == want


def score_record(rec, key=KEY):
    obj = parse(rec["raw"])
    truncated = rec.get("done_reason") == "length"
    schema_ok = obj is not None and not any(VALIDATOR.iter_errors(obj))
    valid = schema_ok and not truncated
    want = key[rec["id"]]
    correct = {f: bool(valid and field_equal(f, obj.get(f), want[f])) for f in FIELDS}
    extra = len(set(obj) - set(SCHEMA["properties"])) if obj else 0
    return {"id": rec["id"], "type": want["notice_type"], "json_valid": obj is not None,
            "valid": valid, "truncated": truncated, "extra_keys": extra, "correct": correct}


def load_cache(path):
    if not path.exists():
        return {}
    return {(c["method"], c["run"], c["id"]): c["cached_tokens"]
            for c in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if c}


def score_run(path, cache):
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]
    scored = [score_record(r) for r in recs]
    n = len(scored)
    by_type = defaultdict(list)
    for s in scored:
        by_type[s["type"]].append(sum(s["correct"].values()) / len(FIELDS))
    cached = [cache.get((r["method"], r["run"], r["id"]), 0) for r in recs]
    return {
        "file": path.name, "calls": n,
        "json_valid_rate": sum(s["json_valid"] for s in scored) / n,
        "schema_valid_rate": sum(s["valid"] for s in scored) / n,
        "invalid_rate": 1 - sum(s["valid"] for s in scored) / n,
        "field_accuracy": sum(sum(s["correct"].values()) for s in scored) / (n * len(FIELDS)),
        "per_field": {f: sum(s["correct"][f] for s in scored) / n for f in FIELDS},
        "per_type": {t: statistics.mean(v) for t, v in sorted(by_type.items())},
        "truncated": sum(s["truncated"] for s in scored),
        "extra_keys": sum(s["extra_keys"] for s in scored),
        "cost_usd": sum(r["cost_usd"] for r in recs),
        "cost_with_cache_usd": sum(notional_cost(r["prompt_tokens"], r["completion_tokens"], c)
                                   for r, c in zip(recs, cached)),
        "cached_tokens": sum(cached), "prompt_tokens": sum(r["prompt_tokens"] for r in recs),
        "seconds": sum(r["seconds"] for r in recs),
        "invalid_ids": [s["id"] for s in scored if not s["valid"]],
    }


def mean_range(runs, key):
    v = [r[key] for r in runs]
    return {"mean": statistics.mean(v), "min": min(v), "max": max(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    runs_dir, out_dir = HERE / args.runs, HERE / args.out
    out_dir.mkdir(exist_ok=True)
    cache = load_cache(runs_dir / "prompt_cache.jsonl")

    out = {}
    for method in ("prompt", "grammar"):
        runs = [score_run(p, cache) for p in sorted(runs_dir.glob(f"{method}_run*.jsonl"))]
        if runs:
            out[method] = {
                "runs": runs,
                **{k: mean_range(runs, k) for k in ("json_valid_rate", "schema_valid_rate",
                   "invalid_rate", "field_accuracy", "cost_usd", "cost_with_cache_usd", "seconds")},
                "per_field_mean": {f: statistics.mean(r["per_field"][f] for r in runs) for f in FIELDS},
                "per_type_mean": {t: statistics.mean(r["per_type"][t] for r in runs)
                                  for t in runs[0]["per_type"]},
            }
    (out_dir / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    pct = lambda s: f"{s['mean']:.1%} ({s['min']:.1%}-{s['max']:.1%})"  # noqa: E731
    usd = lambda s: f"${s['mean']:.4f} ({s['min']:.4f}-{s['max']:.4f})"  # noqa: E731
    head = "| Metric (mean, min-max over runs) | " + " | ".join(out) + " |\n|---|" + "---|" * len(out)
    rows = [("Valid JSON rate", lambda m: pct(m["json_valid_rate"])),
            ("Schema-valid rate", lambda m: pct(m["schema_valid_rate"])),
            ("Invalid rate", lambda m: pct(m["invalid_rate"])),
            ("Field accuracy (invalid = 0)", lambda m: pct(m["field_accuracy"])),
            ("Notional cost / run, no cache", lambda m: usd(m["cost_usd"])),
            ("Notional cost / run, with measured cache hits", lambda m: usd(m["cost_with_cache_usd"])),
            ("Seconds / run", lambda m: f"{m['seconds']['mean']:.0f}")]
    rows += [(f"  field: {f}", lambda m, f=f: f"{m['per_field_mean'][f]:.1%}") for f in FIELDS]
    types = next(iter(out.values()))["per_type_mean"] if out else {}
    rows += [(f"  type: {t}", lambda m, t=t: f"{m['per_type_mean'][t]:.1%}") for t in types]
    table = head + "\n" + "\n".join(f"| {l} | " + " | ".join(fn(m) for m in out.values()) + " |"
                                    for l, fn in rows)
    (out_dir / "summary.md").write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
