#!/usr/bin/env python3
"""English vs Bangla token cost on a local Ollama model. Python 3.10+, standard library only.

    python runner.py                  # all 50 pairs (100 calls), resumes if interrupted
    python runner.py --limit 3        # smoke test
    python runner.py --fresh          # ignore a saved partial run
"""
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
LANGS = ("english", "bangla")
METRICS = ("input_tokens", "output_tokens", "total_tokens", "latency_s", "cost_usd")
PRICE_IN, PRICE_OUT = 0.15, 0.60  # notional USD per 1M tokens (course estimate, not a real bill)
SYSTEM = "Answer correctly and briefly, in the same language as the user's message."


def api(host, path, body=None, timeout=900):
    req = urllib.request.Request(
        host + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def get_json(host, path):
    with api(host, path, timeout=10) as r:
        return json.load(r)


def ensure_model(host, model):
    try:
        names = {m["name"] for m in get_json(host, "/api/tags")["models"]}
    except OSError as e:
        sys.exit(f"Cannot reach Ollama at {host} ({e}). Install/start it: https://ollama.com")
    if model in names or f"{model}:latest" in names:
        return
    print(f"Pulling {model} ...")
    shown = -1
    with api(host, "/api/pull", {"model": model}, timeout=None) as r:
        for line in r:
            m = json.loads(line)
            if m.get("total"):
                pct = 10 * int(10 * m["completed"] / m["total"])
                if pct != shown:
                    shown = pct
                    print(f"  {pct}%")


def chat(host, model, prompt, options):
    body = {"model": model, "stream": False, "options": options,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
    with api(host, "/api/chat", body) as r:
        raw = json.load(r)
    if not raw.get("done") or raw.get("prompt_eval_count") is None or raw.get("eval_count") is None:
        raise RuntimeError(f"incomplete Ollama response: {str(raw)[:200]}")
    return raw


def load_dataset(path):
    text = path.read_bytes()
    rows = list(csv.DictReader(text.decode("utf-8-sig").splitlines(), delimiter="\t"))
    if not rows or any(not (r[k] or "").strip() for r in rows for k in ("id", "category", "english", "bangla")):
        sys.exit(f"{path.name}: empty dataset or empty cell")
    return rows, hashlib.sha256(text).hexdigest()


def ratio(b, a):
    return b / a if a else ""


def write_table(path, rows, columns):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def build_comparison(records, dataset):
    by = {(r["id"], r["language"]): r for r in records}
    rows = []
    for d in dataset:
        en, bn = by.get((d["id"], "english")), by.get((d["id"], "bangla"))
        if not (en and bn):
            continue
        row = {"id": d["id"], "category": d["category"],
               "english_prompt": d["english"], "bangla_prompt": d["bangla"]}
        for m in METRICS:
            row[f"english_{m}"], row[f"bangla_{m}"] = en[m], bn[m]
            row[f"{m}_difference"] = bn[m] - en[m]
            row[f"{m}_ratio"] = ratio(bn[m], en[m])
        rows.append(row)
    return rows


def build_summary(records):
    rows = []
    totals = {lang: [r for r in records if r["language"] == lang] for lang in LANGS}
    n = {lang: len(v) for lang, v in totals.items()}
    rows.append({"metric": "prompt_count", "english": n["english"], "bangla": n["bangla"],
                 "difference": n["bangla"] - n["english"], "ratio": ratio(n["bangla"], n["english"]),
                 "percent_change": ""})
    for m in METRICS:
        e, b = (sum(r[m] for r in totals[lang]) for lang in LANGS)
        rows.append({"metric": m if m != "latency_s" else "total_latency_s", "english": e, "bangla": b,
                     "difference": b - e, "ratio": ratio(b, e),
                     "percent_change": 100 * (b - e) / e if e else ""})
    return rows


def save_all(out, records, dataset, meta):
    (out / "raw.json").write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    cols = ["id", "language", "category", "prompt", "output", "input_tokens", "output_tokens",
            "total_tokens", "cost_usd", "latency_s", "load_s", "done_reason"]
    write_table(out / "results.csv", [{c: r[c] for c in cols} for r in records], cols)
    comparison = build_comparison(records, dataset)
    if comparison:
        write_table(out / "comparison.csv", comparison, list(comparison[0]))
        summary = build_summary(records)
        write_table(out / "summary.csv", summary, list(summary[0]))
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def print_summary(records):
    print(f"\n{'metric':18}{'english':>12}{'bangla':>12}{'bn/en':>8}")
    for r in build_summary(records)[1:]:
        print(f"{r['metric']:18}{r['english']:>12.6g}{r['bangla']:>12.6g}{r['ratio']:>7.2f}x")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--data", default=str(HERE / "dataset.tsv"))
    ap.add_argument("--out", help="output folder (default: results-<model>)")
    ap.add_argument("--num-predict", type=int, default=128, help="max answer tokens per call")
    ap.add_argument("--limit", type=int, help="only the first N pairs")
    ap.add_argument("--fresh", action="store_true", help="ignore any saved partial run")
    a = ap.parse_args()

    dataset, data_hash = load_dataset(Path(a.data))
    dataset = dataset[:a.limit] if a.limit else dataset
    out = Path(a.out) if a.out else HERE / f"results-{a.model.replace(':', '-')}"
    out.mkdir(parents=True, exist_ok=True)
    options = {"temperature": 0, "seed": 42, "num_predict": a.num_predict, "num_ctx": 4096}

    ensure_model(a.host, a.model)
    # template overhead (system prompt + chat template): an empty user message. Also a sanity floor.
    overhead = chat(a.host, a.model, "", {**options, "num_predict": 1})["prompt_eval_count"]

    records, old = [], out / "metadata.json"
    if not a.fresh and old.exists() and (out / "raw.json").exists():
        prev = json.loads(old.read_text(encoding="utf-8"))
        if (prev.get("dataset_sha256"), prev.get("model"), prev.get("options")) == (data_hash, a.model, options):
            records = json.loads((out / "raw.json").read_text(encoding="utf-8"))
            print(f"Resuming: {len(records)} calls already done")
    done = {(r["id"], r["language"]) for r in records}

    meta = {"model": a.model, "ollama_version": get_json(a.host, "/api/version").get("version"),
            "options": options, "system_prompt": SYSTEM, "template_overhead_tokens": overhead,
            "dataset_sha256": data_hash, "pairs": len(dataset), "status": "running",
            "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(), "platform": platform.platform(),
            "price_usd_per_million": {"input": PRICE_IN, "output": PRICE_OUT}}
    t0 = time.perf_counter()
    try:
        for i, d in enumerate(dataset):
            # alternate who goes first so neither language always gets the warmer model
            for lang in (LANGS if i % 2 == 0 else LANGS[::-1]):
                if (d["id"], lang) in done:
                    continue
                prompt = d[lang]
                raw = chat(a.host, a.model, prompt, options)
                n_in, n_out = raw["prompt_eval_count"], raw["eval_count"]
                if n_in <= overhead:
                    raise RuntimeError(f"id {d['id']} {lang}: input tokens {n_in} <= overhead {overhead}"
                                       " (prompt cache hit?)")
                records.append({
                    "id": d["id"], "language": lang, "category": d["category"], "prompt": prompt,
                    "output": raw["message"]["content"], "input_tokens": n_in, "output_tokens": n_out,
                    "total_tokens": n_in + n_out,
                    "cost_usd": (n_in * PRICE_IN + n_out * PRICE_OUT) / 1e6,
                    # inference time only; model load is reported separately
                    "latency_s": round((raw["total_duration"] - raw.get("load_duration", 0)) / 1e9, 3),
                    "load_s": round(raw.get("load_duration", 0) / 1e9, 3),
                    "done_reason": raw.get("done_reason", ""),
                })
                save_all(out, records, dataset, {**meta, "calls_completed": len(records)})
            print(f"[{i + 1}/{len(dataset)}] id {d['id']}", flush=True)
        meta["status"] = "complete"
    except KeyboardInterrupt:
        meta["status"] = "interrupted"
        print("\nStopped. Progress is saved; run again to resume.")
    except Exception as e:
        meta["status"] = "failed"
        meta["error"] = str(e)
        print(f"\nFailed: {e}", file=sys.stderr)
    finally:
        meta.update(calls_completed=len(records), elapsed_s=round(time.perf_counter() - t0, 1),
                    finished_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        if records:
            save_all(out, records, dataset, meta)
    if meta["status"] == "complete":
        print_summary(records)
        print(f"\nSaved to {out}")
    else:
        sys.exit(130 if meta["status"] == "interrupted" else 1)


if __name__ == "__main__":
    main()
