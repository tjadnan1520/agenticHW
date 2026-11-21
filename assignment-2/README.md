# Assignment 2: English vs Bangla token cost

50 aligned English/Bangla prompt pairs (100 calls) are sent to a local `qwen2.5:3b` through Ollama.
For each call we record input tokens, output tokens, latency and a notional cost, then compare the languages.

Python 3.10+, standard library only. Token counts come from Ollama itself (`prompt_eval_count`, `eval_count`).

## Run
```
ollama pull qwen2.5:3b     # optional, the runner pulls it if missing
python runner.py           # use `py` on Windows if `python` opens the Store
python runner.py --limit 3 # smoke test
python runner.py --fresh   # ignore a saved partial run
```
Options: `--model`, `--host`, `--data`, `--out`, `--num-predict` (default 128).

## Method
- Fixed system prompt, `temperature` 0, `seed` 42, `num_ctx` 4096 for both languages.
- Pair order alternates (English first, then Bangla first) so neither language always gets the warmer model.
- The model stays loaded. Latency is inference time only (`total_duration - load_duration`); load time is logged separately as `load_s`.
- Chat-template overhead is measured once with an empty message (`metadata.json`). A call whose input tokens do not exceed it is rejected as a likely prompt-cache hit.
- Results are saved after every call. If the run stops, running again resumes it (only if model, options and dataset hash match).
- Cost is a course estimate: $0.15 / 1M input tokens, $0.60 / 1M output tokens.
- `--num-predict` caps answers, so long answers can be cut off (`done_reason: length`). Small models can also answer Bangla badly.

## Files
| File | Contents |
|---|---|
| `dataset.tsv` | `id, category, english, bangla`, 50 rows in 6 categories |
| `runner.py` | the whole experiment |
| `results-<model>/results.csv` | one row per call |
| `results-<model>/raw.json` | same records as JSON |
| `results-<model>/comparison.csv` | one row per pair: English, Bangla, difference and ratio (BN/EN) for every metric |
| `results-<model>/summary.csv` | totals per language with difference, ratio and percent change |
| `results-<model>/metadata.json` | model, Ollama version, options, dataset hash, status, timing |
