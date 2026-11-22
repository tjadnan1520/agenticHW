# Assignment 3 (Week 02): structured output for domain-name notices

Extract 8 fields from 50 domain-name notices with `qwen2.5:3b` two ways, prompt-only vs
Ollama grammar-constrained (`format=<JSON Schema>`), 3 runs each, and compute the Qwen2.5-7B KV cache.

```
llm.py              30-line Ollama client (format= support, token ledger, notional cost)
extract.py          runs both methods -> runs/<method>_run<k>.jsonl (raw outputs, never edited)
score.py            scores runs against data/answer_key.json -> results/summary.{json,md}
kv_cache.py         KV-cache calculation (8k and 32k tokens)
cache_from_logs.py  measured prompt-cache hits from an OLLAMA_DEBUG=1 server log
data/ schema/       50 notices, answer key, JSON Schema v1 (unchanged from the original repo)
tests/              python -m unittest
REPORT.md           final report; AI_LOG.md AI-use log
```

## Run
```bash
pip install jsonschema
ollama pull qwen2.5:3b
python kv_cache.py
python extract.py --runs 1 --limit 3 --out smoke   # smoke test
python extract.py                                  # 300 calls, ~50 min on CPU, resumable
# optional, for cache hits: start the server with OLLAMA_DEBUG=1, stderr -> ollama_server.log
python cache_from_logs.py ollama_server.log
python score.py
python -m unittest
```
