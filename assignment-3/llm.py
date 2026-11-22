"""Minimal Ollama chat client (stands in for the missing class-1 `sqlagent.llm.OllamaChat`).

`format=None`  -> prompt-only call.
`format=<schema dict>` -> Ollama grammar-constrained decoding against that JSON Schema.
Every call is appended to `self.ledger` with token counts, time and a notional cost.
"""
import json
import time
import urllib.request

# Teaching rates (USD per million tokens). Local Ollama has no real bill: these are notional.
RATE_INPUT, RATE_CACHED_INPUT, RATE_OUTPUT = 2.0, 0.5, 8.0


def notional_cost(prompt_tokens, completion_tokens, cached_tokens=0):
    fresh = prompt_tokens - cached_tokens
    return (fresh * RATE_INPUT + cached_tokens * RATE_CACHED_INPUT
            + completion_tokens * RATE_OUTPUT) / 1e6


class OllamaChat:
    def __init__(self, model, host="http://localhost:11434", **options):
        self.model, self.host, self.options = model, host, options
        self.ledger = []

    def chat(self, messages, format=None):
        body = {"model": self.model, "messages": messages, "stream": False,
                "options": self.options}
        if format is not None:
            body["format"] = format
        req = urllib.request.Request(self.host + "/api/chat", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        start = time.time()
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.load(resp)
        entry = {"model": self.model,
                 "prompt_tokens": data.get("prompt_eval_count", 0),
                 "completion_tokens": data.get("eval_count", 0),
                 "seconds": round(time.time() - start, 3),
                 "prompt_eval_ms": round(data.get("prompt_eval_duration", 0) / 1e6, 1),
                 "done_reason": data.get("done_reason")}
        entry["cost_usd"] = round(notional_cost(entry["prompt_tokens"],
                                                entry["completion_tokens"]), 6)
        self.ledger.append(entry)
        return data["message"]["content"]
