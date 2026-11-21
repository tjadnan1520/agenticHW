"""Minimal Ollama client over plain HTTP (no third-party packages)."""

import json
import urllib.error
import urllib.request
from collections.abc import Iterator


class LLMError(Exception):
    pass


class Ollama:
    def __init__(self, model: str, host: str = "http://localhost:11434",
                 num_ctx: int = 4096, keep_alive: str = "30m"):
        self.model = model
        self._host = host.rstrip("/")
        self._num_ctx = num_ctx
        self._keep_alive = keep_alive

    def _open(self, messages, temperature, num_predict, stream, fmt):
        payload = {
            "model": self.model, "messages": messages, "stream": stream,
            "keep_alive": self._keep_alive,
            "options": {"temperature": temperature, "num_predict": num_predict,
                        "num_ctx": self._num_ctx, "seed": 42},
        }
        if fmt:
            payload["format"] = fmt
        request = urllib.request.Request(
            f"{self._host}/api/chat", json.dumps(payload).encode("utf-8"),
            {"Content-Type": "application/json"},
        )
        try:
            return urllib.request.urlopen(request, timeout=300)
        except urllib.error.HTTPError as exc:
            raise LLMError(f"Ollama error {exc.code}: {exc.read().decode(errors='replace')[:200]}") from exc
        except OSError as exc:
            raise LLMError(f"Cannot reach Ollama at {self._host}: {exc}") from exc

    def chat(self, messages, temperature: float = 0.0, num_predict: int = 256, fmt=None) -> str:
        """One complete reply. `fmt` is an optional JSON schema that constrains the output."""
        try:
            with self._open(messages, temperature, num_predict, False, fmt) as resp:
                return json.load(resp)["message"]["content"]
        except (ValueError, KeyError) as exc:
            raise LLMError(f"Unexpected reply from Ollama: {exc}") from exc

    def stream(self, messages, temperature: float = 0.5, num_predict: int = 200) -> Iterator[str]:
        """Yield the reply piece by piece."""
        try:
            with self._open(messages, temperature, num_predict, True, None) as resp:
                for line in resp:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get("message", {}).get("content"):
                        yield data["message"]["content"]
                    if data.get("done"):
                        return
        except (ValueError, OSError) as exc:
            raise LLMError(f"Reply stream failed: {exc}") from exc
