"""The orchestrator: asks the LLM which tool comes next and is the only link between the tools.

The LLM only picks a tool name. The orchestrator itself passes the user's
original text to generate_sql and the generated SQL to execute_sql, so a small
model cannot garble pasted data or long SQL while copying it.
"""

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass

from .cache import SqlCache
from .llm import LLMError, Ollama
from .tools import ExecuteSql, GenerateSql

FINAL = "final"

ROUTER_PROMPT = """You control a SQL agent. Choose the next action.

Actions:
- {gen}: {gen_desc}
- {exe}: {exe_desc}
- {final}: stop and answer, once the SQL has run successfully.

Normal order: {gen} -> {exe} -> {final}. If {exe} reports an error, choose {gen} again.
You never write SQL yourself. Reply with JSON only: {{"action": "<name>"}}"""

ANSWER_PROMPT = """You answer questions about the user's personal expenses.
Use ONLY the query result you are given and never invent numbers. Amounts are in BDT (taka).
Answer in one or two short sentences."""


@dataclass
class AgentResult:
    ok: bool
    answer: str
    sql: str | None = None
    steps: int = 0
    from_cache: bool = False


@dataclass
class _State:
    question: str
    sql: str | None = None
    key: str | None = None
    from_cache: bool = False
    result: dict | None = None
    error: str | None = None
    retries: int = 0
    use_cache: bool = True


def _rows_text(result: dict, limit: int = 20) -> str:
    text = f"columns={result['columns']} rows={json.dumps(result['rows'][:limit], ensure_ascii=False)}"
    return text + (" (more rows not shown)" if len(result["rows"]) > limit or result["truncated"] else "")


class Orchestrator:
    def __init__(self, llm: Ollama, generator: GenerateSql, executor: ExecuteSql, cache: SqlCache,
                 trace: Callable[[str, str], None], max_steps: int = 8, max_retries: int = 2):
        self._llm, self._gen, self._exe, self._cache = llm, generator, executor, cache
        self._trace = trace
        self._max_steps, self._max_retries = max_steps, max_retries
        self._actions = [generator.name, executor.name, FINAL]
        self._schema = {"type": "object", "required": ["action"],
                        "properties": {"action": {"type": "string", "enum": self._actions}}}
        self._prompt = ROUTER_PROMPT.format(
            gen=generator.name, gen_desc=generator.description,
            exe=executor.name, exe_desc=executor.description, final=FINAL)

    def run(self, question: str, on_token: Callable[[str], None]) -> AgentResult:
        state = _State(question)
        messages = [{"role": "system", "content": self._prompt},
                    {"role": "user", "content": f"User request: {question}"}]
        try:
            for step in range(1, self._max_steps + 1):
                action = self._route(messages, state, step)
                if action == FINAL:
                    return self._answer(state, step, on_token)
                if action == self._gen.name:
                    observation = self._generate(state)
                    if re.fullmatch(r"SELECT\s+'UNSUPPORTED'", state.sql or "", re.I):
                        return AgentResult(False, "I can only record or look up your expenses.",
                                           state.sql, step)
                else:
                    observation = self._execute(state)
                    if state.retries > self._max_retries:
                        message = (f"Blocked for safety ({state.error})" if state.error.startswith("rejected")
                                   else f"The SQL kept failing: {state.error}")
                        return AgentResult(False, message, state.sql, step)
                messages.append({"role": "assistant", "content": json.dumps({"action": action})})
                messages.append({"role": "user", "content": f"Observation: {observation}"})
            return AgentResult(False, "Ran out of steps before finishing.", state.sql, self._max_steps)
        except LLMError as exc:
            return AgentResult(False, str(exc), state.sql)

    # --- routing ---------------------------------------------------------

    def _valid_next(self, state: _State) -> str:
        if state.sql is None or state.error is not None:
            return self._gen.name
        return self._exe.name if state.result is None else FINAL

    def _route(self, messages: list, state: _State, step: int) -> str:
        expected = self._valid_next(state)
        started = time.perf_counter()
        raw = self._llm.chat(messages, temperature=0.0, num_predict=40, fmt=self._schema)
        ms = round((time.perf_counter() - started) * 1000)
        try:
            choice = json.loads(raw)["action"]
        except (ValueError, KeyError, TypeError):
            choice = None
        if choice == expected:
            self._trace("router", f"step {step}: LLM chose {choice}  ({ms} ms)")
            return choice
        # The model picked an impossible step (e.g. execute before any SQL exists).
        self._trace("router", f"step {step}: LLM chose {choice!r}, not valid now -> using {expected}")
        return expected

    # --- tool calls ------------------------------------------------------

    def _generate(self, state: _State) -> str:
        self._trace("tool", f"{self._gen.name} called" + ("  (retry with error)" if state.error else ""))
        out = self._gen.run(state.question, state.sql if state.error else None,
                            state.error, state.use_cache)
        state.sql, state.key, state.from_cache = out["sql"], out["key"], out["cached"]
        state.result = state.error = None
        if out["cached"]:
            self._trace("cache", f"HIT  (saved {out['saved_at']}, used {out['hits']}x)")
        else:
            self._trace("cache", f"MISS  (model wrote SQL in {out['ms']} ms)")
        self._trace("sql", state.sql)
        return "SQL generated."

    def _execute(self, state: _State) -> str:
        self._trace("tool", f"{self._exe.name} called")
        out = self._exe.run(state.sql)
        if out["ok"]:
            state.result = out
            what = f"{out['rowcount']} row(s) {'added' if out['kind'] == 'INSERT' else 'returned'}"
            self._trace("tool", f"{self._exe.name} -> guard OK ({out['kind']}), {what}, {out['ms']} ms")
            if state.from_cache:
                pass
            elif out["kind"] == "SELECT" and not out["rows"]:
                self._trace("cache", "not saved: empty result may mean the SQL is wrong")
            else:  # cache only SQL that actually worked
                self._cache.put(state.key, state.question, state.sql, self._llm.model)
                self._trace("cache", "saved to JSON cache")
            return f"OK, {what}."
        state.error = out["error"]
        if not out["retryable"]:  # e.g. DELETE/DROP: refuse now, do not waste model calls
            state.retries = self._max_retries + 1
            if state.from_cache:
                self._cache.delete(state.key)
            self._trace("tool", f"{self._exe.name} -> BLOCKED: {state.error}")
            return f"ERROR: {state.error}"
        state.retries += 1
        state.use_cache = False
        if state.from_cache:
            self._cache.delete(state.key)  # a cached statement that fails is a bad entry
        self._trace("tool", f"{self._exe.name} -> FAILED: {state.error}")
        if state.retries <= self._max_retries:
            self._trace("retry", f"asking generate_sql to fix it ({state.retries}/{self._max_retries})")
        return f"ERROR: {state.error}"

    # --- final answer ----------------------------------------------------

    def _answer(self, state: _State, step: int, on_token: Callable[[str], None]) -> AgentResult:
        result = state.result

        def done(text: str) -> AgentResult:
            return AgentResult(True, text, state.sql, step, state.from_cache)

        if result["kind"] == "INSERT":
            text = f"Saved {result['rowcount']} expense(s)."
            self._trace("tool", "compose_answer: no model needed for an INSERT")
            on_token(text)
            return done(text)
        if not result["rows"]:
            text = "No matching expenses were found."
            self._trace("tool", "compose_answer: no model needed for an empty result")
            on_token(text)
            return done(text)

        self._trace("tool", f"compose_answer called (model {self._llm.model})")
        messages = [{"role": "system", "content": ANSWER_PROMPT},
                    {"role": "user", "content": f"Question: {state.question}\nResult: {_rows_text(result)}"}]
        chunks: list[str] = []
        try:
            for chunk in self._llm.stream(messages, temperature=0.4, num_predict=200):
                chunks.append(chunk)
                on_token(chunk)
        except LLMError:
            if chunks:
                raise
        text = "".join(chunks).strip()
        if not text:  # model gave nothing: fall back to a plain table
            text = "\n".join(" | ".join(map(str, row)) for row in [result["columns"], *result["rows"]])
            on_token(text)
        return done(text)
