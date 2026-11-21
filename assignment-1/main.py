"""SQL agent CLI.   python main.py            (interactive)
                    python main.py "food last 10 days"   (one question)"""

import argparse
import os
import sys
from pathlib import Path

from sql_agent.cache import SqlCache
from sql_agent.db import Database
from sql_agent.llm import Ollama
from sql_agent.orchestrator import Orchestrator
from sql_agent.tools import ExecuteSql, GenerateSql

ROOT = Path(__file__).resolve().parent
TAG_WIDTH = 6


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")  # Bangla text on Windows consoles

    parser = argparse.ArgumentParser(description="Ask questions about your expenses in plain language.")
    parser.add_argument("question", nargs="?", help="one question; omit for interactive mode")
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "qwen2.5:3b"))
    parser.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "sql_agent.db")
    parser.add_argument("--cache", type=Path, default=ROOT / "data" / "sql_cache.json")
    parser.add_argument("--no-cache", action="store_true", help="always ask the model for SQL")
    parser.add_argument("--quiet", action="store_true", help="hide the step-by-step trace")
    args = parser.parse_args(argv)

    db = Database(args.db)
    db.init()
    cache = SqlCache(args.cache, enabled=not args.no_cache)
    llm = Ollama(args.model, args.host)

    def trace(tag: str, message: str) -> None:
        if not args.quiet:
            print(f"[{tag:<{TAG_WIDTH}}] {message}", flush=True)

    agent = Orchestrator(llm, GenerateSql(llm, db, cache), ExecuteSql(db), cache, trace)

    def ask(question: str) -> None:
        started = {"done": False}

        def on_token(chunk: str) -> None:
            if not started["done"]:
                print(f"[{'answer':<{TAG_WIDTH}}] ", end="")
                started["done"] = True
            print(chunk, end="", flush=True)

        result = agent.run(question, on_token)
        if started["done"]:
            print()
        elif not result.ok:
            print(f"[{'error':<{TAG_WIDTH}}] {result.answer}")

    if args.question:
        ask(args.question)
        return 0

    print(f"SQL agent ready (model {args.model}, db {args.db.name}). Type 'exit' to quit.")
    while True:
        try:
            question = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.lower() in {"exit", "quit"}:
            return 0
        if question:
            ask(question)


if __name__ == "__main__":
    raise SystemExit(main())
