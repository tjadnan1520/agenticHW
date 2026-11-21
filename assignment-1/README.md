# Assignment 1: SQL Agent

A small agent that turns a plain-language question into SQL, runs it on SQLite and answers.
Model: `qwen2.5:3b` via Ollama. Python 3.10+, standard library only.

## Run
```
ollama pull qwen2.5:3b
python main.py                              # interactive
python main.py "show my last 5 expenses"    # one question
python main.py --quiet "..."                # answer only, no trace
python main.py --no-cache "..."             # skip the SQL cache
```

## Flow
1. The orchestrator asks the LLM which tool comes next.
2. `generate_sql` writes one SQL statement from the question and the live schema.
3. `execute_sql` checks the SQL with a guard and runs it.
4. The model writes a short answer from the rows.

The orchestrator hands the question and the SQL between the tools itself, so the tools never call each other.
Each step is printed:
```
[router] step 1: LLM chose generate_sql
[tool  ] generate_sql called
[cache ] MISS  (model wrote SQL in 9020 ms)
[sql   ] SELECT COALESCE(SUM(amount), 0) ...
[tool  ] execute_sql -> guard OK (SELECT), 1 row(s) returned
[tool  ] compose_answer called (model qwen2.5:3b)
[answer] You spent 730.0 BDT on food in the last 10 days.
```

## Database
SQLite file `data/sql_agent.db`, created on first run with 12 sample rows. One table, `expenses`:

| column | type | note |
|---|---|---|
| `id` | integer, auto | |
| `spent_on` | text `YYYY-MM-DD` | default today |
| `amount` | real, > 0 | BDT |
| `category` | text | food, transport, bills, shopping, health, education, entertainment, transfer, other |
| `note` | text | e.g. "lunch" |
| `created_at` | text | auto |

## Safety
- The guard allows one `SELECT` or `INSERT` only and refuses `DELETE`, `DROP`, `UPDATE`, `PRAGMA`, `ATTACH`.
- An SQLite authorizer allows only SELECT/INSERT on `expenses`, even if the guard misses something.
- SELECTs run on a read-only connection.

## Cache
`data/sql_cache.json` stores SQL that ran successfully, with `created_at`, `last_used_at` and `hits`.
The key is question + today's date + model + column layout.

## Files
`main.py` CLI, `sql_agent/` (`orchestrator`, `tools`, `db`, `guard`, `cache`, `llm`).

## Limits
A 3B model sometimes picks a wrong category, and each routing step takes about 3 seconds.
