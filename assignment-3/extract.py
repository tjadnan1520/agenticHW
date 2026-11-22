"""Extract the 50 notices with two methods (prompt-only vs grammar-constrained), N runs each.

Both methods use the same model, prompt, temperature and token limit; run k uses seed k for both.
Writes runs/<method>_run<k>.jsonl (raw model output + ledger). Resumable.

    python extract.py --runs 1 --limit 3   # smoke test
    python extract.py                      # full: 2 methods x 3 runs x 50 notices = 300 calls
"""
import argparse
import json
from pathlib import Path

from llm import OllamaChat

HERE = Path(__file__).resolve().parent
SCHEMA = json.loads((HERE / "schema" / "notice_v1.schema.json").read_text(encoding="utf-8"))
NOTICES = json.loads((HERE / "data" / "notices.json").read_text(encoding="utf-8"))
RUNS_DIR = HERE / "runs"
METHODS = ("prompt", "grammar")

SYSTEM_PROMPT = """You extract one domain-name notice into JSON.
Reply with a single JSON object and nothing else, with exactly these keys:
- "schema_version": always the string "1.0".
- "domain": the registered domain name in lowercase, without "www.", subdomains, "http://" or a path.
- "notice_type": one of "expiry", "renewal", "transfer", "suspension", "payment", "other".
    expiry = the domain will expire or has expired; renewal = the domain was renewed;
    transfer = moving the domain between registrars (requests, auth codes, status);
    suspension = the domain is, will be, or was suspended (including reactivation);
    payment = an invoice, failed payment, payment received or refund;
    other = anything else. If the domain is or will be suspended, use "suspension";
    otherwise if a payment failed or an invoice is due, use "payment".
- "expires_on": the domain's registration expiry date as "YYYY-MM-DD", or null if the notice
    does not state a full date. For a renewal or transfer, use the new expiry date. Do not use
    sent dates, due dates, suspension dates or certificate dates. Never invent a date.
- "registrar": the registrar that sent the notice, as named (company name only); if the sender is
    not a registrar, the registrar named in the notice; otherwise null.
- "amount_due": the amount the recipient must pay or will be charged, as a number; null if nothing
    is due or the payment is already complete (receipts, refunds).
- "currency": ISO 4217 code of amount_due ("$" is "USD", "Tk" or "৳" is "BDT", "€" is "EUR",
    "£" is "GBP", "Rs" is "INR"); null when amount_due is null.
- "action_required": true if the recipient is asked to do something; false if the notice is
    informational or the only action is conditional ("if you did not request this...")."""

SETTINGS = {"model": "qwen2.5:3b", "temperature": 0.3, "num_predict": 256, "num_ctx": 4096}


def extract_one(llm, notice, method):
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": notice["text"]}]
    raw = llm.chat(messages, format=SCHEMA if method == "grammar" else None)
    return {"id": notice["id"], "method": method, "raw": raw, **llm.ledger[-1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--limit", type=int, help="only the first N notices")
    ap.add_argument("--out", default="runs", help="output folder")
    args = ap.parse_args()
    out_dir = HERE / args.out
    out_dir.mkdir(exist_ok=True)
    notices = NOTICES[: args.limit] if args.limit else NOTICES

    for run in range(1, args.runs + 1):
        for method in METHODS:
            llm = OllamaChat(SETTINGS["model"], temperature=SETTINGS["temperature"], seed=run,
                             num_ctx=SETTINGS["num_ctx"], num_predict=SETTINGS["num_predict"])
            path = out_dir / f"{method}_run{run}.jsonl"
            done = set()
            if path.exists():
                done = {json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines() if l}
            with path.open("a", encoding="utf-8") as f:
                for n in notices:
                    if n["id"] in done:
                        continue
                    rec = {"run": run, "seed": run, **extract_one(llm, n, method)}
                    f.write(json.dumps(rec, ensure_ascii=False) + chr(10))
                    f.flush()
                    print(f"run {run} {method:7} {n['id']} {rec['seconds']:5.1f}s {rec['done_reason']}",
                          flush=True)


if __name__ == "__main__":
    main()
