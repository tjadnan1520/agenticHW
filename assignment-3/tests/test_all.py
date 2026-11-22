import json
import unittest
from unittest import mock

import cache_from_logs
import kv_cache
import llm
import score
from score import KEY, field_equal, parse, score_record

GOOD = {"schema_version": "1.0", "domain": "rahmantraders.com", "notice_type": "expiry",
        "expires_on": "2027-01-14", "registrar": "Namewell", "amount_due": 13.99,
        "currency": "USD", "action_required": True}


def rec(obj_or_raw, nid="n01", done="stop"):
    raw = obj_or_raw if isinstance(obj_or_raw, str) else json.dumps(obj_or_raw)
    return {"id": nid, "raw": raw, "done_reason": done}


class TestAnswerKey(unittest.TestCase):
    def test_key_matches_schema_and_notices(self):
        notices = json.load(open("data/notices.json", encoding="utf-8"))
        self.assertEqual(len(notices), 50)
        self.assertEqual({n["id"] for n in notices}, set(KEY))
        for nid, k in KEY.items():
            self.assertEqual(list(score.VALIDATOR.iter_errors({x: v for x, v in k.items()
                                                               if x != "_note"})), [], nid)


class TestScore(unittest.TestCase):
    def test_perfect(self):
        s = score_record(rec(GOOD))
        self.assertTrue(s["valid"] and all(s["correct"].values()))
        self.assertEqual(len(s["correct"]), 7)

    def test_invalid_json_scores_zero(self):
        s = score_record(rec("not json at all"))
        self.assertFalse(s["json_valid"] or s["valid"])
        self.assertFalse(any(s["correct"].values()))

    def test_schema_invalid_currency_symbol(self):
        s = score_record(rec({**GOOD, "currency": "$"}))
        self.assertTrue(s["json_valid"])
        self.assertFalse(s["valid"])
        self.assertFalse(any(s["correct"].values()))

    def test_missing_required_key_invalid(self):
        bad = {k: v for k, v in GOOD.items() if k != "action_required"}
        self.assertFalse(score_record(rec(bad))["valid"])

    def test_truncated_is_invalid(self):
        self.assertFalse(score_record(rec(GOOD, done="length"))["valid"])

    def test_null_values(self):
        self.assertTrue(field_equal("amount_due", None, None))
        self.assertFalse(field_equal("amount_due", 0, None))
        self.assertFalse(field_equal("expires_on", None, "2027-01-14"))
        s = score_record(rec({**GOOD, "expires_on": None}))
        self.assertTrue(s["valid"] and not s["correct"]["expires_on"])

    def test_extra_field_ignored_but_counted(self):
        s = score_record(rec({**GOOD, "note": "x"}))
        self.assertTrue(s["valid"] and all(s["correct"].values()))
        self.assertEqual(s["extra_keys"], 1)

    def test_field_rules(self):
        self.assertTrue(field_equal("amount_due", 13.994, 13.99))
        self.assertFalse(field_equal("amount_due", "13.99", 13.99))
        self.assertTrue(field_equal("registrar", " namewell ", "Namewell"))
        self.assertFalse(field_equal("action_required", 1, True))
        self.assertFalse(field_equal("expires_on", "2027-14-01", "2027-01-14"))

    def test_fence_forgiven(self):
        self.assertEqual(parse("```json\n{\"a\": 1}\n```"), {"a": 1})
        self.assertIsNone(parse("[1, 2]"))

    def test_date_and_currency_pattern(self):
        self.assertFalse(score_record(rec({**GOOD, "expires_on": "14/01/2027"}))["valid"])
        self.assertFalse(score_record(rec({**GOOD, "currency": "usd"}))["valid"])


class TestKvCache(unittest.TestCase):
    def test_numbers(self):
        self.assertEqual(kv_cache.HEAD_DIM, 128)
        self.assertEqual(kv_cache.bytes_per_token(), 57_344)
        self.assertEqual(kv_cache.kv_cache_bytes(8 * 1024), 448 * 2**20)
        self.assertEqual(kv_cache.kv_cache_bytes(32 * 1024), 1792 * 2**20)  # 1.75 GiB
        self.assertEqual(kv_cache.kv_cache_bytes(32 * 1024, kv_heads=28), 7 * 1792 * 2**20)


class TestLog(unittest.TestCase):
    LOG = ["slot x | task 0 | new prompt, n_ctx_slot = 4096, n_keep = 4, task.n_tokens = 509",
           "slot x | task 0 | cached n_tokens = 0, memory_seq_rm [0, end)",
           "slot x | task 1 | new prompt, n_ctx_slot = 4096, n_keep = 4, task.n_tokens = 510",
           "slot x | task 1 | cached n_tokens = 447, memory_seq_rm [447, end)",
           "slot x | task 2 | new prompt, n_ctx_slot = 8192, n_keep = 4, task.n_tokens = 99",
           "slot x | task 2 | cached n_tokens = 5, memory_seq_rm [5, end)"]

    def test_parse(self):
        calls = cache_from_logs.parse_log(self.LOG)
        self.assertEqual([(c["n_tokens"], c["cached_tokens"]) for c in calls], [(509, 0), (510, 447)])


class TestClient(unittest.TestCase):
    def _fake(self, captured):
        class Resp:
            def __enter__(s): return s
            def __exit__(s, *a): return False
            def read(s): return json.dumps({"message": {"content": "{}"}, "prompt_eval_count": 500,
                                            "eval_count": 100, "done_reason": "stop"}).encode()
        def urlopen(req, timeout=0):
            captured.update(json.loads(req.data))
            return Resp()
        return urlopen

    def test_format_only_sent_when_given(self):
        cap = {}
        with mock.patch("llm.urllib.request.urlopen", self._fake(cap)):
            c = llm.OllamaChat("m", seed=1)
            c.chat([{"role": "user", "content": "x"}])
            self.assertNotIn("format", cap)
            c.chat([{"role": "user", "content": "x"}], format={"type": "object"})
            self.assertEqual(cap["format"], {"type": "object"})
            self.assertEqual(cap["options"], {"seed": 1})
        self.assertEqual(len(c.ledger), 2)

    def test_cost(self):
        self.assertAlmostEqual(llm.notional_cost(1_000_000, 0), 2.0)
        self.assertAlmostEqual(llm.notional_cost(1_000_000, 0, 1_000_000), 0.5)
        self.assertAlmostEqual(llm.notional_cost(0, 1_000_000), 8.0)


if __name__ == "__main__":
    unittest.main()
