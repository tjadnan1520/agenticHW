"""Contract test for schema v1 (Lecture 2, slides 30-31).

Policy: consumers ignore unknown fields. So a change is compatible (additive) when every
document a v1 consumer relies on still means the same thing:
  - every v1 property still exists, keeps its type (it may narrow, never widen),
    keeps its enum values (no new values: old code has no branch for them) and its const,
  - every v1 required field stays required,
  - new fields are optional,
  - schema_version keeps the same major version (1.x).
Anything else is breaking.
"""
import copy
import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

V1 = json.loads((Path(__file__).resolve().parents[1] / "schema" / "notice_v1.schema.json")
                .read_text())


def _types(prop):
    t = prop.get("type")
    return {t} if isinstance(t, str) else set(t or [])


def breaking_changes(old, new):
    """Return a list of reasons `new` breaks consumers of `old`; empty means compatible."""
    problems = []
    old_p, new_p = old["properties"], new.get("properties", {})
    for name, op in old_p.items():
        if name not in new_p:
            problems.append(f"{name}: removed or renamed")
            continue
        np_ = new_p[name]
        if not _types(np_) <= _types(op):
            problems.append(f"{name}: type changed {sorted(_types(op))} -> {sorted(_types(np_))}")
        if "enum" in op and not set(np_.get("enum", [])) <= set(op["enum"]):
            problems.append(f"{name}: new enum values {set(np_['enum']) - set(op['enum'])}")
        if name == "schema_version":
            if str(np_.get("const", "")).split(".")[0] != str(op["const"]).split(".")[0]:
                problems.append("schema_version: major version changed")
        elif op.get("const") != np_.get("const"):
            problems.append(f"{name}: const changed")
    for name in set(old.get("required", [])) - set(new.get("required", [])):
        problems.append(f"{name}: no longer required")
    for name in set(new.get("required", [])) - set(old_p):
        problems.append(f"{name}: new field is required")
    return problems


V1_DOC = {"schema_version": "1.0", "domain": "example.com", "notice_type": "expiry",
          "expires_on": "2026-12-20", "registrar": "Namewell", "amount_due": 13.99,
          "currency": "USD", "action_required": True}


class ContractTest(unittest.TestCase):
    def test_v1_is_valid_and_versioned(self):
        Draft202012Validator.check_schema(V1)
        self.assertEqual(V1["properties"]["schema_version"]["const"], "1.0")
        Draft202012Validator(V1).validate(V1_DOC)

    def test_accepts_additive_change(self):
        v1_1 = copy.deepcopy(V1)
        v1_1["properties"]["schema_version"]["const"] = "1.1"
        v1_1["properties"]["registrar_url"] = {"type": ["string", "null"]}  # optional
        self.assertEqual(breaking_changes(V1, v1_1), [])
        # a v1 consumer still accepts a v1.1 document: the unknown field is ignored
        doc = {**V1_DOC, "registrar_url": "https://namewell.example"}
        Draft202012Validator(V1).validate({**doc, "schema_version": "1.0"})

    def test_fails_breaking_changes(self):
        renamed = copy.deepcopy(V1)
        renamed["properties"]["hostname"] = renamed["properties"].pop("domain")
        renamed["required"] = ["hostname" if r == "domain" else r for r in renamed["required"]]

        retyped = copy.deepcopy(V1)
        retyped["properties"]["expires_on"] = {"type": "object"}

        new_enum = copy.deepcopy(V1)
        new_enum["properties"]["notice_type"]["enum"].append("dns_change")

        for name, schema in [("rename domain", renamed), ("expires_on to object", retyped),
                             ("new enum value", new_enum)]:
            with self.subTest(name):
                self.assertNotEqual(breaking_changes(V1, schema), [])
        # and the v1 consumer really rejects a document from the renamed schema
        doc = {k if k != "domain" else "hostname": v for k, v in V1_DOC.items()}
        self.assertFalse(Draft202012Validator(V1).is_valid(doc))


if __name__ == "__main__":
    unittest.main()
