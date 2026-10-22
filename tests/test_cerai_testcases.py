"""The CeRAI test-case file must stay importable by their tool: same shape, known IDs, no gaps a strategy would trip on."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("build_cerai_testcases", ROOT / "scripts" / "build_cerai_testcases.py")
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)

ALLOWED = {"PROMPT_ID", "SYSTEM_PROMPT", "PROMPT", "EXPECTED_OUTPUT", "DOMAIN", "LANGUAGE", "STRATEGY", "LLM_AS_JUDGE", "SUB_METRIC"}


def test_no_problems_in_the_case_table():
    assert build.REPORT.exists(), "the frozen report the cases were written from must be committed beside them"
    assert build.validate() == []


def test_every_case_is_in_the_tools_shape():
    seen = set()
    for metric, cases in build.CASES.items():
        assert metric in build.METRICS
        for c in build.public(cases):
            assert set(c) <= ALLOWED and {"PROMPT_ID", "SYSTEM_PROMPT", "PROMPT"} <= set(c)
            assert c["PROMPT_ID"] not in seen
            seen.add(c["PROMPT_ID"])
            assert isinstance(c["STRATEGY"], list) and len(c["STRATEGY"]) == 1 and c["STRATEGY"][0] in build.STRATEGIES
            assert c["LANGUAGE"] == "english"


def test_judge_and_reference_strategies_always_have_what_they_read():
    for cases in build.CASES.values():
        for c in cases:
            sid = c["STRATEGY"][0]
            if sid in build.NEEDS_EXPECTED:
                assert c["EXPECTED_OUTPUT"].strip(), c["PROMPT_ID"]
            if sid in build.NEEDS_JUDGE_TEXT:
                assert c["LLM_AS_JUDGE"] != "No", c["PROMPT_ID"]


def test_prompts_fit_the_services_message_limit():
    from api.server import MAX_CONTENT_CHARS

    assert all(len(c["PROMPT"]) <= MAX_CONTENT_CHARS for cases in build.CASES.values() for c in cases)


def test_committed_file_matches_the_generator():
    path = ROOT / "eval" / "cerai" / "rca_testcases.json"
    if not path.exists():
        pytest.skip("run scripts/build_cerai_testcases.py first")
    expected = {m: {"cases": build.public(cs)} for m, cs in build.CASES.items()}
    assert json.loads(path.read_text(encoding="utf-8")) == expected
