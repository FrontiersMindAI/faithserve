from __future__ import annotations

import pytest

from faithserve.checks import CHECKS, FAIL, INCONCLUSIVE, PASS, SKIP
from faithserve.client import Client
from faithserve.reference import load_tokenizer
from fake_server import FakeServer


def run(tokenizer, faults=(), only=None, server_tokenizer=None):
    """Run checks against a fake server; returns {(check, case): Result}."""
    with FakeServer(server_tokenizer or tokenizer, faults) as server:
        client = Client(server.base_url, "fake-model", timeout=10)
        try:
            results = [r for name in (only or CHECKS) for r in CHECKS[name](client, tokenizer)]
        finally:
            client.close()
    return {(r.check, r.case): r for r in results}


def test_faithful_server_passes_everything(tokenizer):
    results = run(tokenizer)
    assert len(results) == 13
    assert {key: r.status for key, r in results.items() if r.status != PASS} == {}


ALL_SHAPES = ["single user turn", "system + user", "multi-turn", "unicode", "whitespace", "with tools"]

# fault -> check to run, rows expected to change (status, text expected in detail or hint)
FAILURE_MODES = {
    "extra_bos": ("parity", {case: (FAIL, "double BOS") for case in ALL_SHAPES}),
    "inject_system": (
        "parity",
        {
            **{case: (FAIL, "injects a default system prompt") for case in ALL_SHAPES},
            "system + user": (PASS, "prompt tokens on both sides"),
        },
    ),
    "drop_tools": ("parity", {"with tools": (FAIL, "not rendered into the prompt"), "unicode": (PASS, "")}),
    "ignore_max_tokens": ("sampling", {"max_tokens=5": (FAIL, "ignores or overrides max_tokens")}),
    "wrong_finish_reason": ("sampling", {"max_tokens=5": (FAIL, "expected 'length'")}),
    "ignore_stop": ("sampling", {"stop sequence": (FAIL, "ignores `stop`")}),
    "strip_stop_only": ("sampling", {"stop sequence": (FAIL, "without halting generation")}),
    "nondeterministic": (
        "sampling",
        {"temperature=0 repeatable": (FAIL, "different completions"), "stop sequence": (INCONCLUSIVE, "")},
    ),
    "ignore_seed": ("sampling", {"seed reproducible": (INCONCLUSIVE, "ignores `seed`")}),
    "reject_seed": ("sampling", {"seed reproducible": (SKIP, "does not accept `seed`")}),
    "stream_mismatch": ("streaming", {"stream vs non-stream": (FAIL, "content differs")}),
    "leak_tool_call": (
        "tools",
        {"tool call": (FAIL, "leaked into content"), "tool call (stream)": (FAIL, "leaked into content")},
    ),
    "swallow_tool_call": (
        "tools",
        {"tool call": (FAIL, "carries no tool_calls"), "tool call (stream)": (FAIL, "carries no tool_calls")},
    ),
    "stream_drops_tool_call": ("tools", {"tool call": (PASS, ""), "tool call (stream)": (FAIL, "streamed did not")}),
    "arguments_object": (
        "tools",
        {"tool call": (FAIL, "not a JSON-encoded string"), "tool call (stream)": (FAIL, "not a JSON-encoded string")},
    ),
}


@pytest.mark.parametrize("fault", sorted(FAILURE_MODES))
def test_failure_mode_is_detected(tokenizer, fault):
    check, expected = FAILURE_MODES[fault]
    results = run(tokenizer, [fault], only=[check])
    for case, (status, text) in expected.items():
        result = results[(check, case)]
        assert result.status == status, result
        assert text in result.detail + " " + result.hint, result
    # A fault must not spill over into rows it does not affect.
    unexpected = {k: r for k, r in results.items() if k[1] not in expected and r.status == FAIL}
    assert unexpected == {}


def test_nondeterministic_server_makes_streaming_inconclusive_not_failed(tokenizer):
    results = run(tokenizer, ["nondeterministic"], only=["streaming"])
    assert results[("streaming", "stream vs non-stream")].status == INCONCLUSIVE


def test_missing_usage_is_inconclusive_and_max_tokens_is_counted_locally(tokenizer):
    results = run(tokenizer, ["no_usage"])
    assert {r.status for k, r in results.items() if k[0] == "parity"} == {INCONCLUSIVE}
    assert results[("sampling", "max_tokens=5")].status == PASS
    assert "counted locally" in results[("sampling", "max_tokens=5")].detail
    assert results[("streaming", "stream vs non-stream")].status == PASS
    assert not any(r.status == FAIL for r in results.values())


def test_ignored_max_tokens_is_caught_without_usage(tokenizer):
    results = run(tokenizer, ["no_usage", "ignore_max_tokens"], only=["sampling"])
    assert results[("sampling", "max_tokens=5")].status == FAIL


def test_server_error_is_a_failure_with_the_status_shown(tokenizer):
    results = run(tokenizer, ["error_500"], only=["sampling", "streaming"])
    for case in ["max_tokens=5", "temperature=0 repeatable", "seed reproducible", "stream vs non-stream"]:
        result = next(r for (_, c), r in results.items() if c == case)
        assert result.status == FAIL and "HTTP 500" in result.detail, result
    assert results[("sampling", "stop sequence")].status == INCONCLUSIVE  # nothing to derive a stop sequence from


def test_template_without_tools_skips_tool_checks(tokenizer_dirs):
    tokenizer = load_tokenizer(tokenizer_dirs["no_tools"])
    results = run(tokenizer)
    assert results[("parity", "with tools")].status == SKIP
    assert results[("tools", "tool call")].status == SKIP
    assert results[("tools", "tool call (stream)")].status == SKIP
    assert not any(r.status == FAIL for r in results.values())


def test_bos_added_by_tokenizer_but_absent_from_template_is_accepted(tokenizer_dirs):
    """Template emits no BOS; a server that lets the tokenizer add it is not wrong."""
    tokenizer = load_tokenizer(tokenizer_dirs["no_bos"])
    results = run(tokenizer, ["extra_bos"], only=["parity"])
    assert all(r.status == PASS for r in results.values()), results
    assert "tokenizer's own special tokens" in results[("parity", "single user turn")].detail
    # ... and so is one that tokenizes the template output as-is.
    assert all(r.status == PASS for r in run(tokenizer, only=["parity"]).values())


def test_wrong_template_on_server_is_reported_as_template_mismatch(tokenizer, tokenizer_dirs):
    other = load_tokenizer(tokenizer_dirs["no_bos"])
    results = run(tokenizer, only=["parity"], server_tokenizer=other)
    assert all(r.status == FAIL for r in results.values())
    assert "BOS token the template emits is being dropped" in results[("parity", "unicode")].hint
