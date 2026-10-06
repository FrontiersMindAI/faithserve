from __future__ import annotations

import json

from faithserve.cli import main
from fake_server import FakeServer

SECRET = "sk-test-secret-123"


def test_exit_zero_and_json_report_for_faithful_server(tokenizer, tokenizer_dirs, tmp_path, capsys):
    report = tmp_path / "report.json"
    with FakeServer(tokenizer, api_key=SECRET) as server:
        code = main(
            ["check", tokenizer_dirs["default"], "--base-url", server.base_url, "--served-model", "fake-model"]
            + ["--api-key", SECRET, "--json", str(report)]
        )
        assert set(server.seen_authorization) == {f"Bearer {SECRET}"}
    out, err = capsys.readouterr()
    assert code == 0
    assert "PASS: 13 passed, 0 failed" in out
    data = json.loads(report.read_text())
    assert data["summary"] == {"pass": 13, "fail": 0, "inconclusive": 0, "skip": 0}
    assert len(data["results"]) == 13
    # The API key must never be printed or written.
    assert SECRET not in out + err + report.read_text()


def test_api_key_from_environment(tokenizer, tokenizer_dirs, monkeypatch, capsys):
    monkeypatch.setenv("FAITHSERVE_API_KEY", SECRET)
    with FakeServer(tokenizer, api_key=SECRET) as server:
        args = ["check", tokenizer_dirs["default"], "--base-url", server.base_url, "--served-model", "fake-model"]
        assert main([*args, "--only", "streaming"]) == 0
    out, err = capsys.readouterr()
    assert SECRET not in out + err


def test_exit_one_when_a_check_fails(tokenizer, tokenizer_dirs, capsys):
    with FakeServer(tokenizer, ["extra_bos"]) as server:
        args = ["check", tokenizer_dirs["default"], "--base-url", server.base_url, "--served-model", "fake-model"]
        code = main([*args, "--only", "parity"])
    out, _ = capsys.readouterr()
    assert code == 1
    assert "double BOS" in out
    assert "sampling" not in out


def test_skip_leaves_checks_out(tokenizer, tokenizer_dirs, capsys):
    with FakeServer(tokenizer, ["extra_bos"]) as server:
        args = ["check", tokenizer_dirs["default"], "--base-url", server.base_url, "--served-model", "fake-model"]
        assert main([*args, "--skip", "parity,tools"]) == 0
    out, _ = capsys.readouterr()
    assert "parity" not in out and "streaming" in out


def test_unreachable_endpoint_is_a_clean_error(tokenizer_dirs, capsys):
    code = main(["check", tokenizer_dirs["default"], "--base-url", "http://127.0.0.1:9/v1", "--timeout", "2"])
    _, err = capsys.readouterr()
    assert code == 2
    assert "Is the server running" in err and "Traceback" not in err


def test_wrong_model_name_suggests_served_models(tokenizer, tokenizer_dirs, capsys):
    with FakeServer(tokenizer) as server:
        code = main(["check", tokenizer_dirs["default"], "--base-url", server.base_url])
    _, err = capsys.readouterr()
    assert code == 2
    assert "--served-model" in err and "fake-model" in err


def test_wrong_api_key_is_a_clean_error(tokenizer, tokenizer_dirs, capsys):
    with FakeServer(tokenizer, api_key=SECRET) as server:
        args = ["check", tokenizer_dirs["default"], "--base-url", server.base_url, "--served-model", "fake-model"]
        code = main([*args, "--api-key", "wrong-key-456"])
    out, err = capsys.readouterr()
    assert code == 2
    assert "FAITHSERVE_API_KEY" in err and "wrong-key-456" not in out + err


def test_missing_v1_suffix_gets_a_hint(tokenizer, tokenizer_dirs, capsys):
    with FakeServer(tokenizer) as server:
        base = server.base_url[: -len("/v1")]
        code = main(["check", tokenizer_dirs["default"], "--base-url", base, "--served-model", "fake-model"])
    _, err = capsys.readouterr()
    assert code == 2
    assert "end in /v1" in err


def test_missing_reference_model_is_a_clean_error(tmp_path, capsys):
    code = main(["check", str(tmp_path / "no-such-model"), "--base-url", "http://127.0.0.1:9/v1"])
    _, err = capsys.readouterr()
    assert code == 2
    assert err.count("\n") <= 3 and "Traceback" not in err
