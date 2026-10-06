"""Command line entry point: ``faithserve check <hf-model-id> --base-url ...``"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

from . import __version__
from .checks import CHECKS, FAIL
from .client import APIError, Client
from .reference import TokenizerError, load_tokenizer
from .report import format_report, write_json

EXIT_OK, EXIT_FAILED, EXIT_ERROR = 0, 1, 2


def _check_names(value: str) -> list[str]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = [name for name in names if name not in CHECKS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown check {unknown[0]!r} (choose from {', '.join(CHECKS)})")
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="faithserve", description="Is this endpoint serving this open-weight model faithfully?"
    )
    parser.add_argument("--version", action="version", version=f"faithserve {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser(
        "check",
        help="run the checks against an OpenAI-compatible endpoint",
        description="Compare an OpenAI-compatible chat-completions endpoint with the model's reference "
        "tokenizer and chat template. Exit code: 0 no failures, 1 at least one check failed, 2 could not run.",
    )
    check.add_argument("model", help="Hugging Face model id (or local path) of the reference tokenizer")
    check.add_argument("--base-url", required=True, help="endpoint base URL, e.g. http://localhost:8000/v1")
    check.add_argument("--api-key", help="API key; prefer the FAITHSERVE_API_KEY environment variable")
    check.add_argument("--served-model", help="model name the server expects (default: the Hugging Face id)")
    check.add_argument("--json", metavar="PATH", help="also write the report as JSON to PATH")
    check.add_argument("--only", type=_check_names, metavar="CHECKS", help=f"comma-separated: {', '.join(CHECKS)}")
    check.add_argument("--skip", type=_check_names, metavar="CHECKS", help="comma-separated checks to leave out")
    check.add_argument("--timeout", type=float, default=120.0, help="per-request timeout in seconds (default 120)")
    return parser


def _fail(message: str) -> int:
    print(f"faithserve: error: {message}", file=sys.stderr)
    return EXIT_ERROR


def _explain_preflight(err: APIError, client: Client, served_model_given: bool) -> str:
    message = f"the endpoint did not answer a minimal chat request: {err.message}"
    if err.transient:
        return f"{message}\n  Is the server running, and is --base-url ({client.base_url}) correct?"
    if err.status in (401, 403):
        return f"{message}\n  Pass an API key with --api-key or FAITHSERVE_API_KEY."
    available = client.list_models()
    if available and client.model not in available:
        shown = ", ".join(available[:8]) + (", ..." if len(available) > 8 else "")
        return f"{message}\n  The server does not list {client.model!r}. Try --served-model with one of: {shown}"
    if err.status == 404 and not client.base_url.endswith("/v1"):
        return f"{message}\n  Most servers expect the base URL to end in /v1."
    if not served_model_given:
        return f"{message}\n  If the server knows the model under another name, pass --served-model."
    return message


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected = [name for name in (args.only or CHECKS) if name not in (args.skip or [])]
    if not selected:
        return _fail("--only and --skip leave no checks to run")

    print(f"Loading reference tokenizer for {args.model} ...", file=sys.stderr)
    try:
        tokenizer = load_tokenizer(args.model)
    except TokenizerError as exc:
        return _fail(str(exc))

    served_model = args.served_model or args.model
    api_key = args.api_key or os.environ.get("FAITHSERVE_API_KEY")
    client = Client(args.base_url, served_model, api_key=api_key, timeout=args.timeout)
    try:
        try:
            client.chat([{"role": "user", "content": "Hi"}], max_tokens=1)
        except APIError as err:
            return _fail(_explain_preflight(err, client, bool(args.served_model)))

        results = []
        for name in selected:
            print(f"Running {name} check ...", file=sys.stderr)
            results.extend(CHECKS[name](client, tokenizer))
    except KeyboardInterrupt:
        return _fail("interrupted")
    finally:
        client.close()

    color = sys.stdout.isatty() and "NO_COLOR" not in os.environ
    print(format_report(results, args.model, client.base_url, served_model, color=color))
    if args.json:
        when = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            write_json(args.json, results, args.model, client.base_url, served_model, when)
        except OSError as exc:
            return _fail(f"could not write {args.json}: {exc}")
    return EXIT_FAILED if any(r.status == FAIL for r in results) else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
