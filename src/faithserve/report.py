"""Terminal table and JSON report."""

from __future__ import annotations

import json
import textwrap
from dataclasses import asdict

from . import __version__
from .checks import FAIL, INCONCLUSIVE, PASS, SKIP, Result

_COLORS = {PASS: "32", FAIL: "31", SKIP: "90", INCONCLUSIVE: "33"}


def summarize(results: list[Result]) -> dict[str, int]:
    return {status: sum(r.status == status for r in results) for status in (PASS, FAIL, INCONCLUSIVE, SKIP)}


def format_report(results: list[Result], model: str, base_url: str, served_model: str, color: bool = False) -> str:
    def paint(status: str, text: str) -> str:
        return f"\033[{_COLORS[status]}m{text}\033[0m" if color else text

    check_w = max(len(r.check) for r in results)
    case_w = max(len(r.case) for r in results)
    status_w = len(INCONCLUSIVE)
    indent = check_w + case_w + status_w + 6
    lines = [
        f"faithserve {__version__}",
        f"reference: {model}",
        f"endpoint:  {base_url} (model {served_model!r})",
        "",
    ]
    for r in results:
        detail = textwrap.wrap(r.detail, width=max(40, 118 - indent)) or [""]
        status = paint(r.status, r.status.ljust(status_w))
        lines.append(f"{r.check.ljust(check_w)}  {r.case.ljust(case_w)}  {status}  {detail[0]}")
        lines.extend(" " * indent + more for more in detail[1:])

    hints = []
    for r in results:
        if r.status == FAIL and r.hint and (r.check, r.hint) not in hints:
            hints.append((r.check, r.hint))
    if hints:
        lines += ["", "What the failures usually mean:"]
        for check, hint in hints:
            lines.append(textwrap.fill(f"{check}: {hint}", width=118, initial_indent="  ", subsequent_indent="    "))

    counts = summarize(results)
    verdict = paint(FAIL, "FAIL") if counts[FAIL] else paint(PASS, "PASS")
    lines += [
        "",
        f"{verdict}: {counts[PASS]} passed, {counts[FAIL]} failed, "
        f"{counts[INCONCLUSIVE]} inconclusive, {counts[SKIP]} skipped",
    ]
    return "\n".join(lines)


def write_json(path: str, results: list[Result], model: str, base_url: str, served_model: str, when: str) -> None:
    report = {
        "faithserve_version": __version__,
        "timestamp": when,
        "reference_model": model,
        "base_url": base_url,
        "served_model": served_model,
        "summary": {k.lower(): v for k, v in summarize(results).items()},
        "results": [asdict(r) for r in results],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
