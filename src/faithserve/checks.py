"""The four checks. Each takes a Client and the reference tokenizer and returns Results.

Statuses:
  PASS          the endpoint did what the reference says it should
  FAIL          the endpoint did something a faithful server would not do
  SKIP          not applicable (template has no tool support, server rejects the feature)
  INCONCLUSIVE  could not tell from the outside; never counted as a failure
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .client import APIError, Client, Completion
from .reference import TokenizerError, count_tokens, render, supports_tools

PASS, FAIL, SKIP, INCONCLUSIVE = "PASS", "FAIL", "SKIP", "INCONCLUSIVE"


@dataclass
class Result:
    check: str  # parity | sampling | streaming | tools
    case: str
    status: str
    detail: str
    hint: str = ""  # for failures: what it usually means


def _from_error(check: str, case: str, err: APIError) -> Result:
    """Turn a failed request into a result without guessing more than we know."""
    if err.transient:
        return Result(check, case, INCONCLUSIVE, err.message)
    if err.rejected:
        return Result(check, case, SKIP, f"server rejected the request ({err.message})")
    return Result(
        check, case, FAIL, err.message, "The server errored on a standard chat-completions request; see its logs."
    )


def _short(text: str, limit: int = 60) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


# --------------------------------------------------------------------------- shared fixtures

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "Name of the city"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}
TOOL_MESSAGES = [{"role": "user", "content": "What is the weather in Paris right now? Use the get_weather tool."}]
GEN_MESSAGES = [{"role": "user", "content": "List twelve different animals, one per line, with no other text."}]
SEED_MESSAGES = [{"role": "user", "content": "Write one short, unusual sentence about the sea."}]

SYSTEM = {"role": "system", "content": "You are a terse assistant. Answer in one sentence."}
PARITY_SHAPES = [
    ("single user turn", [{"role": "user", "content": "What is the capital of France?"}]),
    ("system + user", [SYSTEM, {"role": "user", "content": "What is the capital of France?"}]),
    (
        "multi-turn",
        [
            {"role": "user", "content": "Pick a number between 1 and 10."},
            {"role": "assistant", "content": "7"},
            {"role": "user", "content": "Double it."},
            {"role": "assistant", "content": "14"},
            {"role": "user", "content": "Now subtract 3."},
        ],
    ),
    ("unicode", [{"role": "user", "content": "Übersetze: 日本語のテキスト, naïve café, emoji 🙂👍🏽, עברית, x²≠y"}]),
    ("whitespace", [{"role": "user", "content": "  leading spaces\n\n\ttabbed line\ntrailing spaces and newline  \n"}]),
]


# --------------------------------------------------------------------------- 1. prompt-token parity


def check_parity(client: Client, tokenizer) -> list[Result]:
    """Does the server build the same prompt as the model's own chat template?"""
    shapes = [(case, messages, None) for case, messages in PARITY_SHAPES]
    results: list[Result] = []
    if supports_tools(tokenizer, TOOL_MESSAGES, [WEATHER_TOOL]):
        shapes.append(("with tools", TOOL_MESSAGES, [WEATHER_TOOL]))
    else:
        results.append(Result("parity", "with tools", SKIP, "the model's chat template does not render tools"))

    observed = []  # mismatches and matches, for diagnosis
    for case, messages, tools in shapes:
        try:
            ref = render(tokenizer, messages, tools)
        except TokenizerError as exc:
            results.append(Result("parity", case, SKIP, f"the reference template rejects this shape ({exc})"))
            continue
        params = {"max_tokens": 1, "temperature": 0}
        if tools:
            params["tools"] = tools
        try:
            got = client.chat(messages, **params)
        except APIError as err:
            results.append(_from_error("parity", case, err))
            continue
        if got.prompt_tokens is None:
            results.append(
                Result("parity", case, INCONCLUSIVE, "server did not report usage.prompt_tokens; nothing to compare")
            )
            continue

        delta = got.prompt_tokens - ref.tokens
        entry = {"case": case, "delta": delta, "bos": ref.starts_with_bos, "server": got.prompt_tokens}
        if tools:
            entry["without_tools"] = render(tokenizer, messages).tokens
        observed.append(entry)
        if delta == 0:
            results.append(Result("parity", case, PASS, f"{ref.tokens} prompt tokens on both sides"))
        elif not ref.starts_with_bos and got.prompt_tokens == ref.tokens_with_special:
            # The template emits no BOS but the tokenizer's own post-processor adds special tokens.
            # Both conventions are in use for such models, so this is not treated as a mismatch.
            entry["delta"] = 0
            results.append(
                Result(
                    "parity",
                    case,
                    PASS,
                    f"{got.prompt_tokens} prompt tokens; matches the template plus the tokenizer's own special "
                    f"tokens ({delta:+d}), which the template itself does not emit",
                )
            )
        else:
            results.append(
                Result("parity", case, FAIL, f"reference {ref.tokens} tokens, server {got.prompt_tokens} ({delta:+d})")
            )

    hint = _diagnose_parity(observed)
    for result in results:
        if result.status == FAIL and not result.hint:
            result.hint = hint
    return results


def _diagnose_parity(observed: list[dict]) -> str:
    """Name the common mismatch patterns. Deltas are server minus reference."""
    bad = [o for o in observed if o["delta"] != 0]
    if not bad:
        return ""
    plain = [o for o in observed if "without_tools" not in o]
    plain_deltas = {o["delta"] for o in plain}
    bos = observed[0]["bos"]

    if len(plain) > 1 and len(plain_deltas) == 1 and plain_deltas != {0}:
        delta = plain_deltas.pop()
        if delta == 1 and bos:
            return (
                "Exactly +1 token on every shape: the classic double BOS. The chat template already emits the "
                "BOS token and the server adds a second one when tokenizing."
            )
        if delta == -1 and bos:
            return "Exactly -1 token on every shape: the BOS token the template emits is being dropped."
        if abs(delta) == 1:
            return f"Exactly {delta:+d} token on every shape: one special token is being added or dropped."
        return (
            f"A constant {delta:+d} tokens on every shape: the server wraps every prompt differently from the "
            "reference template (a different or outdated chat template)."
        )

    by_case = {o["case"]: o["delta"] for o in observed}
    others = [d for case, d in by_case.items() if case not in ("system + user", "with tools")]
    if others and all(d > 1 for d in others) and by_case.get("system + user", 1) <= 0:
        return (
            "Prompts without a system message come back longer, but not when one is supplied: the server "
            "injects a default system prompt that the reference template does not."
        )
    if by_case.get("system + user", 0) < 0 and all(d == 0 for d in others):
        return "Only the shape with a system message is shorter: the system prompt is dropped or altered."

    if all("without_tools" in o for o in bad):
        tools_case = bad[0]
        if tools_case["server"] == tools_case["without_tools"]:
            return (
                "The server's prompt has the same length as the reference prompt without tools: the tool "
                "definitions are not rendered into the prompt at all."
            )
        return (
            "Only the shape with tools differs: tool definitions are rendered differently from the reference "
            "template (different JSON formatting, a different tool prompt, or tools moved elsewhere)."
        )
    if {o["case"] for o in bad} <= {"unicode", "whitespace"}:
        return "Only the unicode/whitespace shapes differ: the server normalises or strips message content."
    return (
        "The difference varies by conversation shape: the server is using a different chat template from the "
        "one shipped with the model."
    )


# --------------------------------------------------------------------------- 2. sampling parameters


def check_sampling(client: Client, tokenizer) -> list[Result]:
    """Are max_tokens, stop, temperature=0 and seed honoured?"""
    results = [_max_tokens(client, tokenizer)]
    determinism, baseline = _temperature_zero(client)
    results.append(determinism)
    results.append(_stop(client, baseline, deterministic=determinism.status == PASS))
    results.append(_seed(client))
    return results


def _max_tokens(client: Client, tokenizer, limit: int = 5) -> Result:
    case = f"max_tokens={limit}"
    try:
        got = client.chat(GEN_MESSAGES, max_tokens=limit, temperature=0)
    except APIError as err:
        return _from_error("sampling", case, err)
    exact = got.completion_tokens is not None
    n = got.completion_tokens if exact else count_tokens(tokenizer, got.text)
    counted = f"{n} completion tokens" if exact else f"about {n} tokens (counted locally; no usage reported)"
    # Local re-tokenization of the text can be off by a token or two, so allow slack when not exact.
    if n > limit + (0 if exact else 2):
        return Result(
            "sampling",
            case,
            FAIL,
            f"got {counted}, finish_reason={got.finish_reason!r}",
            "The server ignores or overrides max_tokens (wrong parameter mapping, or a server-side default wins).",
        )
    if got.finish_reason == "length":
        return Result("sampling", case, PASS, f"{counted}, finish_reason='length'")
    if exact and n == limit:
        return Result(
            "sampling",
            case,
            FAIL,
            f"stopped at {counted} but finish_reason={got.finish_reason!r}, expected 'length'",
            "The output was truncated by max_tokens but not reported as such, so clients cannot detect truncation.",
        )
    return Result(
        "sampling", case, INCONCLUSIVE, f"got {counted}, finish_reason={got.finish_reason!r}; limit was not reached"
    )


def _temperature_zero(client: Client, repeats: int = 3) -> tuple[Result, str]:
    """Returns the result and the baseline completion content (empty if unavailable)."""
    case = "temperature=0 repeatable"
    try:
        runs = [client.chat(GEN_MESSAGES, max_tokens=32, temperature=0) for _ in range(repeats)]
    except APIError as err:
        return _from_error("sampling", case, err), ""
    texts = [run.text for run in runs]
    baseline = runs[0].content  # stop sequences apply to the visible content, not to reasoning
    if not texts[0]:
        return Result("sampling", case, INCONCLUSIVE, "the server returned empty completions"), ""
    distinct = len(set(texts))
    if distinct == 1:
        return Result("sampling", case, PASS, f"{repeats} identical completions"), baseline
    other = next(t for t in texts if t != texts[0])
    at = next((i for i, (a, b) in enumerate(zip(texts[0], other)) if a != b), min(len(texts[0]), len(other)))
    return (
        Result(
            "sampling",
            case,
            FAIL,
            f"{distinct} different completions from {repeats} identical requests (first divergence at character {at})",
            "temperature=0 is not producing greedy decoding: temperature is ignored or overridden by a server "
            "default. Heavily batched GPU servers can also diverge slightly through numerical noise; an early "
            "divergence points to sampling, a late one to noise.",
        ),
        baseline,
    )


def _pick_stop(baseline: str) -> tuple[str, int] | None:
    """A word from the middle of the baseline, and where it first occurs."""
    for match in re.finditer(r"\w{3,}", baseline):
        cut = baseline.find(match.group())
        if cut >= max(4, len(baseline) // 3):
            return match.group(), cut
    return None


def _stop(client: Client, baseline: str, deterministic: bool) -> Result:
    case = "stop sequence"
    picked = _pick_stop(baseline)
    if picked is None:
        return Result("sampling", case, INCONCLUSIVE, "no usable baseline completion to choose a stop sequence from")
    stop, cut = picked
    try:
        got = client.chat(GEN_MESSAGES, max_tokens=32, temperature=0, stop=[stop])
    except APIError as err:
        return _from_error("sampling", case, err)
    expected = baseline[:cut]
    if stop in got.content:
        where = got.content.find(stop) + len(stop)
        if got.content[where:].strip():
            detail = f"stop={stop!r} but the output continues past it: {_short(got.content)!r}"
            hint = "The server ignores `stop`."
        else:
            detail = f"generation halted at stop={stop!r} but the stop sequence is included in the output"
            hint = "The OpenAI API excludes the stop sequence from the returned text; this server leaves it in."
        return Result("sampling", case, FAIL, detail, hint)
    if got.content and expected.startswith(got.content.rstrip()):
        note = "" if got.finish_reason == "stop" else f" (note: finish_reason={got.finish_reason!r}, expected 'stop')"
        return Result("sampling", case, PASS, f"output cut before stop={stop!r}{note}")
    if deterministic and got.content.startswith(expected) and got.content[len(expected) :].strip():
        return Result(
            "sampling",
            case,
            FAIL,
            f"stop={stop!r} was removed from the text but generation carried on past it: {_short(got.content)!r}",
            "The server strips the stop sequence from the output without halting generation. Typically stop "
            "sequences are matched on token boundaries, so a stop string that starts mid-token is never matched.",
        )
    why = "" if deterministic else "; the server is not repeatable at temperature=0, so this cannot be judged"
    return Result(
        "sampling",
        case,
        INCONCLUSIVE,
        f"output does not contain stop={stop!r} but is not a prefix of the unstopped completion either{why}",
    )


def _seed(client: Client) -> Result:
    case = "seed reproducible"
    params = {"max_tokens": 24, "temperature": 1.0}
    try:
        first = client.chat(SEED_MESSAGES, seed=1234, **params).text
        again = client.chat(SEED_MESSAGES, seed=1234, **params).text
        other = client.chat(SEED_MESSAGES, seed=98765, **params).text
    except APIError as err:
        if err.rejected:
            return Result("sampling", case, SKIP, f"server does not accept `seed` ({err.message})")
        return _from_error("sampling", case, err)
    if first != again:
        return Result(
            "sampling",
            case,
            INCONCLUSIVE,
            "`seed` is accepted but the same seed gave different output: this server ignores `seed` or does not "
            "implement it (not counted as a failure because support cannot be told apart from outside)",
        )
    if first == other:
        return Result(
            "sampling",
            case,
            INCONCLUSIVE,
            "output is identical even with a different seed at temperature=1; cannot tell whether seed works "
            "(temperature may be ignored)",
        )
    return Result("sampling", case, PASS, "same seed reproduces, a different seed changes the output")


# --------------------------------------------------------------------------- 3. streaming == non-streaming


def check_streaming(client: Client, tokenizer) -> list[Result]:
    """Does stream=true return the same completion as stream=false?"""
    case = "stream vs non-stream"
    params = {"max_tokens": 32, "temperature": 0}
    try:
        plain = client.chat(GEN_MESSAGES, **params)
        note = ""
        try:
            streamed = client.chat_stream(GEN_MESSAGES, stream_options={"include_usage": True}, **params)
        except APIError as err:
            if not err.rejected:
                raise
            streamed = client.chat_stream(GEN_MESSAGES, **params)
            note = "; stream_options not accepted"
        if streamed.text != plain.text and client.chat(GEN_MESSAGES, **params).text != plain.text:
            return [
                Result(
                    "streaming",
                    case,
                    INCONCLUSIVE,
                    "the server is not repeatable at temperature=0, so the two cannot be compared",
                )
            ]
    except APIError as err:
        return [_from_error("streaming", case, err)]

    problems = []
    if streamed.text != plain.text:
        at = next((i for i, (x, y) in enumerate(zip(plain.text, streamed.text)) if x != y), None)
        where = f"from character {at}" if at is not None else "in length"
        problems.append(
            f"content differs {where}: non-stream {len(plain.text)} chars ending {plain.text[-25:]!r}, "
            f"stream {len(streamed.text)} chars ending {streamed.text[-25:]!r}"
        )
    if streamed.finish_reason != plain.finish_reason:
        problems.append(f"finish_reason differs: non-stream {plain.finish_reason!r}, stream {streamed.finish_reason!r}")
    if None not in (plain.prompt_tokens, streamed.prompt_tokens) and plain.prompt_tokens != streamed.prompt_tokens:
        problems.append(f"prompt_tokens differ: non-stream {plain.prompt_tokens}, stream {streamed.prompt_tokens}")
    if problems:
        return [
            Result(
                "streaming",
                case,
                FAIL,
                "; ".join(problems),
                "The streaming and non-streaming code paths of the server disagree (separate output parsers, "
                "stop handling or prompt construction).",
            )
        ]
    both = None not in (plain.prompt_tokens, streamed.prompt_tokens)
    usage = "prompt_tokens match" if both else "usage not reported on both sides"
    return [Result("streaming", case, PASS, f"same content and finish_reason={plain.finish_reason!r}; {usage}{note}")]


# --------------------------------------------------------------------------- 4. tool calling round-trip

# Markup that models emit for tool calls and that a server-side parser should have consumed.
_LEAK_MARKERS = (
    "<tool_call>",
    "[TOOL_CALLS]",
    "<|python_tag|>",
    "<function=",
    "<function_call>",
    "<|tool_call",
    "<|tool_calls",
    "<｜tool▁call",
)
_TOOL_NAME = WEATHER_TOOL["function"]["name"]


def _leaked_call(content: str) -> bool:
    if any(marker in content for marker in _LEAK_MARKERS):
        return True
    as_json = re.search(r'"name"\s*:\s*"' + _TOOL_NAME + '"', content)
    as_python = re.search(r"\b" + _TOOL_NAME + r"\s*\(\s*\w+\s*=", content)
    return bool(as_json or as_python)


def _judge_tool_call(got: Completion) -> tuple[str, str, str]:
    """Classify one completion of the tool request. Returns (status, detail, hint)."""
    if not got.tool_calls:
        if _leaked_call(got.content):
            return (
                FAIL,
                f"no structured tool_calls; tool-call markup leaked into content: {_short(got.content, 80)!r}",
                "The model produced a tool call but the server returned it as plain text: its tool-call parser "
                "is missing, not enabled, or does not match this model's format (or the markup was malformed).",
            )
        if got.finish_reason == "tool_calls":
            return (
                FAIL,
                "finish_reason is 'tool_calls' but the response carries no tool_calls"
                + (f" (content: {_short(got.content, 60)!r})" if got.content else " and no content"),
                "The server recognised a tool call in the model output and then dropped it because its parser "
                "could not read it. A chat-template mismatch is a common cause (the model is prompted with a "
                "different call syntax from the one the parser expects): compare the parity 'with tools' row.",
            )
        cut = " (output was cut off by max_tokens)" if got.finish_reason == "length" else ""
        return (
            INCONCLUSIVE,
            f"the model answered in text without calling the tool{cut}: {_short(got.text, 60)!r}; "
            "this depends on the model as much as on the server",
            "",
        )
    call = got.tool_calls[0]
    args = call["arguments"]
    if not isinstance(args, str):
        return (
            FAIL,
            f"tool_calls[0].function.arguments is a {type(args).__name__}, not a JSON-encoded string",
            "OpenAI clients call json.loads() on `arguments`; returning an object breaks them.",
        )
    try:
        parsed = json.loads(args)
    except ValueError:
        return (
            FAIL,
            f"tool_calls[0].function.arguments is not valid JSON: {_short(args, 60)!r}",
            "The server put unparseable text into the structured arguments field (broken tool-call parser or "
            "truncated output).",
        )
    problem = None
    if call["name"] != _TOOL_NAME:
        problem = f"called unknown tool {call['name']!r}"
    elif not isinstance(parsed, dict) or not isinstance(parsed.get("city"), str):
        problem = f"arguments {args} lack the required string `city`"
    elif "unit" in parsed and parsed["unit"] not in ("celsius", "fahrenheit"):
        problem = f"arguments {args} have a `unit` outside the enum"
    if problem:
        return (
            INCONCLUSIVE,
            f"structured tool call arrived, but {problem}; schema adherence is up to the model unless the server "
            "enforces it",
            "",
        )
    note = "" if got.finish_reason == "tool_calls" else f" (note: finish_reason={got.finish_reason!r})"
    return PASS, f"{call['name']}({args}) in structured tool_calls{note}", ""


def check_tools(client: Client, tokenizer) -> list[Result]:
    """Does a tool call arrive as structured tool_calls, streamed or not?"""
    if not supports_tools(tokenizer, TOOL_MESSAGES, [WEATHER_TOOL]):
        detail = "the model's chat template does not render tools"
        return [Result("tools", "tool call", SKIP, detail), Result("tools", "tool call (stream)", SKIP, detail)]

    params = {"tools": [WEATHER_TOOL], "max_tokens": 200, "temperature": 0}
    results = []
    plain = None
    try:
        plain = client.chat(TOOL_MESSAGES, **params)
        results.append(Result("tools", "tool call", *_judge_tool_call(plain)))
    except APIError as err:
        results.append(_from_error("tools", "tool call", err))

    case = "tool call (stream)"
    try:
        streamed = client.chat_stream(TOOL_MESSAGES, **params)
    except APIError as err:
        results.append(_from_error("tools", case, err))
        return results
    status, detail, hint = _judge_tool_call(streamed)
    plain_ok = plain is not None and results[0].status == PASS
    if plain_ok and status == INCONCLUSIVE and not streamed.tool_calls:
        status = FAIL
        detail = f"the non-streaming request produced a tool call but the same request streamed did not: {detail}"
        hint = "The server's streaming tool-call parser loses calls that its non-streaming parser finds."
    elif plain_ok and status == PASS:
        same = streamed.tool_calls[0]["name"] == plain.tool_calls[0]["name"] and json.loads(
            streamed.tool_calls[0]["arguments"]
        ) == json.loads(plain.tool_calls[0]["arguments"])
        if same:
            detail = "streamed deltas assemble to the same call as the non-streaming response"
        else:
            status = INCONCLUSIVE
            detail = (
                f"valid call, but different from the non-streaming one: {detail}; "
                "could be sampling noise rather than a parser problem"
            )
    results.append(Result("tools", case, status, detail, hint))
    return results


CHECKS = {
    "parity": check_parity,
    "sampling": check_sampling,
    "streaming": check_streaming,
    "tools": check_tools,
}
