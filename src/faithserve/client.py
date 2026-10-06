"""Minimal client for OpenAI-compatible ``/chat/completions`` endpoints.

Only what the checks need: one blocking call, one streaming call, and errors
that carry enough information to tell "unsupported" from "broken".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import httpx


class APIError(Exception):
    """A request that did not produce a usable completion.

    ``status`` is the HTTP status code, or None when no HTTP response was
    received at all (connection refused, timeout) or the body was unusable.
    """

    def __init__(self, message: str, status: int | None = None, transient: bool = False):
        super().__init__(message)
        self.message = message
        self.status = status
        self.transient = transient  # timeout / connection problem: says nothing about the server

    @property
    def rejected(self) -> bool:
        """The server understood the request and refused it (4xx)."""
        return self.status is not None and 400 <= self.status < 500


@dataclass
class Completion:
    """The parts of a chat completion the checks look at, stream or not."""

    content: str = ""
    reasoning: str = ""
    finish_reason: str | None = None
    tool_calls: list[dict] = field(default_factory=list)  # [{"name": str, "arguments": str | object}]
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @property
    def text(self) -> str:
        """Everything the model generated, for equality comparisons."""
        return self.reasoning + self.content


def _error_message(response: httpx.Response) -> str:
    """Pull a readable message out of an error body, whatever its shape."""
    try:
        body = response.json()
    except ValueError:
        text = response.text.strip()
        return text[:300] if text else response.reason_phrase
    if isinstance(body, dict):
        err = body.get("error", body.get("detail", body.get("message", body)))
        if isinstance(err, dict):
            err = err.get("message") or err.get("detail") or json.dumps(err)
        if isinstance(err, list):
            err = json.dumps(err)
        return str(err)[:300]
    return str(body)[:300]


def _read_usage(usage: object, out: Completion) -> None:
    if isinstance(usage, dict):
        if isinstance(usage.get("prompt_tokens"), int):
            out.prompt_tokens = usage["prompt_tokens"]
        if isinstance(usage.get("completion_tokens"), int):
            out.completion_tokens = usage["completion_tokens"]


def _reasoning_of(message: dict) -> str:
    return message.get("reasoning_content") or message.get("reasoning") or ""


class Client:
    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(headers=headers, timeout=timeout)

    def close(self) -> None:
        self._http.close()

    def list_models(self) -> list[str]:
        """Model names the server advertises; empty if it does not say."""
        try:
            data = self._http.get(f"{self.base_url}/models").json()
            return [m["id"] for m in data["data"]]
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return []

    def chat(self, messages: list[dict], **params) -> Completion:
        payload = {"model": self.model, "messages": messages, **params}
        try:
            response = self._http.post(f"{self.base_url}/chat/completions", json=payload)
        except httpx.TimeoutException:
            raise APIError("request timed out", transient=True) from None
        except httpx.HTTPError as exc:
            raise APIError(f"could not reach endpoint ({type(exc).__name__}: {exc})", transient=True) from None
        if response.status_code != 200:
            raise APIError(f"HTTP {response.status_code}: {_error_message(response)}", response.status_code)
        try:
            body = response.json()
            choice = body["choices"][0]
            message = choice["message"]
            if not isinstance(message, dict):
                raise TypeError
        except (ValueError, KeyError, IndexError, TypeError):
            raise APIError(f"response is not an OpenAI chat completion: {response.text[:200]!r}") from None

        out = Completion(
            content=message.get("content") or "",
            reasoning=_reasoning_of(message),
            finish_reason=choice.get("finish_reason"),
        )
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            out.tool_calls.append({"name": fn.get("name"), "arguments": fn.get("arguments")})
        _read_usage(body.get("usage"), out)
        return out

    def chat_stream(self, messages: list[dict], **params) -> Completion:
        """Same request with ``stream=true``, reassembled into one Completion."""
        payload = {"model": self.model, "messages": messages, "stream": True, **params}
        out = Completion()
        calls: dict[int, dict] = {}
        try:
            with self._http.stream("POST", f"{self.base_url}/chat/completions", json=payload) as response:
                if response.status_code != 200:
                    response.read()
                    raise APIError(f"HTTP {response.status_code}: {_error_message(response)}", response.status_code)
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except ValueError:
                        raise APIError(f"stream chunk is not JSON: {data[:200]!r}") from None
                    if isinstance(chunk, dict) and chunk.get("error"):
                        raise APIError(f"error in stream: {str(chunk['error'])[:300]}")
                    _read_usage(chunk.get("usage"), out)
                    for choice in chunk.get("choices") or []:
                        if choice.get("finish_reason"):
                            out.finish_reason = choice["finish_reason"]
                        delta = choice.get("delta") or {}
                        out.content += delta.get("content") or ""
                        out.reasoning += _reasoning_of(delta)
                        for position, call in enumerate(delta.get("tool_calls") or []):
                            slot = calls.setdefault(call.get("index", position), {"name": None, "arguments": ""})
                            fn = call.get("function") or {}
                            if fn.get("name"):
                                slot["name"] = fn["name"]
                            args = fn.get("arguments")
                            if isinstance(args, str):
                                slot["arguments"] += args
                            elif args is not None:
                                slot["arguments"] = args  # not a string: kept as-is so the check can report it
        except httpx.TimeoutException:
            raise APIError("request timed out", transient=True) from None
        except httpx.HTTPError as exc:
            raise APIError(f"could not reach endpoint ({type(exc).__name__}: {exc})", transient=True) from None
        out.tool_calls = [calls[i] for i in sorted(calls)]
        return out
