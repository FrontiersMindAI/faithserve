"""A tiny in-process OpenAI-compatible server with switchable faults.

The "model" is deterministic: it recites a fixed word list (one word = one completion
token), or calls the weather tool when tools are supplied. Prompt tokens are counted
with the same tokenizer and chat template the tests use as reference, so with no
faults enabled the server is faithful by construction.
"""

from __future__ import annotations

import json
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliett kilo lima mike november oscar papa "
    "quebec romeo sierra tango uniform victor whiskey xray yankee zulu red orange yellow green blue indigo "
    "violet north east south west spring summer autumn winter"
).split()
TOOL_ARGS = '{"city": "Paris", "unit": "celsius"}'
TOOL_MARKUP = '<tool_call>\n{"name": "get_weather", "arguments": ' + TOOL_ARGS + "}\n</tool_call>"

FAULTS = {
    "extra_bos",  # tokenizes the rendered prompt with special tokens: a second BOS
    "inject_system",  # adds a default system prompt when the request has none
    "drop_tools",  # does not render tools into the prompt
    "ignore_max_tokens",
    "wrong_finish_reason",  # truncates at max_tokens but reports "stop"
    "ignore_stop",
    "strip_stop_only",  # removes the stop text but keeps generating
    "nondeterministic",  # samples even at temperature 0
    "ignore_seed",
    "reject_seed",  # HTTP 400 on the seed parameter
    "stream_mismatch",  # streaming loses the last word
    "leak_tool_call",  # tool call returned as text in content
    "swallow_tool_call",  # finish_reason tool_calls, but no tool_calls at all
    "stream_drops_tool_call",
    "arguments_object",  # arguments as a JSON object instead of a string
    "no_usage",
    "error_500",
}


class FakeServer:
    def __init__(self, tokenizer, faults=(), api_key=None):
        unknown = set(faults) - FAULTS
        assert not unknown, unknown
        self.tokenizer = tokenizer
        self.faults = set(faults)
        self.api_key = api_key
        self.seen_authorization = []
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}/v1"

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._httpd.shutdown()
        self._httpd.server_close()

    # ------------------------------------------------------------------ the "model"

    def prompt_tokens(self, messages, tools):
        if "inject_system" in self.faults and messages[0]["role"] != "system":
            messages = [{"role": "system", "content": "You are a helpful assistant."}, *messages]
        if "drop_tools" in self.faults:
            tools = None
        text = self.tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
        return len(self.tokenizer(text, add_special_tokens="extra_bos" in self.faults)["input_ids"])

    def generate(self, body, stream):
        """Returns (words, finish_reason)."""
        words = list(WORDS)
        seed = None if "ignore_seed" in self.faults else body.get("seed")
        if "nondeterministic" in self.faults:
            random.Random().shuffle(words)
        elif body.get("temperature", 1.0) > 0:
            random.Random(seed).shuffle(words)

        finish = "stop"
        limit = body.get("max_tokens")
        if limit is not None and limit < len(words) and "ignore_max_tokens" not in self.faults:
            words = words[:limit]
            finish = "stop" if "wrong_finish_reason" in self.faults else "length"

        text = " ".join(words)
        stops = body.get("stop") or []
        stops = [stops] if isinstance(stops, str) else stops
        for stop in stops:
            if stop in text and "ignore_stop" not in self.faults:
                if "strip_stop_only" in self.faults:
                    text = text.replace(stop, "", 1)
                else:
                    text, finish = text[: text.index(stop)], "stop"
        words = text.split(" ")
        if stream and "stream_mismatch" in self.faults:
            words = words[:-1]
        return words, finish

    # ------------------------------------------------------------------ HTTP

    def _handler(server):  # noqa: N805 - `server` is the FakeServer, `self` the request handler
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _json(self, status, payload):
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path.endswith("/models"):
                    self._json(200, {"object": "list", "data": [{"id": "fake-model", "object": "model"}]})
                else:
                    self._json(404, {"error": {"message": "not found"}})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                server.seen_authorization.append(self.headers.get("Authorization"))
                if not self.path.endswith("/v1/chat/completions"):
                    return self._json(404, {"detail": "Not Found"})
                if server.api_key and self.headers.get("Authorization") != f"Bearer {server.api_key}":
                    return self._json(401, {"error": {"message": "invalid api key"}})
                if body.get("model") != "fake-model":
                    return self._json(404, {"error": {"message": f"model {body.get('model')!r} not found"}})
                if "error_500" in server.faults and body.get("max_tokens") != 1:
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(b"Internal Server Error")
                    return None
                if "reject_seed" in server.faults and "seed" in body:
                    return self._json(400, {"error": {"message": "unknown parameter: seed"}})

                tools = body.get("tools")
                stream = bool(body.get("stream"))
                usage = None
                if "no_usage" not in server.faults:
                    usage = {"prompt_tokens": server.prompt_tokens(body["messages"], tools)}

                message = {"role": "assistant", "content": None}
                if tools:
                    finish, completion_tokens = "tool_calls", 12
                    arguments = json.loads(TOOL_ARGS) if "arguments_object" in server.faults else TOOL_ARGS
                    call = {"id": "call_1", "type": "function", "function": {"name": "get_weather"}}
                    call["function"]["arguments"] = arguments
                    if "leak_tool_call" in server.faults:
                        message["content"], finish = TOOL_MARKUP, "stop"
                    elif "swallow_tool_call" in server.faults:
                        pass
                    elif stream and "stream_drops_tool_call" in server.faults:
                        message["content"], finish = "", "stop"
                    else:
                        message["tool_calls"] = [call]
                else:
                    words, finish = server.generate(body, stream)
                    message["content"] = " ".join(words)
                    completion_tokens = len(words)
                if usage:
                    usage["completion_tokens"] = completion_tokens
                    usage["total_tokens"] = usage["prompt_tokens"] + completion_tokens

                if not stream:
                    payload = {"object": "chat.completion", "model": "fake-model"}
                    payload["choices"] = [{"index": 0, "message": message, "finish_reason": finish}]
                    if usage:
                        payload["usage"] = usage
                    return self._json(200, payload)
                self._stream(
                    message, finish, usage if (body.get("stream_options") or {}).get("include_usage") else None
                )
                return None

            def _stream(self, message, finish, usage):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()

                def send(delta, finish_reason=None):
                    chunk = {"object": "chat.completion.chunk", "model": "fake-model"}
                    chunk["choices"] = [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())

                send({"role": "assistant", "content": ""})
                if message.get("tool_calls"):
                    call = message["tool_calls"][0]
                    args = call["function"]["arguments"]
                    head = {"index": 0, "id": call["id"], "type": "function"}
                    if isinstance(args, str):
                        send({"tool_calls": [{**head, "function": {"name": "get_weather", "arguments": ""}}]})
                        for piece in (args[:9], args[9:]):
                            send({"tool_calls": [{"index": 0, "function": {"arguments": piece}}]})
                    else:
                        send({"tool_calls": [{**head, "function": {"name": "get_weather", "arguments": args}}]})
                elif message["content"]:
                    pieces = message["content"].split(" ")
                    for i, piece in enumerate(pieces):
                        send({"content": piece if i == 0 else " " + piece})
                send({}, finish)
                if usage:
                    tail = {"object": "chat.completion.chunk", "choices": [], "usage": usage}
                    self.wfile.write(f"data: {json.dumps(tail)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")

        return Handler
