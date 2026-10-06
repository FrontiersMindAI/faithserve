# faithserve

**Is this endpoint serving this open-weight model faithfully?** Point `faithserve` at a Hugging Face model id
and any OpenAI-compatible chat-completions endpoint; it runs a handful of black-box checks and prints a
pass/fail report.

```bash
pip install git+https://github.com/FrontiersMindAI/faithserve
faithserve check Qwen/Qwen2.5-0.5B-Instruct --base-url http://localhost:8000/v1
```

It downloads only the model's tokenizer files (never the weights), sends about twenty short requests, and
exits non-zero if a check fails, so it can gate a deployment in CI.

## Why

The same weights behave differently depending on who serves them. Serving stacks apply the wrong chat
template, add a second BOS token, lose tool calls, or silently ignore sampling parameters, and none of that
raises an error: you just get a slightly worse model. Some upstream reports of this class of problem:
[vllm-project/vllm#39056](https://github.com/vllm-project/vllm/issues/39056),
[ollama/ollama#14493](https://github.com/ollama/ollama/issues/14493),
[EleutherAI/lm-evaluation-harness#1841](https://github.com/EleutherAI/lm-evaluation-harness/issues/1841).
(Linked as motivation; faithserve has not been used to reproduce those specific issues.)

`faithserve` compares what the endpoint does with what the model's own tokenizer and chat template say it
should do.

## Example

A real run (6 October 2026, macOS on Apple silicon) against `mlx_lm.server` from mlx-lm 0.32.0, serving a
local download of `mlx-community/Qwen2.5-0.5B-Instruct-4bit` with default settings:

```console
$ faithserve check Qwen/Qwen2.5-0.5B-Instruct --base-url http://127.0.0.1:8765/v1 --served-model Qwen2.5-0.5B-Instruct-4bit
faithserve 0.1.0
reference: Qwen/Qwen2.5-0.5B-Instruct
endpoint:  http://127.0.0.1:8765/v1 (model 'Qwen2.5-0.5B-Instruct-4bit')

parity     single user turn          PASS          36 prompt tokens on both sides
parity     system + user             PASS          31 prompt tokens on both sides
parity     multi-turn                PASS          71 prompt tokens on both sides
parity     unicode                   PASS          58 prompt tokens on both sides
parity     whitespace                PASS          43 prompt tokens on both sides
parity     with tools                FAIL          reference 208 tokens, server 210 (+2)
sampling   max_tokens=5              PASS          5 completion tokens, finish_reason='length'
sampling   temperature=0 repeatable  PASS          3 identical completions
sampling   stop sequence             FAIL          stop='Lion' was removed from the text but generation carried on
                                                   past it: '1. Elephant\n2. Giraffe\n3. \n4. Bear\n5. Penguin\n6.
                                                   Seal\n7. ...'
sampling   seed reproducible         PASS          same seed reproduces, a different seed changes the output
streaming  stream vs non-stream      PASS          same content and finish_reason='length'; prompt_tokens match
tools      tool call                 FAIL          finish_reason is 'tool_calls' but the response carries no
                                                   tool_calls and no content
tools      tool call (stream)        FAIL          finish_reason is 'tool_calls' but the response carries no
                                                   tool_calls and no content

What the failures usually mean:
  parity: Only the shape with tools differs: tool definitions are rendered differently from the reference template
    (different JSON formatting, a different tool prompt, or tools moved elsewhere).
  sampling: The server strips the stop sequence from the output without halting generation. Typically stop sequences
    are matched on token boundaries, so a stop string that starts mid-token is never matched.
  tools: The server recognised a tool call in the model output and then dropped it because its parser could not read
    it. A chat-template mismatch is a common cause (the model is prompted with a different call syntax from the one
    the parser expects): compare the parity 'with tools' row.

FAIL: 9 passed, 4 failed, 0 inconclusive, 0 skipped
```

Each failure was traced by hand before being trusted:

- **Parity with tools (+2 tokens), and both tool rows.** The chat template inside that MLX conversion differs
  from the one in `Qwen/Qwen2.5-0.5B-Instruct` today by one line: it tells the model to answer with
  `{{"name": ..., "arguments": ...}}` (doubled braces) where the current upstream template has single braces.
  Prompted that way, the model emits `{{"name": "get_weather", ...}}`, which is not JSON, and the server
  drops the call. Restarting the same server with the upstream template
  (`mlx_lm.server --chat-template "$(cat upstream_template.jinja)"`) turned all three rows to PASS
  (208 prompt tokens on both sides, `get_weather({"city": "Paris", "unit": "fahrenheit"})` in `tool_calls`,
  streamed and not), for 12 passed and 1 failed overall.
- **Stop sequence.** Reproduced directly against the server: `stop=["Lion"]` deleted the word from the output
  and generation ran on to `max_tokens`, while `stop=[" Lion"]` (with the leading space, matching the token
  the model generates) stopped correctly with `finish_reason="stop"`. This row fails with either template.

That is one model, one server version and one machine. It says nothing about other models or versions.

## What it checks

| Check | What it does | What a failure usually means |
| --- | --- | --- |
| `parity` | Renders six conversation shapes (single user turn, system + user, multi-turn, unicode, whitespace, with a `tools` list) with the model's own `apply_chat_template(..., add_generation_prompt=True)` and compares the token count with the endpoint's `usage.prompt_tokens`. | The server builds a different prompt: wrong or outdated chat template, double or missing BOS (reported by name when the delta is exactly ±1 on every shape), a system prompt injected or dropped, tools rendered differently or not at all. |
| `sampling` | `max_tokens=5` is respected and reported as `finish_reason="length"`; a stop sequence taken from the model's own output cuts the output there; `temperature=0` gives the same completion three times; the same `seed` reproduces and a different one does not. | A parameter is ignored, renamed or overridden by a server-side default. |
| `streaming` | The same deterministic request with `stream=true` and `stream=false` gives the same content, `finish_reason` and `prompt_tokens`. | The streaming and non-streaming code paths of the server disagree. |
| `tools` | Offers one simple tool with a prompt that plainly calls for it. The call must arrive in `tool_calls`, with `arguments` a JSON string, and the streamed deltas must assemble to the same call. | Tool-call markup leaked into `content`, or a call was detected and then lost: the server's tool-call parser is missing, not enabled, or does not match what the model emits. |

Every row is one of:

- **PASS** the endpoint did what the reference says.
- **FAIL** the endpoint did something a faithful server would not do. Only this sets exit code 1.
- **INCONCLUSIVE** it cannot be told from outside. For example the model answered in prose instead of calling
  the tool (that may be the model, not the server), the server reports no `usage`, or `seed` is accepted but
  has no effect (ignored and unsupported look the same from outside).
- **SKIP** not applicable: the model's template has no tool support, or the server rejects the parameter.

Exit codes: `0` no failures, `1` at least one FAIL, `2` the checks could not run (unreachable endpoint, unknown
model, gated tokenizer).

## Usage

```
faithserve check <hf-model-id> --base-url <url> [--served-model NAME] [--api-key KEY]
                 [--json report.json] [--only parity,tools] [--skip sampling] [--timeout 120]
```

- `<hf-model-id>` is the **original** model repository, the one whose tokenizer and chat template define
  correct behaviour (not the GGUF or MLX conversion being served). A local directory with tokenizer files
  works too. Gated models need `HF_TOKEN`.
- `--served-model` is the name the server knows the model by, when it is not the Hugging Face id.
- The API key is read from `--api-key` or, better, the `FAITHSERVE_API_KEY` environment variable. It is sent
  only as the `Authorization` header and never printed or written to the report.

| Server | Typical command | Tested |
| --- | --- | --- |
| mlx-lm | `mlx_lm.server --model <path-or-id> --port 8080`, then `--base-url http://localhost:8080/v1` | yes, see above |
| vLLM | `vllm serve <model>`, then `--base-url http://localhost:8000/v1` (tool calling needs vLLM's `--enable-auto-tool-choice --tool-call-parser ...`) | not yet |
| llama.cpp | `llama-server -m model.gguf --jinja`, then `--base-url http://localhost:8080/v1` | not yet |
| Ollama | `--base-url http://localhost:11434/v1 --served-model <ollama tag>` | not yet |
| LM Studio | `--base-url http://localhost:1234/v1 --served-model <name shown in LM Studio>` | not yet |
| Hosted providers | `--base-url https://<provider>/v1 --served-model <provider's model name>` with `FAITHSERVE_API_KEY` set | not yet |

"Not yet" means the base URL is the server's documented default but faithserve has not been run against it.
Reports from those servers, passing or failing, are very welcome.

### In CI

```yaml
- name: Check the endpoint serves the model faithfully
  env:
    FAITHSERVE_API_KEY: ${{ secrets.ENDPOINT_API_KEY }}
  run: |
    pip install git+https://github.com/FrontiersMindAI/faithserve
    faithserve check Qwen/Qwen2.5-0.5B-Instruct --base-url "$ENDPOINT_URL" --json faithserve.json
```

The step fails if any check fails. Use `--only` or `--skip` to leave out checks that do not apply to your
setup, and keep `faithserve.json` as a build artifact.

## Limitations

- **Black-box.** It sees requests and responses only. Matching token counts mean the prompt has the right
  length, not that it is byte-identical; it does not look at logits, so it cannot see quantization or weight
  differences.
- **Parity needs `usage.prompt_tokens`.** Servers that omit it get INCONCLUSIVE. Servers that report it
  inaccurately will fail a check they might deserve to pass.
- **The reference is the tokenizer on the Hub today.** If a model's authors intend a different template for
  serving than the one in the repository, faithserve will flag the difference.
- **BOS convention.** When a template emits no BOS and the tokenizer would add one itself, both counts are
  accepted, because both conventions are in use.
- **The tool check depends on the model.** A small model that does not call the tool is INCONCLUSIVE, and
  schema violations inside an otherwise well-formed call are INCONCLUSIVE too. Only server-side evidence
  (leaked markup, a dropped call, unparseable `arguments`) is a FAIL. A FAIL for leaked or dropped markup can
  still originate in malformed model output; the report quotes it so you can judge.
- **`temperature=0` repeatability** can break on heavily batched GPU servers through numerical noise alone.
  The report gives the position of the first divergence to help tell that apart from real sampling.
- **Reasoning models.** Thinking output is not checked, and with small `max_tokens` budgets a model that
  thinks first may give INCONCLUSIVE rows.
- Only `/chat/completions` is exercised, one request at a time.
- Validated so far against one real server (above) and the fake server in `tests/`.

## Roadmap

Not in 0.1, in rough order of interest:

- Logprob comparison against reference weights, to detect quantization drift or swapped weights.
- Reasoning / thinking-block parsing checks.
- HTML reports.
- A public leaderboard of hosted providers.
- Config files, a plugin system for custom checks, concurrent requests, a web UI.

## Contributing

```bash
git clone https://github.com/FrontiersMindAI/faithserve && cd faithserve
pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest
```

The tests run offline in about a second: they train a tiny tokenizer in memory and talk to a fake
OpenAI-compatible server (`tests/fake_server.py`) with one switch per failure mode. A new check or a new
diagnosis should come with a fault in the fake server that triggers it.

The most useful contribution is a run against a server listed as "not yet" above. If faithserve reports a
FAIL that turns out to be wrong, please open an issue with the JSON report: false positives are treated as
bugs.

The code is deliberately small: `client.py` (HTTP), `reference.py` (tokenizer and template), `checks.py`
(the four checks), `report.py` and `cli.py`.

## Licence

Apache-2.0, see [LICENSE](LICENSE).
