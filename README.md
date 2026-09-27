# stunt-double

A local stand-in for System 1 decision models, so you can build your harness before the real model is approved.

## Why this exists

System 1 models such as Jev took everyone by storm. They don't write text; they return a typed decision with a calibrated probability in milliseconds. Teams use them for model routing, guardrails, tool-call gating, inbox triage, reranking, LLM evals, bulk labeling, real-time control, and confidence gates.

Many companies can't use them yet. A new model has to pass security review and whitelisting first, and some teams need it to run on-prem so no context leaves the network. That takes months, and in the meantime development stops: you can't build a harness around an API you're not allowed to call.

stunt-double fills that gap. It serves decision endpoints with the same shape of output (a typed answer, a probability for every option, and a confidence score), and a model you're already allowed to use does the work underneath. Your team writes the routing, guardrails, gates, and thresholds now. When the real model is approved, you swap the engine and keep everything you built.

Like a stunt double on a film set, it stands in while the star isn't available. It's slower and less accurate than a purpose-built decision model, but it lets the work go on.

**Any model behind an OpenAI-compatible API works as the engine.** We develop and test with Qwen 2.5 on a local GPU through Ollama. vLLM, llama.cpp, and LM Studio work the same way, and so can a hosted small model, such as one from OpenAI or Anthropic, if your company has already approved it (see [Choosing a model](#choosing-a-model)).

## Quick start

```bash
# 1. A local model server (Ollama >= 0.12.11 returns the token probabilities we need)
ollama pull qwen2.5:7b      # ~4.7 GB; qwen2.5:3b (~1.9 GB) if you're short on memory

# 2. Install and check the model works
pip install stunt-double
stunt-double doctor

# 3. Serve
stunt-double serve            # http://127.0.0.1:8787, interactive docs at /docs
```

```bash
curl -s localhost:8787/v1/route -H 'content-type: application/json' \
  -d '{"input": "Prove that there are infinitely many primes, then formalize it in Lean."}'
```

```json
{
  "route": "large",
  "probabilities": {"small": 0.0121, "medium": 0.1603, "large": 0.8276},
  "confidence": 0.7414,
  "meta": {"engine": "openai", "model": "qwen2.5:7b", "latency_ms": 212, "usage": {"input_tokens": 187, "output_tokens": 1}, "warnings": []}
}
```

(Numbers are illustrative.)

No model server yet? `stunt-double serve --engine fake` answers from word overlap. It's instant and deterministic; use it for CI, never for quality.

## Endpoints

| Endpoint | Use it for | Returns |
|---|---|---|
| `POST /v1/decide` | Anything: ask several typed questions about one input | per question: `binary`, `choice`, or `scale` answer |
| `POST /v1/route` | Model routing: small, medium, or large model? Or your own routes | `route`, `probabilities`, `confidence` |
| `POST /v1/guard` | Screen input for prompt injection, abuse, off-topic use | `allowed`, `flagged`, per-policy `probability` |
| `POST /v1/gate` | Should an agent's tool call be allowed, confirmed, or denied? | `decision`, `probabilities`, `confidence` |
| `POST /v1/classify` | Inbox triage, intent detection, bulk labeling (`inputs: [...]`) | per item: `label`, `probabilities`, `confidence` |
| `POST /v1/judge` | LLM evals: score an output against a rubric | `score`, `normalized`, `level`, `probabilities` |
| `POST /v1/rerank` | Order documents by relevance (Cohere-style format, also `/v2/rerank`) | `results: [{index, relevance_score}]` |

Every response carries `meta`: engine, model, latency, token usage, and `warnings`, which tell you when an answer is less trustworthy than usual.

### `/v1/decide`: the core

Three question types cover every use case above:

```json
{
  "input": {"ticket": "My flight was cancelled and nobody is answering the phone!!"},
  "questions": {
    "wants_refund": {"type": "binary", "question": "Does the customer ask for their money back?"},
    "intent": {
      "type": "choice",
      "question": "What does the customer want?",
      "options": {
        "refund": "Money returned.",
        "rebooking": "A replacement flight.",
        "information": "Only information."
      }
    },
    "frustration": {
      "type": "scale",
      "question": "How frustrated is the customer?",
      "levels": ["Calm.", "Concerned but civil.", "Very angry."]
    }
  }
}
```

```json
{
  "answers": {
    "wants_refund": {"type": "binary", "value": false, "probability": 0.31, "confidence": 0.38},
    "intent": {"type": "choice", "value": "rebooking", "probabilities": {"refund": 0.22, "rebooking": 0.71, "information": 0.07}, "confidence": 0.565},
    "frustration": {"type": "scale", "value": 1.62, "level": 2, "probabilities": {"0": 0.04, "1": 0.3, "2": 0.66}, "legend": {"0": "Calm.", "1": "Concerned but civil.", "2": "Very angry."}, "confidence": 0.49}
  },
  "meta": {"...": "..."}
}
```

- **binary**: `probability` that the answer is yes. Optional `yes` and `no` fields say what each means.
- **choice**: up to 20 unordered options. `value` is the most likely key.
- **scale**: 2 to 10 ordered levels, lowest first. `value` is the probability-weighted level (it can fall between levels); `level` is the single most likely one.
- **confidence**: how concentrated the distribution is, 1 when all probability is on one answer, 0 when spread evenly.

Questions in one request run in parallel.

### Confidence gates

Decide the thresholds in code, not in a prompt:

```python
from stunt_double.client import StuntDouble

sd = StuntDouble()
g = sd.gate("send_email", {"to": "all-staff@corp.com"}, context="User asked to draft a reply to Bob.")
if g["decision"] == "allow" and g["confidence"] > 0.9:
    run_tool()
elif g["decision"] == "deny" and g["confidence"] > 0.9:
    refuse()
else:
    ask_human()
```

## How it works

Each question becomes one prompt that asks the question, shows the input, asks the question again, and ends with "reply with exactly one label": the option name for choices, a digit for scale levels, yes or no for binary. The model generates at most a few tokens. stunt-double reads the probabilities the model assigned to each label's tokens (`logprobs`) and renormalizes them. So the probabilities come from the model itself, not from a number the model writes down. That is also why it's much faster than asking an LLM for JSON.

Two details that matter with small models:

- **Option names, not letters.** Lettered options (A, B, C) make small models over-pick "B". Answering with the option name avoids that. When two names share their first token ("re" in *refund* and *rebooking*), the next token's probabilities split them.
- **Question before and after the input.** On our test cases this raised accuracy from 6/9 to 8/9 for yes/no questions.

If the model server returns no logprobs, or the model answers with something that isn't a label, you still get an answer, plus a warning in `meta.warnings`.

## Fidelity: what differs from a real decision model

Build your harness knowing these:

- **Latency.** A local 7B model takes roughly 250 ms per question on a good GPU (a rerank of 4 documents about 1 s), several seconds on CPU. Purpose-built decision models aim much lower. Don't design around the slowness: no caching or batching workarounds a real engine won't need. `meta.latency_ms` shows what you're paying.
- **Calibration.** Probabilities from a small general model are only roughly calibrated. Thresholds you tune now (like `confidence > 0.9`) will need re-tuning on the real engine. Keep a labeled set of examples so you can re-tune in an afternoon.
- **Accuracy.** Small models miss nuance. See [Measured results](#measured-results) for what works and what doesn't, and run the evals on your own model before relying on a use case.
- **Repeatability.** Answers are deterministic while a model stays loaded, but can shift slightly after the model server reloads it. Borderline cases may flip. Don't write tests that assert exact probabilities from a real model; use `--engine fake` for that.
- **Limits.** Up to 20 options per choice question (option keys must differ ignoring case and punctuation), 10 scale levels, text input only.
- **Models.** Use non-thinking instruct models. Reasoning models that write out their thinking first (`<think>`) don't work, because the answer is no longer in the first tokens.

## Choosing a model

stunt-double talks to any server with an OpenAI-compatible `/v1/chat/completions` endpoint. What matters is whether that server returns **logprobs** (the probability of each token), because the probabilities in every answer come from them.

| Engine | Logprobs | Notes |
|---|---|---|
| Ollama ≥ 0.12.11 (default) | yes | Tested with `qwen2.5:7b` and `qwen2.5:3b` on a local GPU. |
| vLLM, llama.cpp server, LM Studio | yes | `--base-url http://localhost:8000/v1` (vLLM) or `:8080/v1` (llama.cpp). |
| OpenAI and other hosted APIs | on non-reasoning models | `--base-url https://api.openai.com/v1 --api-key ... --model <small non-reasoning model>`. Not tested yet. |
| Anthropic | no | Via its OpenAI-compatible endpoint (`--base-url https://api.anthropic.com/v1/`). Not tested yet. With no logprobs, answers are 0 or 1 and `meta.warnings` says so. |

Good local choices: Qwen 2.5 (7B recommended, 3B if memory is tight), Llama 3.x, Gemma 3, Phi-4-mini, Mistral.

A hosted model sends your input to that provider. Use one only if your company has already approved it for this data; the point of stunt-double is to not need the unapproved model.

### Measured results

`evals/run.py` holds 57 labeled cases across all use cases. Run it against any model:

```bash
uv run python evals/run.py --model qwen2.5:7b --model qwen2.5:3b
```

Results on a local GPU (acc = top answer correct; p_ok = mean probability on the correct answer):

| Use case | qwen2.5:7b | qwen2.5:3b |
|---|---|---|
| guard (injection, abuse) | 20/20, p_ok 1.00 | 15/20, p_ok 0.74 |
| route | 8/9, 0.89 | 8/9, 0.85 |
| gate | 7/8, 0.85 | 5/8, 0.63 |
| classify (inbox triage) | 6/8, 0.77 | 6/8, 0.78 |
| judge | 8/8, 1.00 | 7/8, 1.00 |
| rerank | 4/4, 1.00 | 4/4, 1.00 |
| **total** | **53/57** | **45/57** |

Median latency is 230–290 ms per call for both models on this GPU (rerank: about 1 s for 4 documents).

What that means in practice:

- **qwen2.5:7b is good enough for development** in every use case. Its misses: it allowed an 8,400 transfer to a new payee when the user asked to pay a bill, routed a legal-conflict question to `medium`, and marked two non-urgent emails as urgent.
- **qwen2.5:3b** is fine for routing, judging, and reranking, but it mixes up guard policies (flags a bomb request as prompt injection), misses role-play jailbreaks, and is erratic on tool gating.
- **Be careful with high confidence.** The 7B allowed the 8,400 transfer with near-certainty. Put hard limits (amounts, destructive tools) in code, not only in a threshold.

The set is small; treat it as a smoke test, and add cases from your own domain.

## Switching to the real engine later

1. Keep stunt-double behind your own small interface, for example `Decisions.route(...)`, `Decisions.gate(...)`.
2. Record real traffic while you develop: `stunt-double serve --record calls.jsonl`. Each line holds the request, response, and latency.
3. When the real engine is approved, implement the same interface with its SDK, replay `calls.jsonl` through both, and compare decisions and latency before switching.

The rest of your harness (thresholds aside) does not change.

## Configuration

| Flag | Environment variable | Default |
|---|---|---|
| `--engine` | `STUNT_DOUBLE_ENGINE` | `openai` (any OpenAI-compatible server) or `fake` |
| `--base-url` | `STUNT_DOUBLE_BASE_URL` | `http://localhost:11434/v1` (Ollama) |
| `--model` | `STUNT_DOUBLE_MODEL` | `qwen2.5:7b` |
| `--api-key` | `STUNT_DOUBLE_API_KEY` | none |
| `--max-concurrency` | `STUNT_DOUBLE_MAX_CONCURRENCY` | `4` |
| `--record` | `STUNT_DOUBLE_RECORD` | off |
| `--host`, `--port` | | `127.0.0.1`, `8787` |

For vLLM: `--base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct`. For llama.cpp server: `--base-url http://localhost:8080/v1`.

The server has no authentication. Keep it on localhost or behind your own gateway.

## Development

```bash
uv sync
uv run pytest                                      # fast, uses the fake engine
uv run python evals/run.py --model qwen2.5:7b      # quality, needs a model server
uv run python evals/run.py --model qwen2.5:7b --only gate   # one use case
```

Run the evals on at least two models before and after any prompt change: a wording that helps one model can hurt another.

## License and trademarks

MIT. stunt-double is an independent project with its own API design. It is not affiliated with or endorsed by TypeSafe AI, and it does not implement the Jev API. Jev is a trademark of its owner, mentioned only to describe what stunt-double stands in for. Product names mentioned anywhere in this project belong to their owners.
