# stunt-double

A local stand-in for System 1 decision models, so you can build your harness before the real model is approved.

## Why this exists

System 1 models such as Jev, Kev, and Laya took everyone by storm. They don't write text; they return a typed decision with a calibrated probability in milliseconds. Teams use them for model routing, guardrails, tool-call gating, inbox triage, reranking, LLM evals, bulk labeling, real-time control, and confidence gates.

Many companies can't use them yet. A new model has to pass security review and whitelisting first, and some teams need it to run on-prem so no context leaves the network. That takes months, and in the meantime development stops: you can't build a harness around an API you're not allowed to call.

stunt-double fills that gap. It serves decision endpoints with the same shape of output (a typed answer, a probability for every option, and a confidence score), and a model you're already allowed to use does the work underneath. Your team writes the routing, guardrails, gates, and thresholds now. When the real model is approved, you swap the engine and keep everything you built.

Kev and Laya are open-weight System 1 models you can host yourself. If your company can approve and run one of them, use it directly. stunt-double is for when you can't yet: it runs on a model you're already allowed to use.

Like a stunt double on a film set, it stands in while the star isn't available. It's slower and less accurate than a purpose-built decision model, but it lets the work go on.

**Any model behind an OpenAI-compatible API works as the engine.** We develop and test with Qwen 2.5 on a local GPU through Ollama. vLLM, llama.cpp, and LM Studio work the same way, and so can a hosted small model, such as one from OpenAI or Anthropic, if your company has already approved it (see [Choosing a model](#choosing-a-model)).

## Quick start

```bash
# 1. A local model server (Ollama >= 0.12.11 returns the token probabilities we need)
ollama pull qwen2.5:7b      # ~4.7 GB; qwen2.5:3b (~1.9 GB) if you're short on memory

# 2. Install and check the model works
pip install git+https://github.com/masteris777/stunt-double
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

No model server yet? `stunt-double serve --engine mock` answers from word overlap. It's instant and deterministic; use it for CI, never for quality.

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
| `POST /v1/extract` | Pull typed fields out of text: invoices, tickets, bookings, tool-call arguments | `values`, per-field `value` and `confidence` |

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

### `/v1/extract`: typed fields

Fields are `string`, `number`, `integer`, `boolean`, or `enum` (with `options`: a list, or option -> description). Up to 30 fields:

```json
{
  "input": "Invoice #2291 from Acme Corp. Amount due: 1.200,00 EUR by 1 October 2026. Status: unpaid.",
  "fields": {
    "vendor":    {"type": "string",  "description": "Company that issued the invoice."},
    "total":     {"type": "number",  "description": "Amount due, without currency symbol."},
    "due_date":  {"type": "string",  "description": "Due date as YYYY-MM-DD."},
    "po_number": {"type": "string",  "description": "Purchase order number."},
    "currency":  {"type": "enum",    "options": ["EUR", "USD", "GBP"]},
    "paid":      {"type": "boolean", "description": "Whether the invoice is already paid."}
  }
}
```

```json
{
  "values": {"vendor": "Acme Corp", "total": 1200.0, "due_date": "2026-10-01", "po_number": null, "currency": "EUR", "paid": false},
  "fields": {
    "vendor":    {"value": "Acme Corp", "confidence": 0.99},
    "total":     {"value": 1200.0, "confidence": 0.99},
    "due_date":  {"value": "2026-10-01", "confidence": 0.98},
    "po_number": {"value": null, "confidence": 0.97},
    "currency":  {"value": "EUR", "confidence": 0.95, "probabilities": {"EUR": 0.97, "USD": 0.02, "GBP": 0.01}},
    "paid":      {"value": false, "confidence": 0.8, "probability": 0.1}
  },
  "meta": {"...": "..."}
}
```

`value` is `null` when the input doesn't contain the field. Enum fields add `probabilities`; boolean fields add `probability` (of true). Numbers are read tolerantly (`$1,200.50`, `1.200,50 EUR`). Fields run in parallel.

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

For `/v1/extract`, enum and boolean fields are scored like choice and binary questions; all string, number, and integer fields are generated together as one JSON object, and each value's confidence comes from the probabilities of the tokens that spell it. Engines that can't generate text (System 1 servers) leave those fields `null` with a warning.

If the model server returns no logprobs, or the model answers with something that isn't a label, you still get an answer, plus a warning in `meta.warnings`.

## Fidelity: what differs from a real decision model

Build your harness knowing these:

- **Latency.** A local 7B model takes roughly 250 ms per question on a good GPU (a rerank of 4 documents about 1 s), several seconds on CPU. Purpose-built decision models aim much lower. Don't design around the slowness: no caching or batching workarounds a real engine won't need. `meta.latency_ms` shows what you're paying.
- **Calibration.** Probabilities from a small general model are only roughly calibrated. Thresholds you tune now (like `confidence > 0.9`) will need re-tuning on the real engine. Keep a labeled set of examples so you can re-tune in an afternoon.
- **Accuracy.** Small models miss nuance. See [Measured results](#measured-results) for what works and what doesn't, and run the evals on your own model before relying on a use case.
- **Repeatability.** Answers are deterministic while a model stays loaded, but can shift slightly after the model server reloads it. Borderline cases may flip. Don't write tests that assert exact probabilities from a real model; use `--engine mock` for that.
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
| Kev, Laya (`--engine systemone`) | native | Open-weight System 1 models, self-hosted. Tested: Kev 0.8B, Laya. |
| Cactus Needle (`--engine needle`) | one confidence per pick | Tiny on-device tool-calling model. Tested; not a good fit (see below). On Windows, cactus-needle 3.0.5 asks for an engine build that isn't published: download the 3.0.1 engine from Hugging Face and set `NEEDLE3_LIB_PATH`. |

Good local choices: Qwen 2.5 (7B recommended, 3B if memory is tight), Llama 3.x, Gemma 3, Phi-4-mini, Mistral.

A hosted model sends your input to that provider. Use one only if your company has already approved it for this data; the point of stunt-double is to not need the unapproved model.

### Which model for which use case

Most teams need one use case, not all nine, so pick per use case. `evals/run.py` holds 140 labeled cases, 20 per use case, including deliberately hard ones (rerank distractors that share the query's words, answers that are subtly wrong, extraction fields that aren't in the text and must come back empty). A model counts as **good enough** for a use case at 17/20 (85%) or better.

| Use case | Recommended | Also good enough | Not good enough |
|---|---|---|---|
| guard (injection, abuse) | **qwen2.5:7b** (18/20) | none | qwen2.5:3b 12, Kev 13, Laya 12 |
| route | **qwen2.5:7b** (18/20) | none | qwen2.5:3b 16, Kev 14, Laya 9 |
| gate | **qwen2.5:7b** (17/20, see caution below) | none | Laya 14, qwen2.5:3b 11, Kev 10 |
| classify (inbox triage) | **Kev 0.8B** (17/20) | none; qwen2.5:7b is just under (16/20) | qwen2.5:3b 15, Laya 12 |
| judge | **qwen2.5:7b** (18/20) | none | qwen2.5:3b 15, Kev 10, Laya 9 |
| rerank | **qwen2.5:3b** (20/20) | qwen2.5:7b 20, Laya 20, Kev 19 | Needle 10 |
| extract | **qwen2.5:3b** (16/20, 92% of fields); close, see below | none | qwen2.5:7b 15 (93% of fields), Needle 4 (69%) |

Full results, on an RTX 4070 Laptop GPU (8 GB), one model loaded at a time:

| Use case | qwen2.5:7b | qwen2.5:3b | Laya | Kev 0.8B | Needle |
|---|---|---|---|---|---|
| guard | **18**/20 | 12 | 12 | 13 | 0 |
| route | **18** | 16 | 9 | 14 | 8 |
| gate | **17** | 11 | 14 | 10 | 7 |
| classify | 16 | 15 | 12 | **17** | 5 |
| judge | **18** | 15 | 9 | 10 | 2 |
| rerank | **20** | **20** | **20** | **19** | 10 |
| extract | 15 | **16** | n/a | n/a | 4 |
| **total** | **122/140** | 105 | 76/120 | 83/120 | 36 |
| median latency per call | 260–300 ms | 250–260 ms | 25–50 ms | 30–45 ms | 55–390 ms |

Rerank latency is per query of 4 documents: about 1.1 s for the Qwen models, 90–110 ms for Laya and Kev. Guard asks two questions per input, so it takes about twice as long. Extract takes 0.9–1.3 s per record on the Qwen models (one call for all text fields plus one per enum or yes/no field). Extract counts a record as right only if every field is right. Laya and Kev can't write text, so they can't fill text fields and weren't scored on extract.

What that means in practice:

- **qwen2.5:7b is the default** and is good enough for every use case except inbox triage, where it rates polite but non-urgent requests as urgent.
- **qwen2.5:3b is enough for rerank.** It's also fine for rough routing, but it mixes up guard policies and wrongly denies harmless tool calls.
- **Be careful with confident mistakes in gate.** The 7B allowed an 8,400 transfer to a new payee, an email to the whole company, and a production deploy, all with near-certainty; each should have asked a human. Put hard limits (amounts, recipients, destructive tools) in code, not only in a confidence threshold.
- **Kev and Laya are open-weight System 1 models, and far faster** (10x). Laya ranks documents perfectly and Kev is the best at triage. Elsewhere they fall short here, but three caveats apply:
  - They got the same question wording as the LLMs, which wasn't tuned for them.
  - Only Kev's 0.8B model fits in 8 GB (the 4B needs 32 GB).
  - Kev ran without its fast kernels, which aren't available on Windows.

  If your company can approve one of them, test it on your own cases with `--engine systemone`.
- **Extraction is close to good enough.** Both Qwen models get about 92% of fields right, but only 15–16 of 20 records completely right. Typical misses: inventing a date from "next summer", leaving out a meeting title that is in the text. Check low-confidence fields, or send them to a human.
- **Needle is not a decision model.** It's a tiny on-device model that *writes* tool calls. Asked to classify, it often returns no call at all, and embedding-based rerank only reached 10/20. Even on extraction, its home ground, it got 69% of fields right out of the box. Cactus pitches fine-tuning on your own schema, which we didn't test. Its fit is the step before a decision model: Needle proposes the tool call on the device, and `/v1/gate` decides whether to run it.

Reproduce any column (one model at a time; load only one model into GPU memory):

```bash
uv run python evals/run.py --model qwen2.5:7b                      # Ollama
uv run python evals/run.py --model laya=http://localhost:8000      # laya-serve
uv run python evals/run.py --model kev=http://localhost:8009       # python -m kev.serve --run jaredpalmer/kev-0.8b --port 8009
uv run --extra needle python evals/run.py --model needle           # set NEEDLE_TELEMETRY=0
```

Twenty cases per use case is still small; add cases from your own domain before relying on a result.

## Switching to the real engine later

1. Keep stunt-double behind your own small interface, for example `Decisions.route(...)`, `Decisions.gate(...)`.
2. Record real traffic while you develop: `stunt-double serve --record calls.jsonl`. Each line holds the request, response, and latency.
3. When the real engine is approved, implement the same interface with its SDK, replay `calls.jsonl` through both, and compare decisions and latency before switching.

The rest of your harness (thresholds aside) does not change.

## Configuration

| Flag | Environment variable | Default |
|---|---|---|
| `--engine` | `STUNT_DOUBLE_ENGINE` | `openai` (any OpenAI-compatible server), `mock` (word overlap, for CI), `systemone` (a self-hosted Kev or Laya server), or `needle` (Cactus Needle on-device; `pip install "stunt-double[needle]"`) |
| `--base-url` | `STUNT_DOUBLE_BASE_URL` | `http://localhost:11434/v1` (Ollama) |
| `--model` | `STUNT_DOUBLE_MODEL` | `qwen2.5:7b` |
| `--api-key` | `STUNT_DOUBLE_API_KEY` | none |
| `--max-concurrency` | `STUNT_DOUBLE_MAX_CONCURRENCY` | `4` |
| `--systemone-url` | `STUNT_DOUBLE_SYSTEMONE_URL` | `http://localhost:8000` (serves `POST /v1/systemone`) |
| | `STUNT_DOUBLE_TIMEOUT` | `120` seconds per model call |
| `--record` | `STUNT_DOUBLE_RECORD` | off |
| `--host`, `--port` | | `127.0.0.1`, `8787` |

For vLLM: `--base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct`. For llama.cpp server: `--base-url http://localhost:8080/v1`.

`--engine systemone` and `--engine needle` send each question to a model that answers typed questions natively, with no stunt-double prompt, so you can compare stunt-double with the real thing behind the same endpoints. Needle has no probability for every option: it returns one confidence for its pick, and the other options share the rest evenly. It reranks with embeddings. Telemetry is off by default (`NEEDLE_TELEMETRY=0`, `DO_NOT_TRACK=1`), and `NEEDLE3_LIB_PATH` points it at a local build.

The server has no authentication. Keep it on localhost or behind your own gateway.

## Development

```bash
uv sync
uv run pytest                                      # unit + e2e on the mock engine; integration tests
                                                   # run when Ollama (qwen2.5:7b) or a System 1
                                                   # server on :8000 answers, else skip
uv run python evals/run.py --model qwen2.5:7b      # quality, needs a model server
uv run python evals/run.py --model qwen2.5:7b --only gate   # one use case
```

Run the evals on at least two models before and after any prompt change: a wording that helps one model can hurt another.

## Architecture

The code has three layers, and imports only go inward:

```
apps ──→ core ←── providers
```

- **`stunt_double/core/decision/`**: the domain, with no third-party imports (no httpx, pydantic, time, or os). It holds the interfaces (`IEngine`, `IDecider`, `IClock`, `IIdProvider`, `IDecisionService`, and the optional capabilities `IReranker`, `IGenerator`, `IRecordReader`, `IExtractor`), the domain types (`TChoiceQuestion`, `TRouteRequest`, ...), the prompts, the label scoring, and the use cases (`DecisionServiceBasicImpl`). `DeciderBasicImpl` turns a question into a prompt and asks an `IEngine`.
- **`stunt_double/providers/<name>/decision/`**: one folder per technology. Each implements core interfaces and depends only on core, never on another provider. `openai` (`EngineOpenAIImpl`), `mock` (`EngineMockImpl`, plus a deterministic clock and id provider for tests), `std` (the real clock and UUIDs), `systemone` (`DeciderSystemOneImpl`), and `needle` (`DeciderNeedleImpl`).
- **`stunt_double/apps/`**: `server/` (FastAPI, the pydantic transport schemas, the recorder), `cli/`, `client/` (the `StuntDouble` SDK), and `composition.py`, the only place that picks implementations from the settings.

To add an engine:

- If it scores labels from a prompt, implement `IEngine` in `providers/<name>/decision/engine_<name>_impl.py`. Implement `IGenerator` too if it can generate short text, so `/v1/extract` can fill string and number fields.
- If it answers typed questions natively, implement `IDecider` in `providers/<name>/decision/decider_<name>_impl.py`, and also `IReranker` if it can rerank without yes/no questions, or `IExtractor` if it extracts whole records.

Then add a branch in `apps/composition.py` and the name to `ENGINES` in `apps/config.py`, and put its tests in `tests/providers/<name>/`. `tests/` mirrors `src/`. `tests/core/test_dependency_rule.py` fails if core imports anything external or if one provider imports another.

## Credits

Built by Marijus Masteika with [Claude](https://claude.com/claude-code) (Anthropic) as a coding partner. Claude co-authored the commits.

## License and trademarks

MIT. stunt-double is an independent project with its own API design. It is not affiliated with or endorsed by TypeSafe AI (Jev), the Kev project, Convai Innovations (Laya), or Cactus Compute (Needle). stunt-double does not serve their APIs; `--engine systemone` and `--engine needle` are clients, for comparison. Those names belong to their owners and are mentioned only to describe what stunt-double stands in for or is compared with.
