# dev-double

A local stand-in for System 1 decision models, so you can build your harness now, on a model you run locally, and swap in a stronger one later.

## Why this exists

System 1 models such as Jev, Kev, and Laya took everyone by storm. They don't write text; they return a typed decision with a calibrated probability in milliseconds. Teams use them for model routing, guardrails, tool-call gating, inbox triage, reranking, LLM evals, bulk labeling, real-time control, and confidence gates.

The inputs these decisions run on are often sensitive: customer messages, internal documents, the arguments of a tool call. You may not want that data going to an external API. And a System 1 model you can run locally, or fully trust, may not be good enough or available yet: it may not run on your hardware, you may not trust it yet, or a stronger open-weight one may not exist yet. Waiting means you can't build the harness around it in the meantime.

dev-double fills that gap. It serves decision endpoints with the same shape of output (a typed answer, a probability for every option, and a confidence score), and a model you already run locally (or already trust) does the work underneath. Your team writes the routing, guardrails, gates, and thresholds now. When a stronger local or trusted model is available, you swap the engine and keep everything you built.

Open-weight System 1 models (Kev, Laya, and the ones [Ollaya](https://github.com/ollaya-dev/ollaya) and Ollama serve) can run on your own machine. If one fits your needs and your hardware, put it behind dev-double (`--engine systemone`) and keep the same endpoints; on our tests the best of them beat a general 7B model at a third of the latency. When none fits yet, dev-double runs on a general model you already have.

It also serves Ollama's System One API (`POST /v1/systemone`), so code written for a System 1 model on Ollama or Ollaya runs on dev-double unchanged, and later runs on that model by changing only the base URL.

Like a test double in your unit tests, it stands in for the real dependency. It's slower and less accurate than a purpose-built decision model, but it has the same shape, so your code doesn't change when the real one arrives, and the work goes on.

**Any model behind an OpenAI-compatible API works as the engine.** We develop and test with Qwen 2.5 on a local GPU through Ollama. vLLM, llama.cpp, and LM Studio work the same way, and so can a hosted small model, such as one from OpenAI or Anthropic, if you're comfortable sending your data to that provider (see [Choosing a model](#choosing-a-model)).

## Quick start

```bash
# 1. A local model server (Ollama >= 0.12.11 returns the token probabilities we need)
ollama pull qwen2.5:7b      # ~4.7 GB; qwen2.5:3b (~1.9 GB) if you're short on memory

# 2. Install and check the model works
pip install dev-double
dev-double doctor

# 3. Serve
dev-double serve            # http://127.0.0.1:8787, interactive docs at /docs
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

Faster, and more accurate on most decision use cases: a System 1 model on [Ollaya](https://github.com/ollaya-dev/ollaya) (see [Which model for which use case](#which-model-for-which-use-case)):

```bash
ollaya pull winnow:e4b      # ~8 GB
dev-double serve --engine systemone --systemone-url http://localhost:11435 --model winnow:e4b
```

No model server yet? `dev-double serve --engine mock` answers from word overlap. It's instant and deterministic; use it for CI, never for quality.

## Endpoints

| Endpoint | Use it for | Returns |
|---|---|---|
| `POST /v1/systemone` | Anything: ask several typed questions about one input, optionally with images. Ollama-compatible | per question: `noul`, `choice`, or `score` answer |
| `POST /v1/route` | Model routing: small, medium, or large model? Or your own routes | `route`, `probabilities`, `confidence` |
| `POST /v1/guard` | Screen input for prompt injection, abuse, off-topic use | `allowed`, `flagged`, per-policy `probability` |
| `POST /v1/gate` | Should an agent's tool call be allowed, confirmed, or denied? | `decision`, `probabilities`, `confidence` |
| `POST /v1/classify` | Inbox triage, intent detection, bulk labeling (`inputs: [...]`) | per item: `label`, `probabilities`, `confidence` |
| `POST /v1/judge` | LLM evals: score an output against a rubric | `score`, `normalized`, `level`, `probabilities` |
| `POST /v1/rerank` | Order documents by relevance (Cohere-style format, also `/v2/rerank`) | `results: [{index, relevance_score}]` |
| `POST /v1/extract` | Pull typed fields out of text: invoices, tickets, bookings, tool-call arguments | `values`, per-field `value` and `confidence` |
| `POST /v1/decide` | Deprecated, removed in 0.3: use `/v1/systemone` | per question: `binary`, `choice`, or `scale` answer |

Every response carries `meta`: engine, model, latency, token usage, and `warnings`, which tell you when an answer is less trustworthy than usual.

### `/v1/systemone`: the core

The request and response follow [Ollama's System One API](https://docs.ollama.com/api/systemone). Three question types cover every use case above:

```json
{
  "model": "qwen2.5:7b",
  "state": {"ticket": "My flight was cancelled and nobody is answering the phone!!"},
  "questions": {
    "wants_refund": {"type": "noul", "instructions": "Does the customer ask for their money back?"},
    "intent": {
      "type": "choice",
      "instructions": "What does the customer want?",
      "criteria": {
        "refund": "Money returned.",
        "rebooking": "A replacement flight.",
        "information": "Only information."
      }
    },
    "frustration": {
      "type": "score",
      "instructions": "How frustrated is the customer?",
      "criteria": ["Calm.", "Concerned but civil.", "Very angry."]
    }
  }
}
```

```json
{
  "model": "qwen2.5:7b",
  "answers": {
    "wants_refund": {"type": "noul", "noul": 0.31},
    "intent": {"type": "choice", "choice": "rebooking", "probabilities": {"refund": 0.22, "rebooking": 0.71, "information": 0.07}, "confidence": 0.306},
    "frustration": {"type": "score", "score": 1.62, "legend": {"0": "Calm.", "1": "Concerned but civil.", "2": "Very angry."}, "probabilities": {"0": 0.04, "1": 0.3, "2": 0.66}, "confidence": 0.304}
  },
  "usage": {"input_tokens": 512, "output_tokens": 3},
  "meta": {"...": "..."}
}
```

- **noul**: the probability that the answer is true (a number, not a boolean). Optional `criteria` with `"true"` and `"false"` say what each means.
- **choice**: 2 to 26 unordered options; a `null` description means the key describes itself. `choice` is the most likely key.
- **score**: 2 to 26 ordered levels, lowest first. `score` is the probability-weighted level (it can fall between levels); `legend` maps each level to its description.
- **confidence**: `1 - H(p) / ln(n)`, as in Ollama's API: 1 when all probability is on one answer, 0 when spread evenly. The other endpoints use a simpler measure with the same two ends.
- **model**: required, and echoed back. The server's configured engine answers; `meta.model` says which, with a warning when it differs from the request. Start the server with `--honor-request-model` to let each request pick the model, which is handy for comparing models.
- **images**: base64 PNG, JPEG, or WebP (no URLs or `data:` prefixes), shared by all the questions. They need an engine that can see: `--vision-model qwen2.5vl:7b` with an OpenAI-compatible server (text-only requests still use `--model`), or a System 1 server with a vision model, such as Clef. Other engines answer 400.
- **meta**: an extra field Ollama doesn't send. Clients that ignore unknown fields are unaffected.

Errors come back as `{"error": "..."}`: 400 for an invalid request, 413 above 64 KiB (32 MiB with images), 500 when the model fails. Questions in one request run in parallel; a System 1 engine gets them all in one call.

From Python:

```python
from dev_double.client import DevDouble, image_base64

sd = DevDouble()
r = sd.systemone("qwen2.5:7b", "Our checkout has returned 500 errors since 9am.", {
    "label": {"type": "choice", "instructions": "Which label fits this ticket?",
              "criteria": {"billing": "Payments and refunds", "bug": "Software errors", "account": "Login and account access"}},
})
r["answers"]["label"]["choice"]   # "bug"
```

The older `/v1/decide` (`input`, and `binary` / `choice` / `scale` questions) still works but is deprecated: it answers with a `Deprecation: true` header and a warning, and will be removed in 0.3.

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
from dev_double.client import DevDouble

sd = DevDouble()
g = sd.gate("send_email", {"to": "all-staff@corp.com"}, context="User asked to draft a reply to Bob.")
if g["decision"] == "allow" and g["confidence"] > 0.9:
    run_tool()
elif g["decision"] == "deny" and g["confidence"] > 0.9:
    refuse()
else:
    ask_human()
```

## How it works

Each question becomes one prompt that asks the question, shows the input, asks the question again, and ends with "reply with exactly one label": the option name for choices, a digit for scale levels, yes or no for binary. The model generates at most a few tokens. dev-double reads the probabilities the model assigned to each label's tokens (`logprobs`) and renormalizes them. So the probabilities come from the model itself, not from a number the model writes down. That is also why it's much faster than asking an LLM for JSON.

Two details that matter with small models:

- **Option names, not letters.** Lettered options (A, B, C) make small models over-pick "B". Answering with the option name avoids that. When two names share their first token ("re" in *refund* and *rebooking*), the next token's probabilities split them.
- **Question before and after the input.** On our test cases this raised accuracy from 6/9 to 8/9 for yes/no questions.

For `/v1/extract`, enum and boolean fields are scored like choice and binary questions; all string, number, and integer fields are generated together as one JSON object, and each value's confidence comes from the probabilities of the tokens that spell it. Engines that can't generate text (System 1 servers) leave those fields `null` with a warning.

If the model server returns no logprobs, or the model answers with something that isn't a label, you still get an answer, plus a warning in `meta.warnings`.

## Fidelity: what differs from a real decision model

Build your harness knowing these:

- **Latency.** A local 7B model takes roughly 100–150 ms per question on a laptop GPU with Ollama 0.35 (a rerank of 4 documents about 0.3 s), several seconds on CPU. Purpose-built decision models aim much lower. Don't design around the slowness: no caching or batching workarounds a real engine won't need. `meta.latency_ms` shows what you're paying.
- **Calibration.** Probabilities from a small general model are only roughly calibrated. Thresholds you tune now (like `confidence > 0.9`) will need re-tuning on the real engine. Keep a labeled set of examples so you can re-tune in an afternoon.
- **Accuracy.** Small models miss nuance. See [Measured results](#measured-results) for what works and what doesn't, and run the evals on your own model before relying on a use case.
- **Repeatability.** Answers are deterministic while a model stays loaded, but can shift slightly after the model server reloads it. Borderline cases may flip. Don't write tests that assert exact probabilities from a real model; use `--engine mock` for that.
- **Limits.** With a System 1 engine, up to 26 options or levels. With an OpenAI-compatible engine, up to 20 options and 10 levels, because the model server returns at most 20 token probabilities. Option keys must differ ignoring case and punctuation. Images only on `/v1/systemone`.
- **Models.** Use non-thinking instruct models. Reasoning models that write out their thinking first (`<think>`) don't work, because the answer is no longer in the first tokens.

## Choosing a model

dev-double talks to any server with an OpenAI-compatible `/v1/chat/completions` endpoint. What matters is whether that server returns **logprobs** (the probability of each token), because the probabilities in every answer come from them.

| Engine | Logprobs | Notes |
|---|---|---|
| Ollama ≥ 0.12.11 (default) | yes | Tested with `qwen2.5:7b` and `qwen2.5:3b` on a local GPU. |
| vLLM, llama.cpp server, LM Studio | yes | `--base-url http://localhost:8000/v1` (vLLM) or `:8080/v1` (llama.cpp). |
| OpenAI and other hosted APIs | on non-reasoning models | `--base-url https://api.openai.com/v1 --api-key ... --model <small non-reasoning model>`. Not tested yet. |
| Anthropic | no | Via its OpenAI-compatible endpoint (`--base-url https://api.anthropic.com/v1/`). Not tested yet. With no logprobs, answers are 0 or 1 and `meta.warnings` says so. |
| System 1 server (`--engine systemone`) | native | Ollaya (`--systemone-url http://localhost:11435`), Ollama ≥ 0.35 (`http://localhost:11434`), or a Kev or Laya server. Set `--model` to the System 1 model. Tested on Ollaya: `winnow:e4b`, `jeb:4b`, `jevk5`, `decider`, `laya:typed-decisions`; on Ollama: `tev1:4b`; earlier, Kev 0.8B and Laya on their own servers. |
| Cactus Needle (`--engine needle`) | one confidence per pick | Tiny on-device tool-calling model. Tested; not a good fit (see below). On Windows, cactus-needle 3.0.5 asks for an engine build that isn't published: download the 3.0.1 engine from Hugging Face and set `NEEDLE3_LIB_PATH`. |

Good local choices: Qwen 2.5 (7B recommended, 3B if memory is tight), Llama 3.x, Gemma 3, Phi-4-mini, Mistral.

A hosted model sends your input to that provider. Use one only if that data is allowed to leave your machine. The main point of dev-double is to keep sensitive input local.

### Which model for which use case

Most teams need one use case, not all nine, so pick per use case. `evals/run.py` holds 140 labeled cases, 20 per use case, including deliberately hard ones (rerank distractors that share the query's words, answers that are subtly wrong, extraction fields that aren't in the text and must come back empty). A model counts as **good enough** for a use case at 17/20 (85%) or better.

| Use case | Recommended | Also good enough | Not good enough |
|---|---|---|---|
| guard (injection, abuse) | **qwen2.5:7b** (18/20) | decider 18 (but 1.6 s per input here) | winnow:e4b 16, tev1:4b 16, jeb:4b 15, laya:typed-decisions 15, Kev 13, qwen2.5:3b 12, Laya 12, jevk5 11 |
| route | **jevk5** (20/20) | jeb:4b 19, tev1:4b 19, qwen2.5:7b 18, winnow:e4b 17, decider 17 | qwen2.5:3b 16, Kev 14, laya:typed-decisions 12, Laya 9 |
| gate | **jeb:4b** (18/20, no confident mistakes) | qwen2.5:7b 17, jevk5 17, winnow:e4b 17 (see caution below) | tev1:4b 15, decider 14, Laya 14, qwen2.5:3b 11, laya:typed-decisions 11, Kev 10 |
| classify (inbox triage) | **winnow:e4b** (20/20) | tev1:4b 20, jeb:4b 19, jevk5 19, decider 18, Kev 0.8B 17 | qwen2.5:7b 16, qwen2.5:3b 15, laya:typed-decisions 14, Laya 12 |
| judge | **winnow:e4b** (20/20) | jeb:4b 19, qwen2.5:7b 18, jevk5 17 | decider 16, tev1:4b 16, qwen2.5:3b 15, Kev 10, Laya 9, laya:typed-decisions 9 |
| rerank | **qwen2.5:3b** (20/20) | qwen2.5:7b 20, decider 20, Laya 20, laya:typed-decisions 20, winnow:e4b 19, Kev 19, jeb:4b 18, jevk5 18, tev1:4b 18 | Needle 10 |
| extract | **qwen2.5:7b** (16/20, 95% of fields); close, see below | none | qwen2.5:3b 15 (91% of fields), Needle 4 (69%). System 1 models can't write text |

Full results, on an RTX 4070 Laptop GPU (8 GB), one model loaded at a time. The general models ran on Ollama 0.35.1 through dev-double's prompts:

| Use case | qwen2.5:7b | qwen2.5:3b | Needle |
|---|---|---|---|
| guard | **18**/20 | 12 | 0 |
| route | **18** | 16 | 8 |
| gate | **17** | 11 | 7 |
| classify | **16** | 15 | 5 |
| judge | **18** | 15 | 2 |
| rerank | **20** | **20** | 10 |
| extract | **16** | 15 | 4 |
| **total** | **123/140** | 114 | 36 |
| median latency per call | 85–150 ms | 55–110 ms | 55–390 ms |

The System 1 models answered natively (`--engine systemone`): five on Ollaya 0.9.0, tev1:4b on Ollama 0.35.1, Kev and Laya earlier on their own servers. They can't write text, so they weren't scored on extract:

| Use case | winnow:e4b | jeb:4b | tev1:4b | jevk5 | decider | laya:typed-decisions | Kev 0.8B | Laya |
|---|---|---|---|---|---|---|---|---|
| guard | 16/20 | 15 | 16 | 11 | **18** | 15 | 13 | 12 |
| route | 17 | 19 | 19 | **20** | 17 | 12 | 14 | 9 |
| gate | 17 | **18** | 15 | 17 | 14 | 11 | 10 | 14 |
| classify | **20** | 19 | **20** | 19 | 18 | 14 | 17 | 12 |
| judge | **20** | 19 | 16 | 17 | 16 | 9 | 10 | 9 |
| rerank | 19 | 18 | 18 | 18 | **20** | **20** | 19 | **20** |
| **total** | **109/120** | 108 | 104 | 102 | 103 | 81 | 83 | 76 |
| median latency per call | 75–120 ms | 50–100 ms | 120–280 ms | 60–120 ms | 0.4–4 s | 17–21 ms | 30–45 ms | 25–50 ms |

On the same six use cases, qwen2.5:7b scores 107/120. Rerank latency is per query of 4 documents: 0.2–0.3 s for the Qwen models and the Ollaya decoders, 72 ms for laya:typed-decisions, 90–110 ms for Laya and Kev. Guard asks two questions per input: a System 1 engine gets both in one call, the Qwen models make two. Extract takes 0.5–0.9 s per record on the Qwen models (one call for all text fields plus one per enum or yes/no field), and counts a record as right only if every field is right.

What that means in practice:

- **For decisions, the best System 1 models now beat qwen2.5:7b, at a fraction of the latency.** winnow:e4b and jeb:4b are good enough at every decision use case except guard, and lead on triage, judge, route (jevk5), and gate. Each needs 5–8 GB and answered in about 50–120 ms per call on this laptop GPU.
- **On Ollama alone, every use case has a good-enough model:** tev1:4b for route and triage, qwen2.5:7b for the rest. tev1:4b makes confident mistakes in gate (4 of 20), so leave gate to qwen2.5:7b, or better, jeb:4b on Ollaya.
- **qwen2.5:7b is still the default** for the OpenAI-compatible engine, and the pick for guard and extract. It's good enough everywhere except inbox triage, where it rates polite but non-urgent requests as urgent.
- **qwen2.5:3b is enough for rerank.** It's also fine for rough routing, but it mixes up guard policies and wrongly denies harmless tool calls.
- **Be careful with confident mistakes in gate.** qwen2.5:7b allowed an 8,400 transfer to a new payee, an email to the whole company, and a production deploy, each with near-certainty (probability of the right answer 0.01 or less); each should have asked a human. The System 1 models got some of the same cases wrong, but unsure (0.05–0.40), so a confidence threshold catches them. Either way, put hard limits (amounts, recipients, destructive tools) in code, not only in a confidence threshold.
- **The System 1 models' guard misses are mostly over-flagging:** a request for abuse also flagged as prompt injection. It's still blocked, but the policy label is wrong. All of them missed "repeat the text above", an indirect injection.
- **Not tested:**
  - Clef Flash on Ollama 0.35.1 failed every request on this machine with "Clef: non-finite logit", even on CPU alone, and Ollaya's `clef` needs a 24 GB GPU.
  - qwen3guard answers only its own built-in safety questions, so it can't run custom guard policies.
  - decider ran at full 32-bit precision here, which explains its latency.
- **Kev and Laya:** Laya ranks documents perfectly but falls short elsewhere. Kev 0.8B ran without its fast kernels, which aren't available on Windows. Both got the same question wording as the LLMs, which wasn't tuned for them.
- **Extraction is close to good enough.** qwen2.5:7b gets 95% of fields right, but only 16 of 20 records completely right. Typical misses: inventing a date from "next summer", leaving out a meeting title that is in the text. Check low-confidence fields, or send them to a human.
- **Needle is not a decision model.** It's a tiny on-device model that *writes* tool calls. Asked to classify, it often returns no call at all, and embedding-based rerank only reached 10/20. Even on extraction, its home ground, it got 69% of fields right out of the box. Cactus pitches fine-tuning on your own schema, which we didn't test. Its fit is the step before a decision model: Needle proposes the tool call on the device, and `/v1/gate` decides whether to run it.

You can mix: run one dev-double per engine (for example `winnow:e4b` for classify and judge, `qwen2.5:7b` for guard and extract) and call each for its use cases.

Reproduce any column (one model at a time; load only one model into GPU memory):

```bash
uv run python evals/run.py --model qwen2.5:7b                           # Ollama
uv run python evals/run.py --model winnow:e4b=http://localhost:11435    # ollaya serve; ollaya pull winnow:e4b
uv run python evals/run.py --model tev1:4b=http://localhost:11434       # Ollama >= 0.35 System 1 model
uv run python evals/run.py --model laya=http://localhost:8000           # laya-serve
uv run python evals/run.py --model kev=http://localhost:8009            # python -m kev.serve --run jaredpalmer/kev-0.8b --port 8009
uv run --extra needle python evals/run.py --model needle                # set NEEDLE_TELEMETRY=0
```

Twenty cases per use case is still small; add cases from your own domain before relying on a result.

## Switching to the real engine later

1. Keep dev-double behind your own small interface, for example `Decisions.route(...)`, `Decisions.gate(...)`.
2. Record real traffic while you develop: `dev-double serve --record calls.jsonl`. Each line holds the request, response, and latency.
3. When a stronger local or trusted engine is available, implement the same interface with its SDK, replay `calls.jsonl` through both, and compare decisions and latency before switching.

The rest of your harness (thresholds aside) does not change.

## Configuration

| Flag | Environment variable | Default |
|---|---|---|
| `--engine` | `DEV_DOUBLE_ENGINE` | `openai` (any OpenAI-compatible server), `mock` (word overlap, for CI), `systemone` (a System 1 server: Ollama ≥ 0.35, Ollaya, Kev, or Laya), or `needle` (Cactus Needle on-device; `pip install "dev-double[needle]"`) |
| `--base-url` | `DEV_DOUBLE_BASE_URL` | `http://localhost:11434/v1` (Ollama) |
| `--model` | `DEV_DOUBLE_MODEL` | `qwen2.5:7b`. With `--engine systemone`, set it to the System 1 model, such as `clef-flash` |
| `--vision-model` | `DEV_DOUBLE_VISION_MODEL` | none, so requests with images are refused. For example `qwen2.5vl:7b` |
| `--honor-request-model` | `DEV_DOUBLE_HONOR_REQUEST_MODEL` | off: the request's `model` field is ignored |
| `--api-key` | `DEV_DOUBLE_API_KEY` | none |
| `--max-concurrency` | `DEV_DOUBLE_MAX_CONCURRENCY` | `4` |
| `--systemone-url` | `DEV_DOUBLE_SYSTEMONE_URL` | `http://localhost:8000`. Ollama is `http://localhost:11434`, Ollaya `http://localhost:11435` |
| | `DEV_DOUBLE_TIMEOUT` | `120` seconds per model call |
| `--record` | `DEV_DOUBLE_RECORD` | off |
| `--host`, `--port` | | `127.0.0.1`, `8787` |

For vLLM: `--base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct`. For llama.cpp server: `--base-url http://localhost:8080/v1`.

`--engine systemone` and `--engine needle` send the questions to a model that answers typed questions natively, with no dev-double prompt (a System 1 server gets all of a request's questions in one call), so you can compare dev-double with the real thing behind the same endpoints. Needle has no probability for every option: it returns one confidence for its pick, and the other options share the rest evenly. It reranks with embeddings. Telemetry is off by default (`NEEDLE_TELEMETRY=0`, `DO_NOT_TRACK=1`), and `NEEDLE3_LIB_PATH` points it at a local build.

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

- **`dev_double/core/decision/`**: the domain, with no third-party imports (no httpx, pydantic, time, or os). It holds the interfaces (`IEngine`, `IDecider`, `IClock`, `IIdProvider`, `IDecisionService`, and the optional capabilities `IReranker`, `IGenerator`, `IRecordReader`, `IExtractor`), the domain types (`TChoiceQuestion`, `TRouteRequest`, ...), the prompts, the label scoring, and the use cases (`DecisionServiceBasicImpl`). `DeciderBasicImpl` turns a question into a prompt and asks an `IEngine`.
- **`dev_double/providers/<name>/decision/`**: one folder per technology. Each implements core interfaces and depends only on core, never on another provider. `openai` (`EngineOpenAIImpl`), `mock` (`EngineMockImpl`, plus a deterministic clock and id provider for tests), `std` (the real clock and UUIDs), `systemone` (`DeciderSystemOneImpl`), and `needle` (`DeciderNeedleImpl`).
- **`dev_double/apps/`**: `server/` (FastAPI, the pydantic transport schemas, the recorder), `cli/`, `client/` (the `DevDouble` SDK), and `composition.py`, the only place that picks implementations from the settings.

To add an engine:

- If it scores labels from a prompt, implement `IEngine` in `providers/<name>/decision/engine_<name>_impl.py`. Implement `IGenerator` too if it can generate short text, so `/v1/extract` can fill string and number fields.
- If it answers typed questions natively, implement `IDecider` in `providers/<name>/decision/decider_<name>_impl.py`, and also `IReranker` if it can rerank without yes/no questions, or `IExtractor` if it extracts whole records.

Then add a branch in `apps/composition.py` and the name to `ENGINES` in `apps/config.py`, and put its tests in `tests/providers/<name>/`. `tests/` mirrors `src/`. `tests/core/test_dependency_rule.py` fails if core imports anything external or if one provider imports another.

## Credits

Built by Marijus Masteika with [Claude](https://claude.com/claude-code) (Anthropic) as a coding partner. Claude co-authored the commits.

## License and trademarks

MIT. dev-double is an independent project. Its use-case endpoints are its own API design. `/v1/systemone` follows the System One API as published by Ollama in its [OpenAPI specification](https://github.com/ollama/ollama/blob/main/docs/openapi.yaml) (MIT License, Copyright (c) Ollama). dev-double is not affiliated with or endorsed by Ollama, Ollaya, Cloudflare (Clef), TypeSafe AI (Jev), the Kev project, Convai Innovations (Laya), or Cactus Compute (Needle). `--engine systemone` and `--engine needle` are clients, for comparison. Those names belong to their owners and are mentioned only to describe what dev-double stands in for or is compared with.
