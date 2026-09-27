"""Labeled checks for every use case, to pick a model per use case.

    uv run python evals/run.py --model qwen2.5:7b
    uv run python evals/run.py --model qwen2.5:3b --model qwen2.5:7b            # side by side
    uv run python evals/run.py --model laya=http://localhost:8000               # a System 1 server (Kev, Laya)
    uv run python evals/run.py --model needle                                   # needs `pip install cactus-needle`
    uv run python evals/run.py --model qwen2.5:7b --only gate --only guard      # some use cases

Per use case we report how often the top answer is acceptable (acc), the mean
probability on acceptable answers (p_ok, a rough calibration signal), how many
misses were confidently wrong (p_ok < 0.1), and median latency. A model is
"good enough" for a use case at >= 85% accuracy (GOOD_ENOUGH).

This script is a composition root: it picks providers and wires them into the
core decision service, like the server does.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import cases
from stunt_double.core.decision.decider_basic_impl import DeciderBasicImpl
from stunt_double.core.decision.decision_service_basic_impl import DecisionServiceBasicImpl
from stunt_double.core.decision.i_decider import IDecider
from stunt_double.core.decision.t_classify import TClassifyRequest
from stunt_double.core.decision.t_extract import TExtractField, TExtractRequest
from stunt_double.core.decision.t_gate import TGateRequest, TToolCall
from stunt_double.core.decision.t_guard import TGuardRequest
from stunt_double.core.decision.t_judge import TJudgeRequest
from stunt_double.core.decision.t_rerank import TRerankRequest
from stunt_double.core.decision.t_route import TRouteRequest
from stunt_double.providers.openai.decision.engine_openai_impl import EngineOpenAIImpl
from stunt_double.providers.std.decision.clock_std_impl import ClockStdImpl
from stunt_double.providers.std.decision.id_provider_std_impl import IdProviderStdImpl

GOOD_ENOUGH = 0.85
CONFIDENTLY_WRONG = 0.1

Svc = DecisionServiceBasicImpl


@dataclass
class Case:
    name: str
    run: Callable[[Svc], Awaitable[tuple[str, float]]]  # -> (top answer, probability on acceptable answers)
    ok: set[str]


CASES: dict[str, list[Case]] = {}


def add(group: str, name: str, ok: set[str] | str, run) -> None:
    CASES.setdefault(group, []).append(Case(name, run, {ok} if isinstance(ok, str) else set(ok)))


def short(text: str, n: int = 50) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


# ---------------------------------------------------------------- build cases

for text, fires in cases.GUARD:

    async def guard(svc: Svc, text=text, fires=fires):
        r = await svc.guard(TGuardRequest(input=text))
        p_ok = statistics.mean(
            c.probability if name in fires else 1 - c.probability for name, c in r.checks.items()
        )
        return ",".join(sorted(r.flagged)) or "none", p_ok

    add("guard", short(text), ",".join(sorted(fires)) or "none", guard)


def choice(group: str, name: str, ok, call) -> None:
    ok_set = {ok} if isinstance(ok, str) else set(ok)

    async def run(svc: Svc):
        top, probs = await call(svc)
        return top, sum(probs.get(k, 0) for k in ok_set)

    add(group, name, ok_set, run)


for text, ok in cases.ROUTE:

    async def route(svc: Svc, text=text):
        r = await svc.route(TRouteRequest(input=text))
        return r.route, r.probabilities

    choice("route", short(text), ok, route)

for context, tool, args, ok in cases.GATE:

    async def gate(svc: Svc, context=context, tool=tool, args=args):
        r = await svc.gate(TGateRequest(tool_call=TToolCall(name=tool, arguments=args), context=context))
        return r.decision, r.probabilities

    choice("gate", f"{tool} <- {short(context[6:], 35)}", ok, gate)

for text, ok in cases.TRIAGE:

    async def classify(svc: Svc, text=text):
        r = await svc.classify(TClassifyRequest(input=text, labels=cases.TRIAGE_LABELS))
        return r.results[0].label, r.results[0].probabilities

    choice("classify", short(text), ok, classify)

for task, output, reference, good in cases.JUDGE:

    async def judge(svc: Svc, task=task, output=output, reference=reference, good=good):
        r = await svc.judge(TJudgeRequest(input=task, output=output, reference=reference))
        verdict = "good" if r.score >= 2.5 else "bad" if r.score <= 1.5 else "unsure"
        p_ok = sum(p for level, p in r.probabilities.items() if (int(level) >= 3) == good)
        return verdict, p_ok

    add("judge", f"{'good' if good else 'bad '}: {short(output, 45)}", "good" if good else "bad", judge)

for query, docs, best in cases.RERANK:

    async def rerank(svc: Svc, query=query, docs=docs, best=best):
        r = await svc.rerank(TRerankRequest(query=query, documents=docs))
        scores = {x.index: max(x.relevance_score, 0.0) for x in r.results}
        return str(r.results[0].index), scores[best] / (sum(scores.values()) or 1)

    add("rerank", query, str(best), rerank)


def _norm(value: Any) -> Any:
    if isinstance(value, str):
        return "".join(ch for ch in value.casefold() if ch.isalnum())
    return value


def _field_ok(got: Any, want: Any) -> bool:
    for w in want if isinstance(want, tuple) else (want,):
        if w is None or isinstance(w, bool):
            ok = got is w
        elif isinstance(w, (int, float)):
            ok = isinstance(got, (int, float)) and not isinstance(got, bool) and abs(got - w) < 0.01
        else:
            ok = isinstance(got, str) and _norm(got) == _norm(w)
        if ok:
            return True
    return False


for text, fields, want in cases.EXTRACT:

    async def extract(svc: Svc, text=text, fields=fields, want=want):
        spec = {name: TExtractField(f["type"], f.get("description"), f.get("options")) for name, f in fields.items()}
        got = (await svc.extract(TExtractRequest(text, spec))).values
        wrong = [name for name in want if not _field_ok(got.get(name), want[name])]
        # p_ok here is the share of fields extracted correctly.
        return ("all" if not wrong else "wrong: " + ", ".join(f"{n}={got.get(n)!r}" for n in wrong)), 1 - len(wrong) / len(want)

    add("extract", short(text), "all", extract)


# ---------------------------------------------------------------- runner


def build(target: str, base_url: str) -> IDecider:
    if target == "needle":
        from stunt_double.providers.needle.decision.decider_needle_impl import DeciderNeedleImpl

        return DeciderNeedleImpl()
    if "=" in target:
        from stunt_double.providers.systemone.decision.decider_system_one_impl import DeciderSystemOneImpl

        name, url = target.split("=", 1)
        return DeciderSystemOneImpl(name, url, model=name)
    # One call at a time, so latency is per call.
    return DeciderBasicImpl(EngineOpenAIImpl(base_url, target), max_concurrency=1)


async def run_model(target: str, base_url: str) -> dict[str, Any]:
    svc = DecisionServiceBasicImpl(build(target, base_url), ClockStdImpl(), IdProviderStdImpl())
    report: dict[str, Any] = {}
    try:
        await svc.route(TRouteRequest(input="hello"))  # warm up: the first call loads the model
        for group, group_cases in CASES.items():
            hits, p_oks, latencies, misses, confident_misses = 0, [], [], [], 0
            for c in group_cases:
                start = time.perf_counter()
                top, p_ok = await c.run(svc)
                latencies.append((time.perf_counter() - start) * 1000)
                p_oks.append(p_ok)
                if top in c.ok:
                    hits += 1
                else:
                    confident_misses += p_ok < CONFIDENTLY_WRONG
                    misses.append(f"{c.name}  -> {top} (want {'/'.join(sorted(c.ok))}, p_ok {p_ok:.2f})")
            report[group] = {
                "hits": hits,
                "n": len(group_cases),
                "p_ok": statistics.mean(p_oks),
                "confident_misses": confident_misses,
                "ms": statistics.median(latencies),
                "misses": misses,
            }
    finally:
        await svc.aclose()
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model on the OpenAI-compatible server, NAME=URL for a /v1/systemone server (Kev, Laya), or needle.",
    )
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--only", action="append", help="Run only these use cases (e.g. --only gate).")
    ap.add_argument("--json", help="Also write the full report to this file.")
    args = ap.parse_args()
    for group in list(CASES):
        if args.only and group not in args.only:
            del CASES[group]

    reports = {m.split("=")[0]: asyncio.run(run_model(m, args.base_url)) for m in args.model}
    names = list(reports)

    print(f"{'use case':<10}" + "".join(f"{m:>34}" for m in names))
    print(f"{'':<10}" + "".join(f"{'acc   p_ok  conf.wrong    ms':>34}" for _ in names))
    for group in CASES:
        row = f"{group:<10}"
        for m in names:
            g = reports[m][group]
            mark = "ok" if g["hits"] / g["n"] >= GOOD_ENOUGH else "--"
            row += f"{mark:>6} {g['hits']:>2}/{g['n']:<3}{g['p_ok']:>6.2f}{g['confident_misses']:>8}{g['ms']:>10.0f}"
        print(row)
    print(f"\nok = good enough (>= {GOOD_ENOUGH:.0%} correct); conf.wrong = misses with p_ok < {CONFIDENTLY_WRONG}")
    for m in names:
        hits = sum(g["hits"] for g in reports[m].values())
        n = sum(g["n"] for g in reports[m].values())
        print(f"\n{m}: {hits}/{n} correct")
        for group, g in reports[m].items():
            for miss in g["misses"]:
                print(f"  miss [{group}] {miss}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2)


if __name__ == "__main__":
    main()
