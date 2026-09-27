"""Command line: `stunt-double serve` and `stunt-double doctor`."""

from __future__ import annotations

import argparse
import asyncio
import sys

from . import __version__
from .config import Settings
from .core import Decider, Tracker
from .engines import EngineError
from .schemas import BinaryQuestion, ChoiceQuestion


def _add_engine_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--engine", choices=["openai", "fake"], help="Backend engine (default: openai).")
    p.add_argument("--base-url", help="OpenAI-compatible base URL (default: Ollama on localhost).")
    p.add_argument("--model", help="Model name on that server (default: qwen2.5:7b).")
    p.add_argument("--api-key", help="API key for the model server, if it needs one.")
    p.add_argument("--max-concurrency", type=int, help="Parallel model calls (default: 4).")


def _settings(args: argparse.Namespace) -> Settings:
    s = Settings.from_env()
    for name in ("engine", "base_url", "model", "api_key", "max_concurrency", "record"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(s, name, value)
    return s


def serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .server import create_app

    settings = _settings(args)
    print(f"stunt-double {__version__}: engine={settings.engine} model={settings.model}")
    if settings.engine == "openai":
        print(f"  model server: {settings.base_url}")
    if settings.record:
        print(f"  recording calls to: {settings.record}")
    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_level="info")
    return 0


async def _doctor(settings: Settings) -> int:
    engine = settings.build_engine()
    decider = Decider(engine, max_concurrency=1)
    print(f"Checking engine={engine.name} model={engine.model} ...")
    ok = True
    try:
        t = Tracker()
        yes = await decider.ask(
            "The package arrived broken and I want my money back.",
            BinaryQuestion(type="binary", question="Is the customer asking for a refund?"),
            t,
        )
        pick = await decider.ask(
            "My flight was cancelled, please put me on the next one.",
            ChoiceQuestion(
                type="choice",
                question="What does the customer want?",
                options={
                    "refund": "Money returned.",
                    "rebooking": "A replacement flight.",
                    "information": "Only information.",
                },
            ),
            t,
        )
    except EngineError as exc:
        print(f"FAIL  {exc}")
        return 1
    finally:
        await engine.aclose()

    meta = t.meta(engine)
    print(f"  binary: refund? p(yes)={yes.probability}  (expected high)")  # type: ignore[union-attr]
    print(f"  choice: {pick.value}  {pick.probabilities}  (expected rebooking)")  # type: ignore[union-attr]
    print(f"  latency: {meta.latency_ms} ms for 2 questions")
    for w in meta.warnings:
        ok = False
        print(f"WARN  {w}")
    if yes.probability < 0.5 or pick.value != "rebooking":  # type: ignore[union-attr]
        ok = False
        print("WARN  Answers look wrong; this model may be too small for your use case.")
    print("OK" if ok else "Finished with warnings.")
    return 0 if ok else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stunt-double", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="Run the HTTP server.")
    _add_engine_args(p_serve)
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8787)
    p_serve.add_argument("--record", help="Append every call to this JSONL file.")
    p_serve.set_defaults(func=serve)

    p_doc = sub.add_parser("doctor", help="Check that the model server works and returns logprobs.")
    _add_engine_args(p_doc)
    p_doc.set_defaults(func=lambda a: asyncio.run(_doctor(_settings(a))))

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
