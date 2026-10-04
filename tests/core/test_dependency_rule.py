"""Enforces the layering: core imports nothing external; providers depend on
core only and never on each other."""

from __future__ import annotations

import ast
from pathlib import Path

import dev_double

PKG = Path(dev_double.__file__).parent
CORE = PKG / "core"
PROVIDERS = PKG / "providers"

BANNED_IN_CORE = (
    "httpx", "fastapi", "pydantic", "uvicorn", "starlette", "needle",
    "time", "uuid", "os", "datetime", "random",
    "dev_double.providers", "dev_double.apps",
)
ALLOWED_IN_CORE = {"__future__", "base64", "dataclasses", "typing", "math", "json", "asyncio", "dev_double"}


def _module_name(path: Path) -> str:
    rel = path.relative_to(PKG.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _imports(path: Path) -> list[str]:
    """Absolute module names imported by a file (relative imports resolved)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    here = _module_name(path)
    package = here if path.name == "__init__.py" else here.rsplit(".", 1)[0]
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                base = base[: len(base) - node.level + 1]
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = node.module or ""
            found.append(mod)
            if node.level and not node.module:  # from . import x
                found.extend(f"{mod}.{alias.name}" for alias in node.names)
    return found


def _within(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(prefix + ".")


def test_core_has_no_external_or_outer_layer_imports():
    violations = []
    for path in CORE.rglob("*.py"):
        for mod in _imports(path):
            if any(_within(mod, banned) for banned in BANNED_IN_CORE):
                violations.append(f"{path.relative_to(PKG)}: {mod}")
            elif mod.split(".")[0] not in ALLOWED_IN_CORE:
                violations.append(f"{path.relative_to(PKG)}: {mod} (not in the stdlib allowlist)")
            elif _within(mod, "dev_double") and not _within(mod, "dev_double.core"):
                violations.append(f"{path.relative_to(PKG)}: {mod}")
    assert not violations, "core must not import:\n" + "\n".join(violations)


def test_providers_depend_on_core_only_never_on_each_other():
    violations = []
    for provider in (p for p in PROVIDERS.iterdir() if p.is_dir() and p.name != "__pycache__"):
        own = f"dev_double.providers.{provider.name}"
        for path in provider.rglob("*.py"):
            for mod in _imports(path):
                if _within(mod, "dev_double.apps"):
                    violations.append(f"{path.relative_to(PKG)}: {mod}")
                elif _within(mod, "dev_double.providers") and not _within(mod, own):
                    violations.append(f"{path.relative_to(PKG)}: {mod}")
    assert not violations, "providers must not import:\n" + "\n".join(violations)


def test_the_rule_checker_sees_relative_imports(tmp_path):
    # Guard against a checker that silently finds nothing.
    mods = _imports(CORE / "decision" / "decider_basic_impl.py")
    assert "dev_double.core.decision.i_engine" in mods
    assert "dev_double.core.decision.prompts" in mods
