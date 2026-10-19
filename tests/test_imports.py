import ast
import importlib
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1] / "cs2model"
FILES = sorted(PKG.rglob("*.py"))


def _imports(path):
    """(module, name) for every `from cs2model... import name`, including function-level imports."""
    tree = ast.parse(path.read_text())
    pkg = ".".join(path.relative_to(PKG.parent).with_suffix("").parts[:-1])
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level:
                base = pkg.split(".")[: len(pkg.split(".")) - node.level + 1]
                mod = ".".join(base + ([mod] if mod else []))
            if mod.startswith("cs2model"):
                for a in node.names:
                    yield node.lineno, mod, a.name


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(PKG.parent)))
def test_internal_imports_resolve(path):
    """Catches a dashboard/app import of a name that no longer exists (or doesn't exist yet) in the target module."""
    missing = []
    for line, mod, name in _imports(path):
        m = importlib.import_module(mod)
        if name != "*" and not hasattr(m, name):
            try:
                importlib.import_module(f"{mod}.{name}")
            except ImportError:
                missing.append(f"line {line}: {mod}.{name}")
    assert not missing, missing
