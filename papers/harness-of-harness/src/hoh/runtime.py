"""Runtime.check(A_t) — deterministic build / test / import checks."""

from __future__ import annotations

import importlib.util
import py_compile
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional


def compile_python_tree(workspace: Path) -> Dict[str, Any]:
    """Byte-compile all .py files; report failures."""
    workspace = Path(workspace)
    py_files = [
        f
        for f in sorted(workspace.rglob("*.py"))
        if "__pycache__" not in f.parts
    ]
    errors: List[str] = []
    for f in py_files:
        try:
            py_compile.compile(str(f), doraise=True)
        except py_compile.PyCompileError as e:
            errors.append(str(e))
    return {
        "check": "py_compile",
        "n_files": len(py_files),
        "ok": len(errors) == 0,
        "errors": errors,
    }


def _run_unittest(workspace: Path, tests: Path, *, timeout: float) -> Dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", str(tests), "-v"],
        cwd=str(workspace),
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**dict(**{k: v for k, v in __import__("os").environ.items()}), "PYTHONPATH": str(workspace)},
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return {
        "check": "unittest",
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "output": out[-4000:],
        "skipped": False,
    }


def run_pytest(
    workspace: Path,
    tests_dir: Optional[Path] = None,
    *,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """Run pytest if available; otherwise unittest discovery."""
    workspace = Path(workspace)
    tests = Path(tests_dir) if tests_dir else workspace / "tests"
    if not tests.exists():
        return {
            "check": "pytest",
            "ok": False,
            "skipped": True,
            "detail": f"tests dir missing: {tests}",
            "passed": 0,
            "failed": 0,
        }

    # Prefer pytest only if importable
    try:
        import pytest  # noqa: F401
        has_pytest = True
    except ImportError:
        has_pytest = False

    if has_pytest:
        cmd = [sys.executable, "-m", "pytest", "-q", str(tests)]
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=timeout,
                env={
                    **{k: v for k, v in __import__("os").environ.items()},
                    "PYTHONPATH": str(workspace),
                },
            )
            out = (proc.stdout or "") + (proc.stderr or "")
            return {
                "check": "pytest",
                "ok": proc.returncode == 0,
                "returncode": proc.returncode,
                "output": out[-4000:],
                "skipped": False,
            }
        except subprocess.TimeoutExpired:
            return {"check": "pytest", "ok": False, "detail": "timeout", "skipped": False}
        except Exception as e:  # noqa: BLE001
            # fall through to unittest
            fallback_err = str(e)
    else:
        fallback_err = "pytest not installed"

    try:
        result = _run_unittest(workspace, tests, timeout=timeout)
        result["fallback_from"] = fallback_err
        return result
    except subprocess.TimeoutExpired:
        return {"check": "unittest", "ok": False, "detail": "timeout", "skipped": False}
    except Exception as e2:  # noqa: BLE001
        return {"check": "tests", "ok": False, "detail": str(e2), "skipped": False}


def try_import_module(workspace: Path, module_file: str = "todo.py") -> Dict[str, Any]:
    """Import a primary module from the workspace for a smoke signal."""
    path = Path(workspace) / module_file
    if not path.exists():
        return {"check": "import", "ok": False, "detail": f"missing {module_file}"}
    try:
        # Unique name avoids colliding with previously imported modules / NoneType quirks
        mod_name = f"_hoh_candidate_{abs(hash(str(path.resolve()))) % (10**8)}"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            return {"check": "import", "ok": False, "detail": "spec failed"}
        mod = importlib.util.module_from_spec(spec)
        # Ensure module is registered before exec (some patterns need this)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        attrs = [a for a in dir(mod) if not a.startswith("_")][:30]
        # cleanup
        sys.modules.pop(mod_name, None)
        return {"check": "import", "ok": True, "attrs": attrs}
    except Exception as e:  # noqa: BLE001
        return {"check": "import", "ok": False, "detail": repr(e)}


def runtime_check(
    workspace: Path,
    *,
    tests_dir: Optional[Path] = None,
    primary_module: str = "todo.py",
) -> Dict[str, Any]:
    """Aggregate deterministic checks used as Runtime.check(A_t) for QA."""
    workspace = Path(workspace)
    compile_r = compile_python_tree(workspace)
    import_r = try_import_module(workspace, primary_module)
    test_r = run_pytest(workspace, tests_dir)
    ok = bool(compile_r.get("ok")) and bool(import_r.get("ok")) and bool(test_r.get("ok"))
    return {
        "ok": ok,
        "compile": compile_r,
        "import": import_r,
        "tests": test_r,
    }
