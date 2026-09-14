#!/usr/bin/env python3
"""GPU experiment driver for Harness-of-Harness mini ProgramBench-style loops.

Loads Qwen2.5-Coder-7B-Instruct (or config model_id) via transformers and
runs Algorithm 1 for T=2..3 on tasks/mini_cli_todo.

HARD GATE: if CUDA is unavailable, exit with a clear RunPod message.
Never silently pretend a GPU run succeeded.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hoh.artifacts import ArtifactStore  # noqa: E402
from hoh.loop import HarnessOfHarness, LoopConfig  # noqa: E402


def _load_yaml_fallback(path: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    resources: Dict[str, Any] = {}
    in_resources = False

    def _strip_comment(s: str) -> str:
        if "#" not in s:
            return s
        out = []
        in_q = False
        qch = ""
        for ch in s:
            if ch in ("\"", "'") and not in_q:
                in_q = True
                qch = ch
                out.append(ch)
            elif in_q and ch == qch:
                in_q = False
                out.append(ch)
            elif ch == "#" and not in_q:
                break
            else:
                out.append(ch)
        return "".join(out).rstrip()

    for raw in path.read_text().splitlines():
        raw = _strip_comment(raw)
        if raw.strip().startswith("#") or not raw.strip():
            continue
        if raw.startswith("resources:"):
            in_resources = True
            continue
        if in_resources:
            if raw.startswith(" ") or raw.startswith("\t"):
                k, _, v = raw.strip().partition(":")
                resources[k.strip()] = v.strip().strip('"').strip("'")
                continue
            in_resources = False
        if ":" in raw and not raw.startswith(" "):
            k, _, v = raw.partition(":")
            val: Any = v.strip().strip('"').strip("'")
            if isinstance(val, str) and val.replace(".", "", 1).isdigit():
                val = float(val) if "." in val else int(val)
            data[k.strip()] = val
    if resources:
        data["resources"] = resources
    return data


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        return _load_yaml_fallback(path)
    with path.open() as f:
        return yaml.safe_load(f)


def _require_cuda() -> None:
    try:
        import torch
    except ImportError:
        print(
            "ERROR: PyTorch is not installed. On RunPod, run runpod/01_setup_env.sh first.\n"
            "This experiment requires a CUDA GPU. Use experiments/run_cpu_demo.sh for the "
            "MockLLM CPU loop on CPU.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if not torch.cuda.is_available():
        print(
            "ERROR: CUDA is not available on this machine.\n"
            "Harness-of-Harness GPU experiments need a GPU pod with Qwen2.5-Coder-7B-Instruct.\n"
            "  • Local CPU MockLLM demo:  bash experiments/run_cpu_demo.sh\n"
            "  • On RunPod:               bash runpod/01_setup_env.sh && bash runpod/03_run_smoke.sh\n"
            "Never silently pretending a GPU run succeeded.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _build_transformers_llm(cfg: Dict[str, Any]):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = cfg["model_id"]
    print(f"[hoh] Loading model: {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    dtype = torch.bfloat16 if str(cfg.get("dtype", "bfloat16")).lower() in (
        "bfloat16",
        "bf16",
    ) else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    max_new = int(cfg.get("max_new_tokens", 1024))
    temperature = float(cfg.get("temperature", 0.2))

    role_systems = {
        "planner": "You are the Project Planner. Output a DevelopmentPlan as JSON.",
        "developer": (
            "You are the Developer (sole writer). Emit full-file patches as "
            "```file path=REL\\n...``` blocks only."
        ),
        "qa": "You are the QA Tester. Output EvidenceReport JSON only.",
    }

    def llm(role: str, prompt: str) -> str:
        system = role_systems.get(role, "You are a helpful coding agent.")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        try:
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = system + "\n\n" + prompt
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                pad_token_id=tok.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[-1] :]
        return tok.decode(gen, skip_special_tokens=True)

    return llm


def main() -> int:
    ap = argparse.ArgumentParser(description="HoH GPU experiment driver")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Override artifacts output directory",
    )
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    _require_cuda()

    task_name = cfg.get("task", "mini_cli_todo")
    task_dir = _ROOT / "tasks" / task_name
    spec_path = task_dir / "SPEC.md"
    if not spec_path.exists():
        print(f"ERROR: missing SPEC at {spec_path}", file=sys.stderr)
        return 2
    spec = spec_path.read_text(encoding="utf-8")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.artifacts_dir or (
        _ROOT / "artifacts" / f"{cfg.get('name', 'gpu')}_{stamp}"
    )
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    store = ArtifactStore(out)
    store.seed_from(task_dir / "starter")
    tests_src = task_dir / "tests"
    tests_dst = store.workspace / "tests"
    if tests_src.exists():
        shutil.copytree(tests_src, tests_dst)

    t0 = time.time()
    llm = _build_transformers_llm(cfg)
    loop_cfg = LoopConfig(
        T=int(cfg.get("T", 2)),
        max_schema_retries=int(cfg.get("max_schema_retries", 2)),
        primary_module=str(cfg.get("primary_module", "todo.py")),
        tests_dir=tests_dst if tests_dst.exists() else None,
        snapshot_each=True,
    )
    hoh = HarnessOfHarness(spec=spec, store=store, llm=llm, config=loop_cfg)
    result = hoh.run()
    elapsed = time.time() - t0

    summary = result.metrics.summary()
    summary["elapsed_sec"] = elapsed
    summary["config"] = {k: cfg[k] for k in cfg if k != "resources"}
    summary["resources"] = cfg.get("resources")
    (out / "gpu_run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("=" * 68)
    print(f"HoH GPU run complete: {cfg.get('name')}")
    print(f"Model: {cfg.get('model_id')}  T={loop_cfg.T}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Final criteria pass rate: {summary['final_pass_rate']:.1f}%")
    print(f"Artifacts: {out}")
    for m in result.metrics.iterations:
        print(
            f"  t={m.iteration}: qa_passed={m.qa_passed} "
            f"runtime_ok={m.runtime_ok} criteria={m.n_criteria_pass}/{m.n_criteria}"
        )
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
