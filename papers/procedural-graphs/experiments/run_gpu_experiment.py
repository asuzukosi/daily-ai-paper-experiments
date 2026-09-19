#!/usr/bin/env python3
"""GPU experiment driver for Procedural Graphs mini_shop evolution.

Loads Qwen2.5-7B-Instruct (or config model_id) via transformers and runs
the self-evolution loop for K iterations.

HARD GATE: if CUDA is unavailable, exit with a clear RunPod message.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from procedural_graphs.loop import EvolutionConfig, EvolutionLoop, load_jsonl_tasks  # noqa: E402
from procedural_graphs.runtime import TransformersBackend  # noqa: E402


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore

        return dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    except ImportError:
        pass
    data: Dict[str, Any] = {}
    resources: Dict[str, Any] = {}
    in_resources = False

    def _strip_comment(s: str) -> str:
        out, in_q, qch = [], False, ""
        for ch in s:
            if ch in ("\"", "'") and not in_q:
                in_q, qch = True, ch
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
        if not raw.strip():
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
            if isinstance(val, str) and val.lower() in ("true", "false"):
                val = val.lower() == "true"
            elif isinstance(val, str) and val.replace(".", "", 1).isdigit():
                val = float(val) if "." in val else int(val)
            data[k.strip()] = val
    if resources:
        data["resources"] = resources
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    model_id = str(cfg.get("model_id", "Qwen/Qwen2.5-7B-Instruct"))
    print(f"[procedural-graphs] config={args.config.name} model={model_id}")

    try:
        import torch
    except ImportError:
        print("ERROR: torch not installed. On RunPod: bash runpod/01_setup_env.sh", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print(
            "ERROR: CUDA unavailable. This GPU experiment hard-gates without a GPU.\n"
            "Use RunPod (see runpod/) or run the CPU demo: bash experiments/run_cpu_demo.sh",
            file=sys.stderr,
        )
        return 2

    skip_load = bool(cfg.get("smoke_skip_model_load")) or os.environ.get("PG_SMOKE_SKIP_LOAD") == "1"
    if args.smoke and skip_load:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        print(f"SMOKE OK: tokenizer loaded ({tok.__class__.__name__}), CUDA devices={torch.cuda.device_count()}")
        return 0

    backend = TransformersBackend(
        model_id=model_id,
        max_new_tokens=int(cfg.get("max_new_tokens", 256)),
        temperature=float(cfg.get("temperature", 0.0)),
        torch_dtype=str(cfg.get("torch_dtype", "bfloat16")),
    )
    if args.smoke:
        # Load model once and do a tiny generate
        out = backend.generate(
            system="You are a helpful assistant.",
            user="Reply with exactly: ACTION: Search(usb-c hub)",
            role="solver",
        )
        print("SMOKE generate:", out[:200].replace("\n", " "))
        print("SMOKE OK")
        return 0

    task_dir = _ROOT / str(cfg.get("task_dir", "tasks/mini_shop"))
    out = _ROOT / "artifacts" / str(cfg.get("artifacts_subdir", "gpu_run"))
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    loop = EvolutionLoop(
        llm=backend,
        train=load_jsonl_tasks(task_dir / "train.jsonl"),
        val=load_jsonl_tasks(task_dir / "val.jsonl"),
        test=load_jsonl_tasks(task_dir / "test.jsonl"),
        workspace=out,
        config=EvolutionConfig(
            K=int(cfg.get("K", 3)),
            hop=int(cfg.get("hop", 2)),
            history_window=int(cfg.get("history_window", 3)),
            init=str(cfg.get("init", "skeleton")),
            eval_modes=True,
        ),
    )
    summary = loop.run()
    print(json.dumps(summary["mode_scores"], indent=2))
    print(f"Wrote {out / 'run_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
