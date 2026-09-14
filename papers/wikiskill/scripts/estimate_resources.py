#!/usr/bin/env python3
"""Print resource estimates from experiment YAML configs (no GPU required)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = _ROOT / "experiments" / "configs"


def _parse(path: Path) -> dict:
    data, resources, in_res = {}, {}, False
    for raw in path.read_text().splitlines():
        if "#" in raw:
            raw = raw.split("#", 1)[0]
        if not raw.strip():
            continue
        if raw.startswith("resources:"):
            in_res = True
            continue
        if in_res:
            if raw.startswith(" ") or raw.startswith("\t"):
                k, _, v = raw.strip().partition(":")
                resources[k.strip()] = v.strip().strip('"').strip("'")
                continue
            in_res = False
        if ":" in raw and not raw.startswith(" "):
            k, _, v = raw.partition(":")
            data[k.strip()] = v.strip().strip('"').strip("'")
    data["resources"] = resources
    return data


def main() -> int:
    print("WikiSkill resource estimates (from configs)\n")
    print(f"{'Experiment':<28} {'Model':<32} {'GPU':<28} {'VRAM':<8} {'Wall'}")
    print("-" * 120)
    rows = [
        ("cpu_demo", "MockLLM", "none", "—", "<10 s"),
    ]
    for p in sorted(CFG_DIR.glob("*.yaml")):
        c = _parse(p)
        r = c.get("resources") or {}
        rows.append(
            (
                c.get("name", p.stem),
                c.get("model_id", "?"),
                r.get("gpu", "?"),
                r.get("vram_gb", "?"),
                r.get("wall_clock", "?"),
            )
        )
    for name, model, gpu, vram, wall in rows:
        print(f"{name:<28} {model:<32} {gpu:<28} {vram:<8} {wall}")
    print("\nPrimary: Qwen/Qwen2.5-7B-Instruct on ~24GB. Alt: Qwen2.5-14B-Instruct on A100 80GB.")
    print("Do not auto-create paid pods; use runpod/00_create_pod.example.sh only after editing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
