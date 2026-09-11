#!/usr/bin/env python3
"""Print a resource table from experiment YAML configs (no GPU required)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_CONFIGS = _ROOT / "experiments" / "configs"


def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        # Minimal fallback parser for our simple configs (no nested complexity needed).
        data: dict = {}
        resources: dict = {}
        in_resources = False
        for line in path.read_text().splitlines():
            raw = line
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
                data[k.strip()] = v.strip().strip('"').strip("'")
        if resources:
            data["resources"] = resources
        return data
    with path.open() as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    rows = []
    for path in sorted(_CONFIGS.glob("*.yaml")):
        cfg = load_yaml(path)
        res = cfg.get("resources") or {}
        rows.append(
            {
                "experiment": cfg.get("name", path.stem),
                "model": cfg.get("model_id", "?"),
                "gpu": res.get("gpu", "?"),
                "vram": f"≥{res.get('vram_gb', '?')}GB",
                "disk": f"≥{res.get('disk_gb', '?')}GB",
                "wall": res.get("wall_time", "?"),
                "notes": f"max_model_len={cfg.get('max_model_len')} n_chunks={cfg.get('n_chunks')}",
            }
        )

    # Also emit the canonical table from RESOURCES expectations.
    print("Declarative Attention — estimated resources (from configs)")
    print("=" * 100)
    header = (
        f"{'Experiment':<22} {'Model':<28} {'GPU':<36} {'VRAM':<8} {'Disk':<8} {'Est. wall time':<42}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['experiment']:<22} {r['model']:<28} {str(r['gpu']):<36} "
            f"{r['vram']:<8} {r['disk']:<8} {r['wall']:<42}"
        )
        print(f"  notes: {r['notes']}")
    print("=" * 100)
    print("See RESOURCES.md for the curated table + heavier alt (Qwen2.5-14B-Instruct).")
    print(f"Configs dir: {_CONFIGS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
