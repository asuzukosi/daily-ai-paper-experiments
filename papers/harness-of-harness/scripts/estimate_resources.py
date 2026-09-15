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
                "notes": f"T={cfg.get('T')} max_new_tokens={cfg.get('max_new_tokens')}",
            }
        )

    print("Harness-of-Harness — estimated resources (from configs)")
    print("=" * 100)
    header = (
        f"{'Experiment':<36} {'Model':<36} {'GPU':<40} {'VRAM':<8} {'Disk':<8}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['experiment']:<36} {r['model']:<36} {str(r['gpu']):<40} "
            f"{r['vram']:<8} {r['disk']:<8}"
        )
        print(f"  wall: {r['wall']}")
        print(f"  notes: {r['notes']}")
    print("=" * 100)
    print("See RESOURCES.md for the curated table + CPU demo row.")
    print(f"Configs dir: {_CONFIGS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
