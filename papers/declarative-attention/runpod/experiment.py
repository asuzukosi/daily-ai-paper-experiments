#!/usr/bin/env python3
"""Declarative Attention — GPU experiment stub (RunPod).

This does NOT run a full DA eval. It documents the intended loop and hard-gates
any model load behind an explicit flag so automation cannot accidentally burn GPU.

Paper: arXiv:2609.02737
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="", help="HF model id (unused unless gate flag set)")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument(
        "--dry-config-check",
        action="store_true",
        help="Print the planned experiment config and exit 0 (safe / no GPU).",
    )
    p.add_argument(
        "--i-understand-this-costs-gpu",
        action="store_true",
        help="Required to attempt any model load. Still incomplete — fill in vLLM DA hooks.",
    )
    return p


def planned_config(args: argparse.Namespace) -> dict:
    return {
        "paper": "2609.02737",
        "modes": ["global", "focus", "local"],
        "chunking": "~2K-token magic chunks",
        "engine": "vLLM attention-metadata hooks (to implement)",
        "model": args.model or "<unset>",
        "max_new_tokens": args.max_new_tokens,
        "status": "scaffold-only",
    }


def main() -> int:
    args = build_parser().parse_args()
    cfg = planned_config(args)
    print("Declarative Attention experiment scaffold")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    if args.dry_config_check and not args.i_understand_this_costs_gpu:
        print("\nDry config check OK — no GPU work started.")
        return 0

    if not args.i_understand_this_costs_gpu:
        print(
            "\nRefusing to load a model. Re-run with --dry-config-check, "
            "or pass --i-understand-this-costs-gpu after implementing vLLM DA masking.",
            file=sys.stderr,
        )
        return 2

    print(
        "\nGate flag set, but the vLLM Declarative Attention integration "
        "is not implemented in this stub. Add block-aligned KV masking next.",
        file=sys.stderr,
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
