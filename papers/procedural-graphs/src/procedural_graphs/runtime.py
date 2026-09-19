"""LLM backends: MockLLM (CPU) and TransformersBackend (GPU)."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Protocol


class LLMBackend(Protocol):
    def generate(self, *, system: str, user: str, role: str = "solver") -> str: ...


class MockLLM:
    """Deterministic CPU LLM.

    Without guidance: often Search→Answer (skips verify) → fails budget/stock tasks.
    With guidance mentioning Verify/CheckInventory: follows proper procedure.
    Refiner returns scripted topology repairs over rounds.
    """

    def __init__(self) -> None:
        self.call_log: List[Dict[str, Any]] = []
        self._refiner_calls = 0
        self._solver_steps: Dict[str, int] = {}

    def generate(self, *, system: str, user: str, role: str = "solver") -> str:
        self.call_log.append({"role": role, "system_len": len(system), "user_len": len(user)})
        if role == "guidance":
            return self._guidance(user)
        if role == "refiner":
            return self._refiner(user)
        return self._solver(user)

    def _meta(self, user: str) -> Dict[str, str]:
        m = re.search(r"<!--TASK_META id=(.*?) gold=(.*?)-->", user)
        if m:
            return {"id": m.group(1).strip(), "gold": m.group(2).strip()}
        return {"id": "?", "gold": ""}

    def _guidance(self, user: str) -> str:
        # Pull active node and suggest next
        ut = "Start"
        m = re.search(r"Active node ut:\s*(\S+)", user)
        if m:
            ut = m.group(1)
        # Prefer paths that include CheckInventory / Verify when present in subgraph
        has_verify = "Verify" in user
        has_check = "CheckInventory" in user
        has_compare = "Compare" in user
        if ut in ("Start",):
            return (
                "Immediate goal: gather candidates.\n"
                "Preferred next: Search for the request.\n"
                "Avoid: answering before any tool use."
            )
        if ut == "Search":
            nxt = "CheckInventory" if has_check else ("Verify" if has_verify else "Answer")
            return (
                f"Immediate goal: inspect top hit.\n"
                f"Preferred next: {nxt} on a search hit.\n"
                "Avoid: repeating the same Search query; avoid Answer yet."
            )
        if ut == "CheckInventory":
            if has_compare:
                return (
                    "Immediate goal: rank feasible options.\n"
                    "Preferred next: Compare two in-budget candidates, then Verify.\n"
                    "Avoid: Answer without Verify."
                )
            return (
                "Immediate goal: confirm constraints.\n"
                "Preferred next: Verify(item_key) then Answer(sku).\n"
                "Avoid: skipping Verify on budget/stock tasks."
            )
        if ut in ("Compare", "Verify"):
            return (
                "Immediate goal: commit only if verified.\n"
                "Preferred next: Answer with the verified SKU.\n"
                "Avoid: inventing SKUs; avoid another Search loop."
            )
        if ut == "Answer":
            return "Episode complete; stop."
        return (
            "Stay on valid graph transitions. Prefer CheckInventory → Verify → Answer.\n"
            "Avoid premature Answer and Search loops."
        )

    def _solver(self, user: str) -> str:
        meta = self._meta(user)
        tid = meta["id"]
        gold = meta["gold"]
        self._solver_steps[tid] = self._solver_steps.get(tid, 0) + 1
        n = self._solver_steps[tid]
        guided = "situational guidance" in user.lower() or "preferred next" in user.lower()
        # Infer item key from gold SKU
        sku_to_key = {
            "SKU-HUB-7": "usb-c-hub",
            "SKU-KB-RGB": "mechanical-keyboard",
            "SKU-MS-W": "wireless-mouse",
            "SKU-ST-AL": "laptop-stand",
            "SKU-WC-1080": "webcam-1080p",
            "SKU-HP-NC": "noise-cancelling-headphones",
        }
        key = sku_to_key.get(gold, "usb-c-hub")
        alt = "wireless-mouse" if key != "wireless-mouse" else "usb-c-hub"

        # Parse last observation hints
        traj = user
        has_search_hits = "Search hits:" in traj
        has_inventory = "Inventory[" in traj
        has_verify_pass = "Verify[" in traj and "PASS" in traj
        has_verify_fail = "Verify[" in traj and "FAIL" in traj

        if not guided:
            # Unguided: Search then premature Answer (often wrong on OOS / budget)
            if n == 1:
                return f"Thinking... need info.\nACTION: Search({key})"
            # Often answer with wrong/first-ish without verify — use alt for fail-prone ids
            if tid.endswith(("3", "7", "b", "f")) or "budget" in user.lower():
                wrong = alt
                return f"I'll go with the first option.\nACTION: Answer({wrong})"
            return f"Done.\nACTION: Answer({key})"

        # Guided path
        if n == 1 or not has_search_hits:
            return f"Follow guidance: search first.\nACTION: Search({key})"
        if has_search_hits and not has_inventory:
            return f"Check stock/price.\nACTION: CheckInventory({key})"
        if has_inventory and not (has_verify_pass or has_verify_fail):
            # Optionally compare when guidance mentions it
            if "Compare" in user and n == 3:
                return f"Compare alternatives.\nACTION: Compare({key}, {alt})"
            return f"Verify constraints.\nACTION: Verify({key})"
        if has_verify_fail:
            # try alt
            return f"Verification failed; try alternate.\nACTION: CheckInventory({alt})"
        if has_verify_pass:
            return f"Verified; submit SKU.\nACTION: Answer({gold})"
        return f"Fallback verify.\nACTION: Verify({key})"

    def _refiner(self, user: str) -> str:
        self._refiner_calls += 1
        n = self._refiner_calls
        # Scripted evolution: strengthen Verify path; prune Answer-after-Search
        if n == 1:
            edits = [
                {
                    "op": "add_edge",
                    "payload": {
                        "source": "Search",
                        "relation": "REQUIRES",
                        "target": "CheckInventory",
                        "attrs": {
                            "condition": "candidates available",
                            "guidance": "Always check inventory before Answer.",
                            "pitfalls": "Do not Answer right after Search.",
                        },
                    },
                    "rationale": "Failures skip inventory checks.",
                },
                {
                    "op": "add_edge",
                    "payload": {
                        "source": "CheckInventory",
                        "relation": "LEADS_TO",
                        "target": "Verify",
                        "attrs": {
                            "condition": "item in catalog",
                            "guidance": "Verify budget and stock.",
                            "pitfalls": "Out-of-stock laptop-stand must not be answered.",
                        },
                    },
                    "rationale": "Successes verify before checkout.",
                },
                {
                    "op": "delete_edge",
                    "payload": {"source": "Search", "relation": "LEADS_TO", "target": "Answer"},
                    "rationale": "Remove premature Answer transition if present.",
                },
            ]
        elif n == 2:
            edits = [
                {
                    "op": "add_node",
                    "payload": {"id": "Compare", "kind": "reasoning", "description": "Compare candidates"},
                    "rationale": "Need explicit compare step for budget tasks.",
                },
                {
                    "op": "add_edge",
                    "payload": {
                        "source": "CheckInventory",
                        "relation": "LEADS_TO",
                        "target": "Compare",
                        "attrs": {
                            "condition": "budget tight or multiple hits",
                            "guidance": "Compare price/stock before Verify.",
                            "pitfalls": "Don't Compare identical keys.",
                        },
                    },
                    "rationale": "Budget failures need Compare.",
                },
                {
                    "op": "add_edge",
                    "payload": {
                        "source": "Compare",
                        "relation": "LEADS_TO",
                        "target": "Verify",
                        "attrs": {
                            "condition": "preferred candidate chosen",
                            "guidance": "Verify the cheaper in-stock option.",
                            "pitfalls": "Skipping Verify after Compare.",
                        },
                    },
                    "rationale": "Close Compare→Verify gap.",
                },
            ]
        else:
            edits = [
                {
                    "op": "add_edge",
                    "payload": {
                        "source": "Verify",
                        "relation": "LEADS_TO",
                        "target": "Answer",
                        "attrs": {
                            "condition": "Verify PASS",
                            "guidance": "Answer with observed SKU only.",
                            "pitfalls": "Never invent SKUs.",
                        },
                    },
                    "rationale": "Reinforce verify-to-answer.",
                }
            ]
        # Avoid repeating rejected ops if listed
        if "delete_edge" in user and "Search" in user and "Answer" in user and n > 1:
            # still ok — different payloads
            pass
        return json.dumps(edits, indent=2)


class TransformersBackend:
    """Qwen2.5-*-Instruct via transformers. Lazy-imports torch/transformers."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "GPU extras required: pip install -e '.[gpu]'. "
                f"Original error: {e}"
            ) from e
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA unavailable. Run on a GPU pod (see runpod/). "
                "For CPU, use MockLLM via demos/cpu_pg_demo.py."
            )
        dtype = getattr(torch, self.torch_dtype, torch.bfloat16)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map=self.device_map,
            trust_remote_code=True,
        )

    def generate(self, *, system: str, user: str, role: str = "solver") -> str:
        self._ensure_loaded()
        assert self._tokenizer is not None and self._model is not None
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        if self.temperature > 0:
            gen_kwargs["temperature"] = max(self.temperature, 1e-5)
        out = self._model.generate(**inputs, **gen_kwargs)
        gen = out[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(gen, skip_special_tokens=True)


def load_backend(name: str, **kwargs: Any) -> Any:
    name = (name or "mock").lower()
    if name in ("mock", "cpu", "mockllm"):
        return MockLLM()
    if name in ("hf", "transformers", "qwen", "gpu"):
        return TransformersBackend(**kwargs)
    raise ValueError(f"Unknown backend: {name}")
