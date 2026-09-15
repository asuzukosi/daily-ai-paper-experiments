"""LLM backends: MockLLM (CPU) and TransformersBackend (GPU, lazy import)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Protocol


class LLMBackend(Protocol):
    def generate(self, *, system: str, user: str, role: str = "inference") -> str: ...


# Failure modes MockLLM exhibits WITHOUT skills, keyed by task id.
_FORMAT_FAIL_IDS = {
    "t03", "t07", "t11", "t15", "t19", "t23", "t27", "t31",
    "v03", "v07", "v11",
    "e03", "e07", "e11", "e15",
}
_ARITH_FAIL_IDS = {
    "t02", "t06", "t10", "t14", "t18", "t22", "t26", "t30",
    "v02", "v06", "v10",
    "e02", "e06", "e10", "e14",
}
_UNIT_FAIL_IDS = {
    "t04", "t08", "t12", "t16", "t20", "t24", "t28", "t32",
    "v04", "v08", "v12",
    "e04", "e08", "e12", "e16",
}


def _has_skill(system: str, *needles: str) -> bool:
    blob = system.lower()
    return any(n.lower() in blob for n in needles)


class MockLLM:
    """Deterministic CPU LLM that improves when the right skills are injected.

    Without skills: fails format / multi-step arithmetic / unit-strip cases.
    With skills (final-answer-format, multi-step-arithmetic, unit-normalization):
    those failure modes are fixed — gating sees real validation gains.
    """

    def __init__(self) -> None:
        self.call_log: List[Dict[str, Any]] = []
        self._maintainer_calls = 0
        self._proposer_calls = 0

    def generate(self, *, system: str, user: str, role: str = "inference") -> str:
        self.call_log.append({"role": role, "system_len": len(system), "user_len": len(user)})
        if role == "maintainer":
            return self._maintainer(user)
        if role == "proposer":
            return self._proposer(user)
        return self._inference(system, user)

    def _inference(self, system: str, user: str) -> str:
        gold = ""
        task_id = ""
        meta = re.search(r"<!--TASK_META id=(.*?) gold=(.*?)-->", user)
        if meta:
            task_id = meta.group(1).strip()
            gold = meta.group(2).strip()

        has_fmt = _has_skill(system, "final-answer-format", "final answer format")
        has_arith = _has_skill(system, "multi-step-arithmetic", "multi step arithmetic")
        has_unit = _has_skill(system, "unit-normalization", "normalize units", "strip units")

        if task_id in _FORMAT_FAIL_IDS and not has_fmt:
            wrong = self._perturb(gold, "format")
            return f"I think the result is roughly {wrong} but I'm not sure."
        if task_id in _ARITH_FAIL_IDS and not has_arith:
            wrong = self._perturb(gold, "arith")
            return f"Working through it... Final answer: {wrong}"
        if task_id in _UNIT_FAIL_IDS and not has_unit:
            wrong = self._perturb(gold, "unit")
            return f"Including units. Final answer: {wrong} kg"

        if gold:
            note = "Followed active skills. " if (has_fmt or has_arith or has_unit) else ""
            return f"{note}Computed result = {gold}.\nFinal answer: {gold}"

        q = user
        m = re.search(r"Problem:\n(.*?)(?:\n\nShow brief|$)", user, re.S)
        if m:
            q = m.group(1).strip()
        nums = re.findall(r"-?\d+", q)
        if len(nums) >= 2 and any(w in q.lower() for w in ("sum", "add", "total")):
            s = sum(int(n) for n in nums[:4])
            return f"Summing. Final answer: {s}"
        return "Cannot solve. Final answer: 0"

    @staticmethod
    def _perturb(gold: str, mode: str) -> str:
        try:
            v = float(gold.replace(",", ""))
            if mode == "arith":
                return str(int(v + 1) if v == int(v) else v + 1)
            if mode in ("unit", "format"):
                return str(int(v) if v == int(v) else v)
        except ValueError:
            pass
        return (gold + "1") if gold else "0"

    def _maintainer(self, user: str) -> str:
        self._maintainer_calls += 1
        n = self._maintainer_calls
        parts = [
            "## PATTERN: missing-final-answer-format\n"
            "Failure mode: model buries the numeric result in prose and omits "
            "`Final answer: <value>`.\n"
            "Workaround: require a dedicated final-answer line; reject free-form endings.\n"
            "Evidence: format-sensitive train/val items recur across iterations.\n",
            "## PATTERN: off-by-one-arithmetic\n"
            "Failure mode: multi-step word problems lose an intermediate quantity.\n"
            "Workaround: write each arithmetic step explicitly; re-check the last operation.\n"
            "Evidence: failing arithmetic traces on train split.\n",
        ]
        if n >= 2:
            parts.append(
                "## PATTERN: unit-contamination\n"
                "Failure mode: answers include units (kg, m, $) that break exact-match scoring.\n"
                "Workaround: normalize to bare numeric / canonical string before final line.\n"
            )
        parts.append(
            "## LOG\n"
            f"- Consolidated patterns for maintainer call #{n}\n"
            "- Root causes: format omission, arithmetic slip, unit contamination\n"
        )
        return "\n".join(parts)

    def _proposer(self, user: str) -> str:
        self._proposer_calls += 1
        n = self._proposer_calls
        if n == 1:
            return self._skill_final_answer()
        if n == 2:
            return self._skill_arithmetic()
        if n == 3:
            return self._skill_units()
        if "multi-step-arithmetic" in user.lower():
            return self._skill_arithmetic_update()
        return (
            "## ACTION: noop\n"
            "## SKILL_NAME: none\n"
            "## RATIONALE: Validation already strong; no new skill warranted.\n"
            "## PURPOSE\nNo change.\n"
            "## SKILL\nNo change.\n"
        )

    @staticmethod
    def _skill_final_answer() -> str:
        return (
            "## ACTION: create\n"
            "## SKILL_NAME: final-answer-format\n"
            "## RATIONALE: Wiki pattern missing-final-answer-format; FAIL traces omit required line.\n"
            "## PURPOSE\n"
            "Motivated by wiki/patterns/missing-final-answer-format.md.\n"
            "## SKILL\n"
            "# Final Answer Format\n\n"
            "Trigger: every math / reasoning problem.\n\n"
            "Rules:\n"
            "1. Show brief reasoning.\n"
            "2. End with exactly one line: `Final answer: <value>`.\n"
            "3. `<value>` must be the bare answer (no extra commentary on that line).\n"
            "4. Do not bury the answer only inside prose.\n"
        )

    @staticmethod
    def _skill_arithmetic() -> str:
        return (
            "## ACTION: create\n"
            "## SKILL_NAME: multi-step-arithmetic\n"
            "## RATIONALE: Wiki pattern off-by-one-arithmetic; train failures on multi-step items.\n"
            "## PURPOSE\n"
            "Motivated by wiki/patterns/off-by-one-arithmetic.md.\n"
            "## SKILL\n"
            "# Multi-Step Arithmetic\n\n"
            "Trigger: word problems with 2+ numeric operations.\n\n"
            "Rules:\n"
            "1. Extract all quantities with labels.\n"
            "2. Write each arithmetic step on its own line.\n"
            "3. Re-add / re-check the final combination before answering.\n"
            "4. Prefer exact integers when inputs are integers.\n"
        )

    @staticmethod
    def _skill_units() -> str:
        return (
            "## ACTION: create\n"
            "## SKILL_NAME: unit-normalization\n"
            "## RATIONALE: Wiki pattern unit-contamination; exact-match fails when units leak.\n"
            "## PURPOSE\n"
            "Motivated by wiki/patterns/unit-contamination.md.\n"
            "## SKILL\n"
            "# Unit Normalization\n\n"
            "Trigger: problems whose gold answers are unitless numbers or canonical strings.\n\n"
            "Rules:\n"
            "1. Compute with units internally if helpful.\n"
            "2. Strip units (kg, m, cm, $, %) from the Final answer line.\n"
            "3. Use the same numeric type as the expected answer (int vs float).\n"
        )

    @staticmethod
    def _skill_arithmetic_update() -> str:
        return (
            "## ACTION: update\n"
            "## SKILL_NAME: multi-step-arithmetic\n"
            "## RATIONALE: Refine after recurring off-by-one errors in later iterations.\n"
            "## PURPOSE\n"
            "Update linked to off-by-one-arithmetic.md + skill-impact history.\n"
            "## SKILL\n"
            "# Multi-Step Arithmetic (refined)\n\n"
            "Trigger: word problems with 2+ numeric operations.\n\n"
            "Rules:\n"
            "1. Extract all quantities with labels.\n"
            "2. Write each arithmetic step on its own line.\n"
            "3. After the last step, recompute independently once.\n"
            "4. Prefer exact integers when inputs are integers.\n"
            "5. Never drop an addend mentioned in the problem statement.\n"
        )


class TransformersBackend:
    """Qwen2.5-*-Instruct via transformers. Lazy-imports torch/transformers."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
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
                "For CPU, use MockLLM via demos/cpu_wikiskill_demo.py."
            )
        dtype = getattr(torch, self.torch_dtype, torch.bfloat16)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map=self.device_map,
            trust_remote_code=True,
        )

    def generate(self, *, system: str, user: str, role: str = "inference") -> str:
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
        out = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=max(self.temperature, 1e-5),
            pad_token_id=self._tokenizer.eos_token_id,
        )
        gen = out[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(gen, skip_special_tokens=True)


def load_backend(name: str, **kwargs: Any) -> Any:
    name = (name or "mock").lower()
    if name in ("mock", "cpu", "mockllm"):
        return MockLLM()
    if name in ("hf", "transformers", "qwen", "gpu"):
        return TransformersBackend(**kwargs)
    raise ValueError(f"Unknown backend: {name}")
