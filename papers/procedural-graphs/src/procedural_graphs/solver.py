"""ReAct-style solver Φ with optional PG guidance (§3.2 eq. 3)."""

from __future__ import annotations

import re
from typing import Any, List, Optional

from .env import ShopEnv, Task, score_episode
from .graph import ProceduralGraph
from .guidance import GuidanceEngine
from .prompts import SOLVER_SYSTEM, solver_user_prompt
from .schemas import EpisodeResult, TrajectoryStep


class Solver:
    def __init__(
        self,
        llm: Any,
        *,
        graph: Optional[ProceduralGraph] = None,
        use_guidance: bool = True,
        hop: int = 2,
        history_window: int = 3,
        max_steps: int = 10,
        mode: str = "guided",
    ):
        self.llm = llm
        self.graph = graph
        self.use_guidance = use_guidance and graph is not None
        self.guidance = GuidanceEngine(llm, hop=hop, history_window=history_window) if self.use_guidance else None
        self.max_steps = max_steps
        self.mode = mode
        self.history_window = history_window

    def run(self, task: Task) -> EpisodeResult:
        env = ShopEnv(task)
        obs0 = env.reset()
        steps: List[TrajectoryStep] = []
        last_action: Optional[str] = None
        answer = ""
        success = False

        for _ in range(self.max_steps):
            traj_text = self._format_traj(steps, window=self.history_window)
            guidance = ""
            active = ""
            if self.use_guidance and self.guidance and self.graph is not None:
                active, guidance, _ = self.guidance.generate(
                    self.graph, task.query, traj_text, last_action
                )

            user = solver_user_prompt(task.query, traj_text or obs0, guidance)
            # Embed meta for MockLLM
            user = f"<!--TASK_META id={task.id} gold={task.gold}-->\n" + user
            raw = self.llm.generate(system=SOLVER_SYSTEM, user=user, role="solver")
            action = self._extract_action(raw)
            thought = raw.strip()[:400]
            obs, done, info = env.step(action)
            steps.append(
                TrajectoryStep(
                    action=action,
                    observation=obs,
                    thought=thought,
                    guidance=guidance,
                    active_node=active,
                )
            )
            last_action = action
            if done:
                answer = str(info.get("answer", ""))
                success = bool(info.get("success", False))
                break

        if not answer and steps:
            # last ditch parse
            m = re.search(r"Answer\((.*?)\)", steps[-1].action)
            if m:
                answer = m.group(1).strip().strip("'\"")
        score = 1.0 if success else score_episode(answer, task.gold)
        if score >= 1.0:
            success = True
        return EpisodeResult(
            task_id=task.id,
            query=task.query,
            success=success,
            score=float(score),
            answer=answer,
            gold=task.gold,
            steps=steps,
            mode=self.mode,
        )

    @staticmethod
    def _format_traj(steps: List[TrajectoryStep], window: int = 3) -> str:
        recent = steps[-window:] if window > 0 else steps
        lines = []
        for i, s in enumerate(recent, 1):
            lines.append(f"{i}. ACTION {s.action} -> {s.observation}")
        return "\n".join(lines) if lines else "(empty)"

    @staticmethod
    def _extract_action(raw: str) -> str:
        m = re.search(r"ACTION:\s*(.+)", raw)
        if m:
            return m.group(1).strip().splitlines()[0].strip()
        # bare Tool(...)
        m = re.search(
            r"\b(Search|CheckInventory|Compare|Verify|Answer)\s*\([^\n]*\)",
            raw,
        )
        if m:
            return m.group(0).strip()
        for line in raw.splitlines():
            line = line.strip()
            if any(line.startswith(t) for t in ("Search", "CheckInventory", "Compare", "Verify", "Answer")):
                return line
        return "Search(query)"
