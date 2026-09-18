"""Mini shopping / multi-hop tool-use environment for Procedural Graph demos."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


CATALOG = {
    "usb-c-hub": {
        "name": "USB-C Hub 7-in-1",
        "price": 42,
        "stock": 8,
        "tags": ["usb-c", "hub", "laptop"],
        "sku": "SKU-HUB-7",
    },
    "mechanical-keyboard": {
        "name": "Mechanical Keyboard RGB",
        "price": 89,
        "stock": 3,
        "tags": ["keyboard", "mechanical", "rgb"],
        "sku": "SKU-KB-RGB",
    },
    "wireless-mouse": {
        "name": "Wireless Mouse Quiet",
        "price": 29,
        "stock": 15,
        "tags": ["mouse", "wireless", "quiet"],
        "sku": "SKU-MS-W",
    },
    "laptop-stand": {
        "name": "Aluminum Laptop Stand",
        "price": 35,
        "stock": 0,
        "tags": ["stand", "laptop", "aluminum"],
        "sku": "SKU-ST-AL",
    },
    "webcam-1080p": {
        "name": "Webcam 1080p",
        "price": 55,
        "stock": 6,
        "tags": ["webcam", "1080p", "camera"],
        "sku": "SKU-WC-1080",
    },
    "noise-cancelling-headphones": {
        "name": "Noise Cancelling Headphones",
        "price": 120,
        "stock": 4,
        "tags": ["headphones", "noise-cancelling", "audio"],
        "sku": "SKU-HP-NC",
    },
}


@dataclass
class Task:
    id: str
    query: str
    gold: str  # expected SKU or short answer
    budget: Optional[int] = None
    require_in_stock: bool = True
    tags: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "gold": self.gold,
            "budget": self.budget,
            "require_in_stock": self.require_in_stock,
            "tags": self.tags or [],
        }


TOOLS = {
    "Search": "Search(query) — search catalog by keywords",
    "CheckInventory": "CheckInventory(item_key) — stock and price for a catalog key",
    "Compare": "Compare(item_a, item_b) — compare two catalog keys on price/stock",
    "Verify": "Verify(item_key) — verify item matches query constraints / budget",
    "Answer": "Answer(sku_or_text) — emit final answer and finish",
}


class ShopEnv:
    """Deterministic tool environment. Observations are textual."""

    def __init__(self, task: Task):
        self.task = task
        self.done = False
        self.last_search: List[str] = []
        self.verified: Optional[str] = False  # type: ignore
        self.checked: set = set()
        self.steps = 0
        self.max_steps = 12

    def reset(self) -> str:
        self.done = False
        self.last_search = []
        self.verified = None
        self.checked = set()
        self.steps = 0
        return (
            f"ShopEnv ready. Query: {self.task.query}\n"
            f"Budget: {self.task.budget if self.task.budget is not None else 'none'}\n"
            f"Tools: {', '.join(TOOLS)}"
        )

    def step(self, action: str) -> Tuple[str, bool, Dict[str, Any]]:
        self.steps += 1
        if self.steps > self.max_steps:
            self.done = True
            return "Max steps exceeded.", True, {"answer": "", "success": False}

        act = action.strip()
        name, args = _parse_action(act)

        if name == "Search":
            q = (args[0] if args else self.task.query).lower()
            hits = []
            for key, item in CATALOG.items():
                blob = " ".join([key, item["name"], " ".join(item["tags"])]).lower()
                if any(tok in blob for tok in re.findall(r"[a-z0-9\-]+", q) if len(tok) > 2):
                    hits.append(key)
            if not hits:
                # soft fallback: tag overlap with task tags
                for key, item in CATALOG.items():
                    if self.task.tags and any(t in item["tags"] for t in self.task.tags):
                        hits.append(key)
            self.last_search = hits[:4]
            return f"Search hits: {self.last_search}", False, {}

        if name == "CheckInventory":
            key = args[0] if args else (self.last_search[0] if self.last_search else "")
            key = key.strip().strip("'\"")
            if key not in CATALOG:
                return f"Unknown item_key={key}", False, {}
            item = CATALOG[key]
            self.checked.add(key)
            return (
                f"Inventory[{key}]: name={item['name']} price={item['price']} "
                f"stock={item['stock']} sku={item['sku']}",
                False,
                {},
            )

        if name == "Compare":
            if len(args) < 2:
                return "Compare needs two item keys.", False, {}
            a, b = args[0], args[1]
            if a not in CATALOG or b not in CATALOG:
                return "Compare: unknown key.", False, {}
            ia, ib = CATALOG[a], CATALOG[b]
            cheaper = a if ia["price"] <= ib["price"] else b
            return (
                f"Compare: {a}(${ia['price']}, stock={ia['stock']}) vs "
                f"{b}(${ib['price']}, stock={ib['stock']}); cheaper={cheaper}",
                False,
                {},
            )

        if name == "Verify":
            key = args[0] if args else ""
            key = key.strip().strip("'\"")
            ok, reason = self._verify(key)
            self.verified = key if ok else None
            return f"Verify[{key}]: {'PASS' if ok else 'FAIL'} — {reason}", False, {}

        if name == "Answer":
            ans = args[0] if args else ""
            ans = ans.strip().strip("'\"")
            # allow answering with item key → map to SKU
            if ans in CATALOG:
                ans = CATALOG[ans]["sku"]
            self.done = True
            success = self._score_answer(ans)
            return f"Answer submitted: {ans}", True, {"answer": ans, "success": success}

        return f"Unknown action '{act}'. Use one of: {list(TOOLS)}", False, {}

    def _verify(self, key: str) -> Tuple[bool, str]:
        if key not in CATALOG:
            return False, "unknown item"
        item = CATALOG[key]
        if self.task.require_in_stock and item["stock"] <= 0:
            return False, "out of stock"
        if self.task.budget is not None and item["price"] > self.task.budget:
            return False, f"over budget ({item['price']} > {self.task.budget})"
        if self.task.tags:
            if not any(t in item["tags"] for t in self.task.tags):
                return False, "tags mismatch"
        return True, "matches constraints"

    def _score_answer(self, ans: str) -> bool:
        gold = self.task.gold.strip()
        if ans == gold:
            return True
        # also accept item key that maps to gold SKU
        for key, item in CATALOG.items():
            if item["sku"] == gold and ans in (key, item["sku"], item["name"]):
                return True
        return False


def _parse_action(act: str) -> Tuple[str, List[str]]:
    act = act.strip()
    m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\((.*)\)\s*$", act, re.S)
    if not m:
        # bare tool name
        if act in TOOLS:
            return act, []
        return act.split()[0] if act else "", []
    name = m.group(1)
    raw = m.group(2).strip()
    if not raw:
        return name, []
    parts = [p.strip().strip("'\"") for p in re.split(r",(?![^\[]*\])", raw)]
    return name, parts


def load_tasks(path: Path) -> List[Task]:
    tasks: List[Task] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        tasks.append(
            Task(
                id=str(d["id"]),
                query=str(d["query"]),
                gold=str(d["gold"]),
                budget=d.get("budget"),
                require_in_stock=bool(d.get("require_in_stock", True)),
                tags=d.get("tags"),
            )
        )
    return tasks


def score_episode(answer: str, gold: str) -> float:
    return 1.0 if answer.strip() == gold.strip() else 0.0
