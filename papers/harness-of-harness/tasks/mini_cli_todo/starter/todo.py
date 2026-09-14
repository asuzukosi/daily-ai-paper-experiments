"""Starter todo module — intentionally incomplete (A0).

HoH loops should grow this into a working TodoStore + CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Todo:
    id: int
    title: str
    done: bool = False


@dataclass
class TodoStore:
    """Minimal stub — add / list / complete / remove / persistence missing."""

    items: List[Todo] = field(default_factory=list)
    _next_id: int = 1
    path: str = "todos.json"

    def add(self, title: str) -> Todo:
        raise NotImplementedError("add not implemented yet")

    def list(self) -> List[Todo]:
        return list(self.items)

    def complete(self, todo_id: int) -> Todo:
        raise NotImplementedError("complete not implemented yet")

    def remove(self, todo_id: int) -> None:
        raise NotImplementedError("remove not implemented yet")

    def save(self) -> None:
        raise NotImplementedError("save not implemented yet")

    def load(self) -> None:
        raise NotImplementedError("load not implemented yet")


def main(argv: Optional[List[str]] = None) -> int:
    raise NotImplementedError("CLI not implemented yet")


if __name__ == "__main__":
    raise SystemExit(main())
