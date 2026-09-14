"""Unit tests for mini_cli_todo — ProgramBench-style verifier."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

# Allow importing todo.py from workspace root (cwd) or sibling starter.
_HERE = Path(__file__).resolve().parent
_CANDIDATES = [
    Path.cwd(),
    _HERE.parent / "starter",
    _HERE.parent,
]
for c in _CANDIDATES:
    if (c / "todo.py").exists() and str(c) not in sys.path:
        sys.path.insert(0, str(c))

import todo  # noqa: E402


class TestTodoStore(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / "todos.json")
        self.store = todo.TodoStore(path=self.path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_add_and_list(self) -> None:
        t = self.store.add("buy milk")
        self.assertEqual(t.title, "buy milk")
        self.assertFalse(t.done)
        self.assertTrue(isinstance(t.id, int))
        items = self.store.list()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].id, t.id)

    def test_add_rejects_empty(self) -> None:
        with self.assertRaises(Exception):
            self.store.add("")
        with self.assertRaises(Exception):
            self.store.add("   ")

    def test_complete(self) -> None:
        t = self.store.add("task")
        done = self.store.complete(t.id)
        self.assertTrue(done.done)

    def test_complete_missing(self) -> None:
        with self.assertRaises(Exception):
            self.store.complete(999)

    def test_remove(self) -> None:
        t = self.store.add("gone")
        self.store.remove(t.id)
        self.assertEqual(self.store.list(), [])

    def test_remove_missing(self) -> None:
        with self.assertRaises(Exception):
            self.store.remove(999)

    def test_persistence(self) -> None:
        t = self.store.add("persist me")
        self.store.save()
        self.assertTrue(Path(self.path).exists())
        data = json.loads(Path(self.path).read_text())
        self.assertTrue(isinstance(data, (list, dict)))
        other = todo.TodoStore(path=self.path)
        other.load()
        titles = [x.title for x in other.list()]
        self.assertIn("persist me", titles)

    def test_cli_list_empty(self) -> None:
        # main should exist and handle list without crashing when store empty
        rc = todo.main(["list"])
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
