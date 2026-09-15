# Mini CLI Todo — ProgramBench-style task specification (S)

Build a small command-line todo application in Python that satisfies:

## Required capabilities

1. **Add** a todo item with a non-empty title; assign a unique integer id.
2. **List** todos; show id, title, and done status.
3. **Complete** a todo by id (mark done); error clearly if id missing.
4. **Remove** a todo by id; error clearly if id missing.
5. **Persistence**: save/load todos from a JSON file (`todos.json` by default).
6. **CLI**: expose a `main(argv)` entry point supporting subcommands:
   - `add <title...>`
   - `list`
   - `complete <id>`
   - `remove <id>`
7. **Library API**: module `todo.py` should expose a `TodoStore` class (or equivalent)
   with methods `add`, `list`, `complete`, `remove`, `save`, `load`.

## Non-goals

- No web UI, no multi-user auth, no cloud sync.
- Keep the increment small and verifiable each HoH loop.

## Acceptance (overall)

Unit tests under `tests/test_todo.py` must pass. Import of `todo.py` must succeed.
