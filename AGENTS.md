# AGENTS.md

## Codex Task Bookkeeping

- This section is mandatory for every task.
- For each task, update the relevant files inside `.codex/`.
- Follow `.codex/rules/general-rules.md` and `.codex/rules/coding-style.md`.
- Track pending work in `.codex/tasks/todo.md`.
- Move completed work to `.codex/tasks/done.md`.
- Record durable project context in `.codex/memory/project-context.md`.
- Record important decisions in `.codex/memory/decisions.md`.
- Record reusable lessons in `.codex/memory/lessons-learned.md`.
- Add user-visible changes to `.codex/changelog.md`.

## Coding Guidance

- Do not preserve backward compatibility. Remove obsolete paths instead of adding compatibility layers, fallbacks, or migrations.
- Choose the simplest implementation that fully meets the current requirements. Avoid speculative abstractions, configuration, and indirection.
- Grow the system in layers. Start from the smallest version that works end to end, and add each new capability on top of a product that already works. Never trade a working product for unfinished complexity.
- Keep components modular and concerns clearly separated.
- Prefer established, well-maintained libraries when they reduce overall complexity or improve reliability. Do not reimplement common functionality without a clear reason.
- Lean on the dependencies already in the project before writing your own implementation or adding packages. Do not assume a library lacks a capability without checking its documentation and types.
- Make architectural decisions for the long term. Do not accept a stopgap that only works for now and is meant to be replaced later.
