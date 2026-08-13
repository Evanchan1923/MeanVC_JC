# Coding Style

- Follow the style already present in the files being edited.
- Keep changes focused and easy to review.
- Prefer clear names and simple structure over unnecessary abstraction.
- Remove obsolete paths directly instead of adding compatibility layers, fallbacks, or migrations.
- Build the smallest end-to-end version that fully meets the current requirements before adding more capability.
- Keep components modular and concerns clearly separated.
- Prefer established, well-maintained libraries when they reduce complexity or improve reliability.
- Lean on dependencies already in the project before writing custom implementations or adding packages.
- Make long-term architectural decisions. Avoid temporary stopgaps meant to be replaced later.
- Use concise comments only when they clarify non-obvious behavior.
- Long code lines are acceptable when they improve readability; use 250 characters as the soft maximum before wrapping.
