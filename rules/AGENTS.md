# Prompt Forge Rules

- When a `Prompt Forge` `PreInvocation` packet is injected, obey `MODE`:
  - `confirm`: NEVER edit files or run mutating commands before presenting the enhanced prompt and confirming via `ask_question` (ask 1–3 `ask_question` clarifications first only when critical unknowns exist).
  - `only`: Output enhanced prompt + brief changelog, then stop; NEVER edit files or execute.
  - `execute`: State enriched spec/assumptions briefly, then execute directly.
  - `prefs`: Run `~/.gemini/config/plugins/prompt-forge/scripts/run_python.sh ~/.gemini/config/plugins/prompt-forge/scripts/prompt_forge_cli.py` for prefs/templates, report, and stop.
- For natural-language requests (without `++`) to optimize prompts or manage prefs/templates, follow the `prompt-forge` skill.
