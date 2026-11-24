# Prompt Forge Execution Rules

- When a `Prompt Forge` `PreInvocation` packet is injected for the current turn, strictly obey its `MODE`:
  - `confirm`: NEVER edit files or run mutating commands before presenting the enhanced prompt and getting user confirmation via `ask_question`; ask 1–3 high-value clarifying questions via `ask_question` first only when critical unknowns exist.
  - `only`: Output only the enhanced prompt and a brief changelog, then stop without editing files or executing the task.
  - `execute`: Briefly state the enriched specification/assumptions and execute directly.
  - `prefs`: Use `~/.gemini/config/plugins/prompt-forge/scripts/run_python.sh ~/.gemini/config/plugins/prompt-forge/scripts/prompt_forge_cli.py` to inspect or update personal preferences / custom templates, then report and stop.
- When the user asks in natural language (without `++`) to optimize/enhance a prompt or manage Prompt Forge preferences/templates, load and follow the `prompt-forge` skill.
