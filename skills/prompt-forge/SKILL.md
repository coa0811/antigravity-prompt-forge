---
name: prompt-forge
description: >-
  Use when the user asks in natural language to optimize, refine, expand, or structure
  a prompt/instruction before execution (including recommending /swarm, /teamwork,
  /run-workflow, /boost, or direct execution), or to inspect/manage Prompt Forge
  personal preferences and reusable templates.
---

# Prompt Forge

Use the global symlink below for all subcommands (NEVER `./scripts/...`—`run_command` runs in workspace CWD).

## Workflow 1: Optimize a Prompt

1. Detect mode (`confirm`/`only`/`execute`), scenario (`auto`/template ID), and `--tool-routing <auto|off>`; **omit `--mode` and `--tool-routing` unless overriding** saved prefs.
2. Build packet via `--prompt-file <file>` or `--prompt-file -` with a single-quoted delimiter absent from `<raw prompt>` (e.g. `__PROMPT_FORGE_INPUT_9F3A__`) so embedded `EOF`/metacharacters cannot end heredoc early:
   ```bash
   ~/.gemini/config/plugins/prompt-forge/scripts/run_python.sh \
     ~/.gemini/config/plugins/prompt-forge/scripts/prompt_forge_cli.py packet \
     --workspace "<active_workspace_or_cwd>" --scenario "<auto|template_id>" \
     --prompt-file - <<'__PROMPT_FORGE_INPUT_9F3A__'
   <raw prompt>
   __PROMPT_FORGE_INPUT_9F3A__
   ```
   Diagnostic: `route --workspace "<active_workspace_or_cwd>" --prompt-file -` (or `--no-workspace`).
3. Obey returned `packet` rubric, `MODE`, and `EXECUTION TOOL ROUTING`:
   - Ground strictly in `<WORKSPACE_SNAPSHOT>` / read-only facts (NEVER fabricate paths/scripts/constraints); ask 1–3 `ask_question` MCQs before finalizing only if ambiguous.
   - Default tool: `none` (YAGNI); follow returned prefix and `执行工具建议` summary rules. `/run-workflow` runs via `run-workflow` skill/`run_workflow` tool; `/boost`, `/swarm`, `/teamwork` are host slash commands (cannot self-start mid-turn).
   - `only`: stop after prompt+summary (NEVER edit/mutate). `confirm`: NEVER edit/mutate before exact returned `ask_question` branch (`none`, `/run-workflow`, host slash). `execute`: run directly (note host `/` requirement for future host slash runs). NEVER call `swarm` tool unprompted.

## Workflow 2: Preferences & Templates

Data dir: `~/.gemini/antigravity/plugin_data/prompt-forge/`

- `prefs show` | `prefs reset` | `prefs set [--language <lang>] [--default-mode <confirm|only|execute>] [--tool-routing <auto|off>]`
- `prefs add-{constraint,style,avoid} "<text>"` | `prefs remove-{constraint,style,avoid} "<index_or_text>"`
- `prefs set-glossary "<term>" "<definition>"` | `prefs remove-glossary "<term>"`
- `template list` | `template show <id>` | `template delete <id>` (reverts to built-in if shadowing)
- `template save <id> --name "<name>" --description "<desc>" --keywords "<k1,k2>" --sections "<s1|s2|s3>" --clarifying "<c1|c2>" --verify "<v1|v2>"` (or `--json '<json>'`)

Custom IDs: `^[a-z0-9][a-z0-9_-]{0,31}$`; reserved: `only`, `copy`, `draft`, `go`, `run`, `prefs`, `auto`, `notool`, `direct`, `autotool`.
