---
name: prompt-forge
description: >-
  Use this skill when the user asks in natural language to optimize, refine, expand,
  or structure a prompt/instruction before execution, or asks to inspect, customize,
  or manage Prompt Forge personal preferences and reusable prompt templates.
---

# Prompt Forge

Prompt Forge turns brief or under-specified user instructions into clear, grounded,
executable prompts using verified workspace context, targeted clarifying questions,
scenario templates, and persistent cross-project preferences.

## Portable CLI Invocation

Always run the bundled CLI using the global plugin symlink path (do NOT use `./scripts/...`
because `run_command` executes in the user's workspace CWD, not the plugin directory):

```bash
~/.gemini/config/plugins/prompt-forge/scripts/run_python.sh \
  ~/.gemini/config/plugins/prompt-forge/scripts/prompt_forge_cli.py <subcommand>
```

For multi-line prompts or special shell characters, pipe via stdin with `--prompt-file -`.

## Workflow 1: Optimize a Prompt (Natural-Language Trigger)

1. Determine whether the user explicitly requested a mode (`confirm`, `only`, or `execute`)
   and scenario (`auto` or a built-in/custom template ID). **If the user did not specify a
   mode, omit `--mode`** so the CLI applies the persisted `default_mode` preference.
2. Assemble the enhancement packet using the CLI. To prevent embedded `EOF` lines or shell
   metacharacters in the raw prompt from terminating a heredoc early, either pass
   `--prompt-file <temp_file>` or choose a single-quoted unique delimiter verified NOT to
   appear in `<raw prompt>` (e.g. `__PROMPT_FORGE_INPUT_9F3A__`):
   ```bash
   # Omit --mode when the user did not explicitly override default_mode
   ~/.gemini/config/plugins/prompt-forge/scripts/run_python.sh \
     ~/.gemini/config/plugins/prompt-forge/scripts/prompt_forge_cli.py packet \
     --workspace "<active_workspace_or_cwd>" \
     --scenario "<auto|template_id>" \
     --prompt-file - <<'__PROMPT_FORGE_INPUT_9F3A__'
   <raw prompt>
   __PROMPT_FORGE_INPUT_9F3A__
   ```
3. Follow the `MODE` and rubric in the returned packet strictly:
   - **Ground in verified facts**: Only reference file paths, git status, commands, and
     rules observed in `<WORKSPACE_SNAPSHOT>` or verified via read-only tools. Never invent
     paths, scripts, or constraints.
   - **Clarify high-impact unknowns**: If critical decisions are genuinely ambiguous (target
     scope, destructive trade-offs, incompatible options), call `ask_question` with 1–3
     focused multiple-choice questions BEFORE finalizing the enhanced prompt. Skip if intent
     is already clear.
   - **Present the enhanced prompt**: Output a clean, copy-ready Markdown code block with
     the structured prompt matching the target scenario's sections and active preferences,
     followed by a concise 3–5 bullet summary of added context and assumptions. Write in
     direct imperative voice to the executing agent, scale detail proportionally to task
     complexity without filler, preserve literal constraints verbatim, and label unconfirmed
     defaults as revocable assumptions.
   - **Respect the mode boundary**:
     - `only`: Stop immediately after presenting the prompt and summary. Do NOT edit files
       or run mutating actions.
     - `confirm`: Do NOT edit files or run mutating actions yet. Call `ask_question` with:
       1. `(Recommended) 直接按优化后的指令执行`
       2. `仅保留优化稿（不执行，方便复制使用）`
       3. `按补充意见微调优化稿（可在下方自定义输入栏直接填写修改点）`
       Proceed only after the user confirms execution.
     - `execute`: Proceed directly using the enriched specification.

## Workflow 2: Manage Personal Preferences & Custom Templates

Inspect or update persistent cross-project settings stored in
`~/.gemini/antigravity/plugin_data/prompt-forge/`:

- **Preferences**:
  - Show current preferences: `prefs show`
  - Update language or default mode: `prefs set [--language <lang>] [--default-mode <confirm|only|execute>]`
  - Manage standing constraints: `prefs add-constraint "<text>"` / `prefs remove-constraint "<index_or_text>"`
  - Manage style notes: `prefs add-style "<text>"` / `prefs remove-style "<index_or_text>"`
  - Manage anti-patterns to avoid: `prefs add-avoid "<text>"` / `prefs remove-avoid "<index_or_text>"`
  - Manage domain glossary: `prefs set-glossary "<term>" "<definition>"` / `prefs remove-glossary "<term>"`
  - Reset all preferences: `prefs reset`
- **Custom Templates**:
  - List built-in & custom templates: `template list`
  - Inspect one template: `template show <id>`
  - Create or update a template:
    `template save <id> --name "<display_name>" --description "<desc>" --keywords "<k1,k2>" --sections "<s1|s2|s3>" --clarifying "<c1|c2>" --verify "<v1|v2>"`
    (or `--json '<json>'`)
  - Delete a custom template (reverts to built-in if ID shadowed a default): `template delete <id>`

Note: Custom template IDs must match `^[a-z0-9][a-z0-9_-]{0,31}$` and cannot use reserved
keywords (`only`, `copy`, `draft`, `go`, `run`, `prefs`, `auto`).
