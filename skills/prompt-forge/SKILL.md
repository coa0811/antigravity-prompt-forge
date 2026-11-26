---
name: prompt-forge
description: >-
  Use when the user asks in natural language to optimize, refine, expand, or structure
  a prompt/instruction before execution (including recommending /swarm, /teamwork,
  /run-workflow, /boost, or direct execution), or to inspect/manage Prompt Forge
  personal preferences and reusable templates.
---

# Prompt Forge

Ground brief instructions into clear, executable prompts via verified workspace context, targeted clarifications, scenario templates, and cross-project preferences.

## Portable CLI Invocation

Always invoke via the global plugin symlink (NEVER `./scripts/...`—`run_command` runs in workspace CWD):

```bash
~/.gemini/config/plugins/prompt-forge/scripts/run_python.sh \
  ~/.gemini/config/plugins/prompt-forge/scripts/prompt_forge_cli.py <subcommand>
```

Pipe multi-line prompts or special characters via `--prompt-file -`.

## Workflow 1: Optimize a Prompt (Natural-Language Trigger)

1. Identify explicit mode (`confirm`, `only`, `execute`), scenario (`auto` or template ID), and optional tool-routing override (`auto` or `off`). **Omit `--mode` and `--tool-routing` unless explicitly requested** so persisted preferences apply.
2. Build the packet via CLI. To prevent embedded `EOF` or metacharacters from ending the heredoc early, pass `--prompt-file <temp_file>` or a single-quoted delimiter verified absent from `<raw prompt>` (e.g. `__PROMPT_FORGE_INPUT_9F3A__`):
   ```bash
   # Append --mode <mode> or --tool-routing <auto|off> only when explicitly overriding prefs
   ~/.gemini/config/plugins/prompt-forge/scripts/run_python.sh \
     ~/.gemini/config/plugins/prompt-forge/scripts/prompt_forge_cli.py packet \
     --workspace "<active_workspace_or_cwd>" \
     --scenario "<auto|template_id>" \
     --prompt-file - <<'__PROMPT_FORGE_INPUT_9F3A__'
   <raw prompt>
   __PROMPT_FORGE_INPUT_9F3A__
   ```
3. Obey the returned `MODE` and rubric:
   - **Ground in facts**: Cite ONLY paths, git status, commands, and rules in `<WORKSPACE_SNAPSHOT>` or verified via read-only tools; NEVER fabricate paths, scripts, or constraints.
   - **Execution tool routing** (ONLY when `tool_routing=auto`; skip when `tool_routing=off`): Verify the heuristic hint against workspace context and pick ONE final choice—defaulting to `none` (direct single-session) per YAGNI:
     - `none` (in-session): Single-file/localized edits, straightforward fixes, docs, or Q&A. Keep fenced prompt pure (no `/` prefix).
     - `/run-workflow` (in-session via `run_workflow` tool/skill): Deterministic Python fan-out/pipeline over `>=5` independent items returning structured schemas.
     - `/boost` (host slash cmd): High-difficulty single-task deep work (race conditions, deadlocks, memory leaks, subtle root-cause debugging, complex/security-critical algorithms).
     - `/swarm` (host slash cmd): Multi-worker parallel modules requiring shared-board coordination or independent verifier cross-checks (quota-heavy; avoid same-file edits).
     - `/teamwork` (host slash cmd): End-to-end `0->1` project/system delivery handed over to an autonomous team.
   - **Clarify unknowns**: If critical decisions are ambiguous (scope, destructive trade-offs, incompatible options), call `ask_question` (1–3 MCQs) BEFORE finalizing; skip if clear.
   - **Present prompt**:
     - When `tool_routing=auto`: if final tool choice is `/run-workflow`, `/boost`, `/swarm`, or `/teamwork`, start line 1 inside the fenced Markdown block with `/<cmd> <concise imperative headline>` (no duplicate `/<cmd>`); if `none`, keep it pure with no leading `/`. Include `执行工具建议：<tool 或 不选（常规直跑）> — <一句话选型理由与权衡>` outside the block.
     - When `tool_routing=off`: emit a pure fenced Markdown prompt (no `/` prefix) and omit `执行工具建议`.
     - Always output a 3–5 bullet summary of added context/assumptions outside the block, use direct imperative voice, scale depth to complexity without filler, preserve literal constraints verbatim, and label unconfirmed defaults as revocable assumptions.
   - **Mode boundaries**:
     - `only`: Stop after presenting prompt and summary; NEVER edit files or run mutating actions.
     - `confirm`: NEVER edit files or run mutating actions yet.
       - When `tool_routing=off` (or `tool_routing=auto` with final choice `none`): `(Recommended) 直接按优化后的指令执行` / `仅保留优化稿（不执行，方便复制使用）` / `按补充意见微调优化稿（可在下方自定义输入栏直接填写修改点）`.
       - When `tool_routing=auto` with `/run-workflow`: `(Recommended) 直接在当前会话调用 run_workflow 按优化稿执行` / `仅保留带 /run-workflow 前缀的优化稿（方便复制手动发送）` / `不启用 /run-workflow，改为在当前会话常规直跑` / `按补充意见微调优化稿（可在下方自定义输入栏直接填写修改点）`.
       - When `tool_routing=auto` with `/boost` | `/swarm` | `/teamwork` (host slash commands cannot be self-started mid-turn): `(Recommended) 保留带 /<cmd> 前缀的优化稿并结束本轮（复制发送即可激活 /<cmd>）` / `不启用 /<cmd>，直接在当前会话按优化稿常规执行` / `按补充意见微调优化稿（可在下方自定义输入栏直接填写修改点）`.
       Proceed only after confirmation (load `run-workflow` skill if executing `/run-workflow`; NEVER call `swarm` tool unprompted).
     - `execute`: Proceed directly using the enriched specification (when `tool_routing=off` or `none`, run directly; when `auto` + `/run-workflow`, load `run-workflow` skill and invoke `run_workflow`; when `auto` + host slash tool, note in 1 line that host `/` input is required for future runs and execute directly in-session without calling `swarm` unprompted).

## Workflow 2: Manage Personal Preferences & Custom Templates

Data dir: `~/.gemini/antigravity/plugin_data/prompt-forge/`

- **Preferences**:
  - `prefs show` / `prefs reset`
  - `prefs set [--language <lang>] [--default-mode <confirm|only|execute>] [--tool-routing <auto|off>]`
  - `prefs add-constraint "<text>"` / `prefs remove-constraint "<index_or_text>"`
  - `prefs add-style "<text>"` / `prefs remove-style "<index_or_text>"`
  - `prefs add-avoid "<text>"` / `prefs remove-avoid "<index_or_text>"`
  - `prefs set-glossary "<term>" "<definition>"` / `prefs remove-glossary "<term>"`
- **Templates**:
  - `template list` / `template show <id>`
  - `template save <id> --name "<display_name>" --description "<desc>" --keywords "<k1,k2>" --sections "<s1|s2|s3>" --clarifying "<c1|c2>" --verify "<v1|v2>"` (or `--json '<json>'`)
  - `template delete <id>` (reverts to built-in if shadowing a default)

Custom template IDs must match `^[a-z0-9][a-z0-9_-]{0,31}$` and avoid reserved words (`only`, `copy`, `draft`, `go`, `run`, `prefs`, `auto`).
