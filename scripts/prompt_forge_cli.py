#!/usr/bin/env python3
"""CLI helper for Prompt Forge skill-triggered packets, preferences, and templates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import prompt_forge_core as core  # noqa: E402


def _read_prompt_arg(prompt: str | None, prompt_file: str | None) -> str:
    if prompt_file is not None:
        if prompt_file == "-":
            return sys.stdin.read().strip()
        return Path(prompt_file).expanduser().read_text(encoding="utf-8").strip()
    return (prompt or "").strip()


def _split_pipe_or_comma(val: str | None, sep: str) -> list[str]:
    if not val:
        return []
    return [part.strip() for part in val.split(sep) if part.strip()]


def _cmd_packet(args: argparse.Namespace) -> int:
    raw_prompt = _read_prompt_arg(args.prompt, args.prompt_file)
    if not raw_prompt:
        print("Error: --prompt or --prompt-file must provide a non-empty prompt.", file=sys.stderr)
        return 2
    prefs = core.load_preferences()
    templates = core.load_all_templates()
    scenario = (args.scenario or "auto").strip().lower()
    if scenario != "auto" and scenario not in templates:
        print(f"Error: unknown scenario/template '{scenario}'.", file=sys.stderr)
        return 2
    mode = (args.mode or prefs["default_mode"]).strip().lower()
    if mode not in core.VALID_MODES:
        print(f"Error: invalid mode '{mode}'.", file=sys.stderr)
        return 2
    tool_routing = (args.tool_routing or prefs.get("tool_routing") or "auto").strip().lower()
    if tool_routing not in core.VALID_TOOL_ROUTING:
        print(f"Error: invalid tool_routing '{tool_routing}'.", file=sys.stderr)
        return 2

    match = core.TriggerMatch(
        mode=mode,
        scenario=scenario,
        scenario_explicit=(scenario != "auto"),
        raw_prompt=raw_prompt,
        prefix_token="skill",
    )
    ws_paths = args.workspace or [str(Path.cwd())]
    ctx = core.collect_workspace_context(ws_paths, raw_prompt)
    system_msg, user_badge = core.render_enhancement_packet(
        match,
        templates=templates,
        preferences=prefs,
        workspace_ctx=ctx,
        tool_routing_override=tool_routing,
    )
    if args.format == "json":
        tool_rec = core.recommend_execution_tool(raw_prompt, ctx) if tool_routing == "auto" else None
        print(
            json.dumps(
                {
                    "badge": user_badge,
                    "mode": mode,
                    "scenario": core.infer_scenario(raw_prompt, templates) if scenario == "auto" else scenario,
                    "workspace": ctx["workspace_path"],
                    "tool_routing": tool_routing,
                    "tool_recommendation": tool_rec,
                    "packet": system_msg,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(user_badge)
        print()
        print(system_msg)
    return 0


def _cmd_prefs(args: argparse.Namespace) -> int:
    action = args.prefs_action
    if action == "show":
        print(json.dumps(core.load_preferences(), ensure_ascii=False, indent=2))
        return 0
    if action == "set":
        if args.language is None and args.default_mode is None and args.tool_routing is None:
            print("Error: specify --language, --default-mode, and/or --tool-routing.", file=sys.stderr)
            return 2
        updated = core.update_preferences(
            language=args.language,
            default_mode=args.default_mode,
            tool_routing=args.tool_routing,
        )
        print(json.dumps(updated, ensure_ascii=False, indent=2))
        return 0
    if action == "add-constraint":
        print(json.dumps(core.add_pref_item("always_include_constraints", args.text), ensure_ascii=False, indent=2))
        return 0
    if action == "remove-constraint":
        prefs, removed = core.remove_pref_item("always_include_constraints", args.target)
        print(json.dumps({"removed": removed, "preferences": prefs}, ensure_ascii=False, indent=2))
        return 0 if removed else 1
    if action == "add-style":
        print(json.dumps(core.add_pref_item("style_notes", args.text), ensure_ascii=False, indent=2))
        return 0
    if action == "remove-style":
        prefs, removed = core.remove_pref_item("style_notes", args.target)
        print(json.dumps({"removed": removed, "preferences": prefs}, ensure_ascii=False, indent=2))
        return 0 if removed else 1
    if action == "add-avoid":
        print(json.dumps(core.add_pref_item("avoid", args.text), ensure_ascii=False, indent=2))
        return 0
    if action == "remove-avoid":
        prefs, removed = core.remove_pref_item("avoid", args.target)
        print(json.dumps({"removed": removed, "preferences": prefs}, ensure_ascii=False, indent=2))
        return 0 if removed else 1
    if action == "set-glossary":
        print(json.dumps(core.set_glossary(args.term, args.definition), ensure_ascii=False, indent=2))
        return 0
    if action == "remove-glossary":
        prefs, removed = core.remove_glossary(args.term)
        print(json.dumps({"removed": removed, "preferences": prefs}, ensure_ascii=False, indent=2))
        return 0 if removed else 1
    if action == "reset":
        print(json.dumps(core.reset_preferences(), ensure_ascii=False, indent=2))
        return 0
    return 2


def _cmd_template(args: argparse.Namespace) -> int:
    action = args.template_action
    if action == "list":
        templates = core.load_all_templates()
        summary = [
            {
                "id": tid,
                "name": tpl["name"],
                "custom": bool(tpl.get("custom")),
                "description": tpl.get("description", ""),
            }
            for tid, tpl in templates.items()
        ]
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if action == "show":
        templates = core.load_all_templates()
        tid = args.id.strip().lower()
        if tid not in templates:
            print(f"Error: template '{tid}' not found.", file=sys.stderr)
            return 1
        print(json.dumps(templates[tid], ensure_ascii=False, indent=2))
        return 0
    if action == "save":
        spec: dict[str, Any]
        if args.json:
            parsed = json.loads(args.json)
            if not isinstance(parsed, dict):
                raise ValueError("--json must decode to a JSON object.")
            spec = parsed
        else:
            spec = {
                "name": args.name or args.id,
                "description": args.description or "",
                "keywords": _split_pipe_or_comma(args.keywords, ","),
                "sections": _split_pipe_or_comma(args.sections, "|"),
                "clarifying_focus": _split_pipe_or_comma(args.clarifying, "|"),
                "verification_hints": _split_pipe_or_comma(args.verify, "|"),
            }
        saved = core.save_custom_template(args.id, spec)
        print(json.dumps(saved, ensure_ascii=False, indent=2))
        return 0
    if action == "delete":
        deleted = core.delete_custom_template(args.id)
        print(json.dumps({"id": args.id.strip().lower(), "deleted": deleted}, ensure_ascii=False, indent=2))
        return 0 if deleted else 1
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prompt Forge CLI")
    parser.add_argument("--data-dir", help="Override PLUGIN_DATA directory path")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pkt = sub.add_parser("packet", help="Assemble a workspace-grounded Prompt Forge enhancement packet")
    p_pkt.add_argument("--workspace", action="append", help="Workspace directory path (repeatable)")
    p_pkt.add_argument("--prompt", help="Raw prompt text")
    p_pkt.add_argument("--prompt-file", help="Path to file containing prompt, or '-' for stdin")
    p_pkt.add_argument("--mode", choices=list(core.VALID_MODES), help="Execution confirmation mode")
    p_pkt.add_argument("--scenario", default="auto", help="Scenario/template ID or 'auto'")
    p_pkt.add_argument("--tool-routing", choices=list(core.VALID_TOOL_ROUTING), help="Override execution tool routing mode")
    p_pkt.add_argument("--format", choices=["markdown", "json"], default="markdown")

    p_prefs = sub.add_parser("prefs", help="Manage persistent Prompt Forge personal preferences")
    prefs_sub = p_prefs.add_subparsers(dest="prefs_action", required=True)
    prefs_sub.add_parser("show")
    p_pset = prefs_sub.add_parser("set")
    p_pset.add_argument("--language")
    p_pset.add_argument("--default-mode", choices=list(core.VALID_MODES))
    p_pset.add_argument("--tool-routing", choices=list(core.VALID_TOOL_ROUTING))

    for act in ("add-constraint", "add-style", "add-avoid"):
        p_add = prefs_sub.add_parser(act)
        p_add.add_argument("text")
    for act in ("remove-constraint", "remove-style", "remove-avoid"):
        p_rem = prefs_sub.add_parser(act)
        p_rem.add_argument("target", help="1-based index or exact text")

    p_sg = prefs_sub.add_parser("set-glossary")
    p_sg.add_argument("term")
    p_sg.add_argument("definition")
    p_rg = prefs_sub.add_parser("remove-glossary")
    p_rg.add_argument("term")
    prefs_sub.add_parser("reset")

    p_tpl = sub.add_parser("template", help="Manage Prompt Forge scenario templates")
    tpl_sub = p_tpl.add_subparsers(dest="template_action", required=True)
    tpl_sub.add_parser("list")
    p_tshow = tpl_sub.add_parser("show")
    p_tshow.add_argument("id")
    p_tsave = tpl_sub.add_parser("save")
    p_tsave.add_argument("id")
    p_tsave.add_argument("--json", help="Full JSON object specification")
    p_tsave.add_argument("--name")
    p_tsave.add_argument("--description")
    p_tsave.add_argument("--keywords", help="Comma-separated keywords")
    p_tsave.add_argument("--sections", help="Pipe-separated section titles")
    p_tsave.add_argument("--clarifying", help="Pipe-separated clarifying questions/focus")
    p_tsave.add_argument("--verify", help="Pipe-separated verification hints")
    p_tdel = tpl_sub.add_parser("delete")
    p_tdel.add_argument("id")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.data_dir:
        core.set_data_dir(args.data_dir)
    try:
        if args.command == "packet":
            return _cmd_packet(args)
        if args.command == "prefs":
            return _cmd_prefs(args)
        if args.command == "template":
            return _cmd_template(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
