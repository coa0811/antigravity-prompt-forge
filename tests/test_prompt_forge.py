#!/usr/bin/env python3
"""Unit and end-to-end tests for the Prompt Forge Antigravity plugin."""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PLUGIN_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import prompt_forge_cli as cli  # noqa: E402
import prompt_forge_core as core  # noqa: E402
import prompt_forge_hook as hook  # noqa: E402


def _write_transcript(path: Path, steps: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for st in steps:
            fh.write(json.dumps(st, ensure_ascii=False) + "\n")


class PromptForgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.data_dir = self.root / "plugin_data"
        self.workspace = self.root / "workspace"
        self.workspace.mkdir(parents=True, exist_ok=True)
        core.set_data_dir(self.data_dir)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run_hook(self, payload: dict) -> dict:
        env = dict(os.environ)
        env["PLUGIN_DATA"] = str(self.data_dir)
        proc = subprocess.run(
            [str(SCRIPTS_DIR / "run_python.sh"), str(SCRIPTS_DIR / "prompt_forge_hook.py")],
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            text=True,
            env=env,
            cwd=str(PLUGIN_ROOT),
            timeout=5,
            check=True,
        )
        return json.loads(proc.stdout.strip())

    # 1. Trigger Parsing
    def test_trigger_parsing_variants_and_non_triggers(self) -> None:
        custom = {"bio-assay"}

        m = core.parse_trigger("++ refactor auth flow", custom_template_ids=custom)
        self.assertIsNotNone(m)
        self.assertEqual((m.mode, m.scenario, m.scenario_explicit, m.raw_prompt), ("confirm", "auto", False, "refactor auth flow"))

        m_fw = core.parse_trigger("  ＋＋   帮我写测试", custom_template_ids=custom)
        self.assertIsNotNone(m_fw)
        self.assertEqual((m_fw.mode, m_fw.scenario, m_fw.raw_prompt), ("confirm", "auto", "帮我写测试"))

        for kw, exp_mode in (("only", "only"), ("copy", "only"), ("draft", "only"), ("go", "execute"), ("run", "execute")):
            mk = core.parse_trigger(f"++{kw} build summary", custom_template_ids=custom)
            self.assertIsNotNone(mk, f"Expected match for ++{kw}")
            self.assertEqual((mk.mode, mk.scenario, mk.raw_prompt), (exp_mode, "auto", "build summary"))

        for sc in ("code", "plan", "research", "doc", "general"):
            ms = core.parse_trigger(f"++{sc} task body", custom_template_ids=custom)
            self.assertIsNotNone(ms, f"Expected match for ++{sc}")
            self.assertEqual((ms.mode, ms.scenario, ms.scenario_explicit, ms.raw_prompt), ("confirm", sc, True, "task body"))

        m_combo1 = core.parse_trigger("++only:code fix parser", custom_template_ids=custom)
        self.assertEqual((m_combo1.mode, m_combo1.scenario, m_combo1.raw_prompt), ("only", "code", "fix parser"))

        m_combo2 = core.parse_trigger("++code:only fix parser", custom_template_ids=custom)
        self.assertEqual((m_combo2.mode, m_combo2.scenario, m_combo2.raw_prompt), ("only", "code", "fix parser"))

        m_cust1 = core.parse_trigger("++:bio-assay design ELISA", custom_template_ids=custom)
        self.assertEqual((m_cust1.mode, m_cust1.scenario, m_cust1.raw_prompt), ("confirm", "bio-assay", "design ELISA"))

        m_cust2 = core.parse_trigger("++only:bio-assay design ELISA", custom_template_ids=custom)
        self.assertEqual((m_cust2.mode, m_cust2.scenario, m_cust2.raw_prompt), ("only", "bio-assay", "design ELISA"))

        m_cust3 = core.parse_trigger("++bio-assay design ELISA", custom_template_ids=custom)
        self.assertIsNotNone(m_cust3)
        self.assertEqual((m_cust3.mode, m_cust3.scenario, m_cust3.scenario_explicit, m_cust3.raw_prompt), ("confirm", "bio-assay", True, "design ELISA"))

        m_cust4 = core.parse_trigger("++bio-assay:go design ELISA", custom_template_ids=custom)
        self.assertIsNotNone(m_cust4)
        self.assertEqual((m_cust4.mode, m_cust4.scenario, m_cust4.scenario_explicit, m_cust4.raw_prompt), ("execute", "bio-assay", True, "design ELISA"))

        m_prefs_empty = core.parse_trigger("++prefs", custom_template_ids=custom)
        self.assertIsNotNone(m_prefs_empty)
        self.assertEqual((m_prefs_empty.mode, m_prefs_empty.raw_prompt), ("prefs", ""))

        m_prefs_text = core.parse_trigger("++prefs add constraint keep diffs small", custom_template_ids=custom)
        self.assertEqual((m_prefs_text.mode, m_prefs_text.raw_prompt), ("prefs", "add constraint keep diffs small"))

        for bad in (
            "",
            "++",
            "++   ",
            "++only",
            "++only:code   ",
            "hello ++ world",
            "C++ refactor",
            "++i",
            "++i;",
            "++counter += 1",
            "++unknown foo",
            "++:nonexistent foo",
            "++only:nonexistent foo",
            "++only:code:extra foo",
        ):
            self.assertIsNone(core.parse_trigger(bad, custom_template_ids=custom), f"Should NOT match: {bad!r}")

    # 2. Per-turn Deduplication
    def test_hook_fast_path_no_session_created(self) -> None:
        out = self._run_hook(
            {
                "conversationId": "conv-fast",
                "lastUserInput": "++i in C++ should not trigger",
                "workspacePaths": [str(self.workspace)],
            }
        )
        self.assertEqual(out, {})
        self.assertFalse(self.data_dir.exists())

    def test_hook_dedupe_across_invocations_and_exit_plan_mode(self) -> None:
        tr = self.root / "transcript.jsonl"
        prompt_text = "++code refactor worker"
        wrapped_prompt = (
            f"<USER_REQUEST>\n{prompt_text}\n</USER_REQUEST>\n"
            "<ADDITIONAL_METADATA>\nThe current local time is: 2026-11-23T21:39:10-06:00.\n</ADDITIONAL_METADATA>"
        )
        _write_transcript(
            tr,
            [
                {"step_index": 184, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": wrapped_prompt},
            ],
        )

        base_payload = {
            "conversationId": "conv-dedupe",
            "transcriptPath": str(tr),
            "workspacePaths": [str(self.workspace)],
            "lastUserInput": prompt_text,
            "invocationNum": 1,
            "initialNumSteps": 185,
        }
        out0 = self._run_hook(base_payload)
        self.assertIn("injectSteps", out0)
        msg_obj = out0["injectSteps"][0]["systemMessage"]
        self.assertNotIn("metadata", msg_obj)
        self.assertIn("MODE=confirm", msg_obj["systemMessage"])
        self.assertIn("Prompt Forge", msg_obj["systemMessage"])
        sess0 = json.loads((self.data_dir / "sessions" / "conv-dedupe.json").read_text(encoding="utf-8"))
        self.assertEqual(sess0["lastUserStepIndex"], 184)

        # Same turn, subsequent tool invocations where BOTH invocationNum AND initialNumSteps advance
        out1 = self._run_hook({**base_payload, "invocationNum": 2, "initialNumSteps": 187})
        self.assertEqual(out1, {})
        out2 = self._run_hook({**base_payload, "invocationNum": 3, "initialNumSteps": 189})
        self.assertEqual(out2, {})

        # exit_plan_mode handoff: invocationNum resets (3 -> 1), initialNumSteps advances (189 -> 193),
        # same cur_user_step, and transcript contains exit_plan_mode + SYSTEM_MESSAGE after cur_user_step.
        _write_transcript(
            tr,
            [
                {"step_index": 184, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": wrapped_prompt},
                {"step_index": 185, "type": "PLANNER_RESPONSE", "status": "DONE", "tool_calls": [{"name": "exit_plan_mode"}]},
                {"step_index": 186, "type": "GENERIC", "status": "DONE"},
                {"step_index": 187, "type": "SYSTEM_MESSAGE", "status": "DONE", "content": "Plan mode is off:"},
            ],
        )
        out_handoff = self._run_hook({**base_payload, "invocationNum": 1, "initialNumSteps": 193})
        self.assertEqual(out_handoff, {})

        # Subsequent read-only tool invocation after exit_plan_mode handoff (inv=2, initialNumSteps=195)
        # must still dedupe rather than re-injecting!
        _write_transcript(
            tr,
            [
                {"step_index": 184, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": wrapped_prompt},
                {"step_index": 185, "type": "PLANNER_RESPONSE", "status": "DONE", "tool_calls": [{"name": "exit_plan_mode"}]},
                {"step_index": 186, "type": "GENERIC", "status": "DONE"},
                {"step_index": 187, "type": "SYSTEM_MESSAGE", "status": "DONE", "content": "Plan mode is off:"},
                {"step_index": 188, "type": "PLANNER_RESPONSE", "status": "DONE", "tool_calls": [{"name": "read_file"}]},
                {"step_index": 189, "type": "GENERIC", "status": "DONE"},
            ],
        )
        out_post_handoff_tool = self._run_hook({**base_payload, "invocationNum": 2, "initialNumSteps": 195})
        self.assertEqual(out_post_handoff_tool, {})

        # Turn 1 tool call fails (GENERIC status=ERROR) -> Inv 2 in the SAME turn must still dedupe!
        only_prompt = "++only summarize repo"
        wrapped_only = f"<USER_REQUEST>\n{only_prompt}\n</USER_REQUEST>"
        _write_transcript(
            tr,
            [
                {"step_index": 10, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": wrapped_only},
            ],
        )
        t1_payload = {
            "conversationId": "conv-repeat-only",
            "transcriptPath": str(tr),
            "workspacePaths": [str(self.workspace)],
            "lastUserInput": only_prompt,
            "invocationNum": 1,
            "initialNumSteps": 11,
        }
        t1_out = self._run_hook(t1_payload)
        self.assertIn("MODE=only", t1_out["injectSteps"][0]["systemMessage"]["systemMessage"])

        _write_transcript(
            tr,
            [
                {"step_index": 10, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": wrapped_only},
                {"step_index": 11, "type": "PLANNER_RESPONSE", "status": "DONE", "tool_calls": [{"name": "read_file"}]},
                {"step_index": 12, "type": "GENERIC", "status": "ERROR"},
            ],
        )
        t1_err_cont = self._run_hook({**t1_payload, "invocationNum": 2, "initialNumSteps": 13})
        self.assertEqual(t1_err_cont, {})
        self.assertEqual(
            json.loads((self.data_dir / "sessions" / "conv-repeat-only.json").read_text(encoding="utf-8"))["lastUserStepIndex"],
            10,
        )

        # Turn 1 completes with terminal PLANNER_RESPONSE, then Turn 2 starts with IDENTICAL ++only prompt
        # while Turn 2's first TWO invocations still read Turn 1's finished transcript (stale step 10).
        _write_transcript(
            tr,
            [
                {"step_index": 10, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": wrapped_only},
                {"step_index": 11, "type": "PLANNER_RESPONSE", "status": "DONE", "tool_calls": []},
            ],
        )
        t2_out0 = self._run_hook(
            {**t1_payload, "invocationNum": 1, "initialNumSteps": 15}
        )
        self.assertIn("injectSteps", t2_out0)
        self.assertIn("MODE=only", t2_out0["injectSteps"][0]["systemMessage"]["systemMessage"])

        # Inv 2 still sees stale Turn 1 transcript (step 10) -> must dedupe AND NOT overwrite step to 10!
        t2_out_stale2 = self._run_hook(
            {**t1_payload, "invocationNum": 2, "initialNumSteps": 17}
        )
        self.assertEqual(t2_out_stale2, {})
        stale_sess = json.loads((self.data_dir / "sessions" / "conv-repeat-only.json").read_text(encoding="utf-8"))
        self.assertEqual(stale_sess["lastUserStepIndex"], -1)
        self.assertEqual(stale_sess["staleUserStepIndex"], 10)

        # Once Turn 2 transcript flushes (step 12), Inv 3 (with initialNumSteps=19) dedupes and backfills step 12.
        _write_transcript(
            tr,
            [
                {"step_index": 10, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": wrapped_only},
                {"step_index": 11, "type": "PLANNER_RESPONSE", "status": "DONE", "tool_calls": []},
                {"step_index": 12, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": wrapped_only},
            ],
        )
        t2_out1 = self._run_hook(
            {**t1_payload, "invocationNum": 3, "initialNumSteps": 19}
        )
        self.assertEqual(t2_out1, {})
        repeat_sess = json.loads((self.data_dir / "sessions" / "conv-repeat-only.json").read_text(encoding="utf-8"))
        self.assertEqual(repeat_sess["lastUserStepIndex"], 12)
        self.assertNotIn("staleUserStepIndex", repeat_sess)

        # User prompt containing literal <ADDITIONAL_METADATA>...</ADDITIONAL_METADATA> inside <USER_REQUEST>
        # must preserve inner tags when matching step_index (no false -1 -> no double-inject on handoff).
        tag_prompt = "++code test <ADDITIONAL_METADATA>x</ADDITIONAL_METADATA> parsing"
        wrapped_tag = (
            f"<USER_REQUEST>\n{tag_prompt}\n</USER_REQUEST>\n"
            "<ADDITIONAL_METADATA>\nThe current local time is: 2026-11-23T21:45:00-06:00.\n</ADDITIONAL_METADATA>"
        )
        _write_transcript(
            tr,
            [
                {"step_index": 50, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": wrapped_tag},
            ],
        )
        tag_payload = {
            "conversationId": "conv-xml-tags",
            "transcriptPath": str(tr),
            "workspacePaths": [str(self.workspace)],
            "lastUserInput": tag_prompt,
            "invocationNum": 1,
            "initialNumSteps": 51,
        }
        tag_out0 = self._run_hook(tag_payload)
        self.assertIn("injectSteps", tag_out0)
        self.assertEqual(
            json.loads((self.data_dir / "sessions" / "conv-xml-tags.json").read_text(encoding="utf-8"))["lastUserStepIndex"],
            50,
        )
        _write_transcript(
            tr,
            [
                {"step_index": 50, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": wrapped_tag},
                {"step_index": 51, "type": "PLANNER_RESPONSE", "status": "DONE", "tool_calls": [{"name": "exit_plan_mode"}]},
                {"step_index": 52, "type": "SYSTEM_MESSAGE", "status": "DONE", "content": "Plan mode is off:"},
            ],
        )
        tag_handoff = self._run_hook({**tag_payload, "invocationNum": 1, "initialNumSteps": 54})
        self.assertEqual(tag_handoff, {})

        # Short prompt intact while trailing host metadata is truncated (missing </ADDITIONAL_METADATA>)
        short_trunc_prompt = "++code fix parser"
        wrapped_short_trunc = (
            f"<USER_REQUEST>\n{short_trunc_prompt}\n</USER_REQUEST>\n"
            "<ADDITIONAL_METADATA>\nThe current local time is: 2026-11-23T21:48:00"
        )
        _write_transcript(
            tr,
            [
                {
                    "step_index": 50,
                    "type": "USER_INPUT",
                    "source": "USER_EXPLICIT",
                    "content": wrapped_short_trunc,
                    "truncated_fields": ["content"],
                },
            ],
        )
        short_trunc_payload = {
            "conversationId": "conv-xml-trunc-meta",
            "transcriptPath": str(tr),
            "workspacePaths": [str(self.workspace)],
            "lastUserInput": short_trunc_prompt,
            "invocationNum": 1,
            "initialNumSteps": 51,
        }
        st_out0 = self._run_hook(short_trunc_payload)
        self.assertIn("injectSteps", st_out0)
        self.assertEqual(
            json.loads((self.data_dir / "sessions" / "conv-xml-trunc-meta.json").read_text(encoding="utf-8"))["lastUserStepIndex"],
            50,
        )
        _write_transcript(
            tr,
            [
                {
                    "step_index": 50,
                    "type": "USER_INPUT",
                    "source": "USER_EXPLICIT",
                    "content": wrapped_short_trunc,
                    "truncated_fields": ["content"],
                },
                {"step_index": 51, "type": "PLANNER_RESPONSE", "status": "DONE", "tool_calls": [{"name": "exit_plan_mode"}]},
                {"step_index": 52, "type": "SYSTEM_MESSAGE", "status": "DONE", "content": "Plan mode is off:"},
            ],
        )
        st_handoff = self._run_hook({**short_trunc_payload, "invocationNum": 1, "initialNumSteps": 54})
        self.assertEqual(st_handoff, {})

        # Next turn expands the prompt ("++code fix parser and tests") while transcript still shows closed step 50:
        # must NOT prefix-match closed step 50 -> Inv 1 saves -1, Inv 2 dedupes & backfills step 55.
        expanded_prompt = "++code fix parser and tests"
        exp_payload = {
            "conversationId": "conv-xml-trunc-meta",
            "transcriptPath": str(tr),
            "workspacePaths": [str(self.workspace)],
            "lastUserInput": expanded_prompt,
            "invocationNum": 1,
            "initialNumSteps": 56,
        }
        exp_out0 = self._run_hook(exp_payload)
        self.assertIn("injectSteps", exp_out0)
        self.assertEqual(
            json.loads((self.data_dir / "sessions" / "conv-xml-trunc-meta.json").read_text(encoding="utf-8"))["lastUserStepIndex"],
            -1,
        )
        _write_transcript(
            tr,
            [
                {
                    "step_index": 50,
                    "type": "USER_INPUT",
                    "source": "USER_EXPLICIT",
                    "content": wrapped_short_trunc,
                    "truncated_fields": ["content"],
                },
                {"step_index": 51, "type": "PLANNER_RESPONSE", "status": "DONE", "tool_calls": [{"name": "exit_plan_mode"}]},
                {"step_index": 52, "type": "SYSTEM_MESSAGE", "status": "DONE", "content": "Plan mode is off:"},
                {"step_index": 54, "type": "PLANNER_RESPONSE", "status": "DONE", "tool_calls": []},
                {
                    "step_index": 55,
                    "type": "USER_INPUT",
                    "source": "USER_EXPLICIT",
                    "content": f"<USER_REQUEST>\n{expanded_prompt}\n</USER_REQUEST>",
                },
            ],
        )
        exp_out1 = self._run_hook({**exp_payload, "invocationNum": 2, "initialNumSteps": 58})
        self.assertEqual(exp_out1, {})
        self.assertEqual(
            json.loads((self.data_dir / "sessions" / "conv-xml-trunc-meta.json").read_text(encoding="utf-8"))["lastUserStepIndex"],
            55,
        )

        # 1) Turn 1 & Turn 2 Inv 1 both read an older unrelated message, then Turn 2 Inv 2 sees
        # Turn 1's finished step 10 for the first time while Turn 2 step 12 is still unflushed:
        # must inject only once on Inv 1, dedupe & mark step 10 stale on Inv 2, backfill 12 on Inv 3.
        _write_transcript(
            tr,
            [
                {"step_index": 8, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": "<USER_REQUEST>\nhello\n</USER_REQUEST>"},
            ],
        )
        lag_payload = {
            "conversationId": "conv-lag-stale",
            "transcriptPath": str(tr),
            "workspacePaths": [str(self.workspace)],
            "lastUserInput": only_prompt,
            "invocationNum": 1,
            "initialNumSteps": 13,
        }
        lag_out1 = self._run_hook(lag_payload)
        self.assertIn("injectSteps", lag_out1)
        _write_transcript(
            tr,
            [
                {"step_index": 8, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": "<USER_REQUEST>\nhello\n</USER_REQUEST>"},
                {"step_index": 10, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": wrapped_only},
                {"step_index": 11, "type": "PLANNER_RESPONSE", "status": "DONE", "tool_calls": []},
            ],
        )
        lag_out2 = self._run_hook({**lag_payload, "invocationNum": 2, "initialNumSteps": 15})
        self.assertEqual(lag_out2, {})
        lag_sess = json.loads((self.data_dir / "sessions" / "conv-lag-stale.json").read_text(encoding="utf-8"))
        self.assertEqual(lag_sess["lastUserStepIndex"], -1)
        self.assertEqual(lag_sess["staleUserStepIndex"], 10)
        _write_transcript(
            tr,
            [
                {"step_index": 8, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": "<USER_REQUEST>\nhello\n</USER_REQUEST>"},
                {"step_index": 10, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": wrapped_only},
                {"step_index": 11, "type": "PLANNER_RESPONSE", "status": "DONE", "tool_calls": []},
                {"step_index": 12, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": wrapped_only},
            ],
        )
        lag_out3 = self._run_hook({**lag_payload, "invocationNum": 3, "initialNumSteps": 17})
        self.assertEqual(lag_out3, {})
        self.assertEqual(
            json.loads((self.data_dir / "sessions" / "conv-lag-stale.json").read_text(encoding="utf-8"))["lastUserStepIndex"],
            12,
        )

        # 2) Truncated prompt > 100 chars sharing identical 120-char prefix must NOT match different suffix
        shared_prefix = "++code " + ("a" * 120) + " "
        old_long = shared_prefix + "old content extra"
        new_long = shared_prefix + "new content extra"
        _write_transcript(
            tr,
            [
                {
                    "step_index": 10,
                    "type": "USER_INPUT",
                    "source": "USER_EXPLICIT",
                    "content": f"<USER_REQUEST>\n{shared_prefix}old content",
                    "truncated_fields": ["content"],
                },
            ],
        )
        long_out0 = self._run_hook(
            {
                "conversationId": "conv-long-trunc",
                "transcriptPath": str(tr),
                "workspacePaths": [str(self.workspace)],
                "lastUserInput": new_long,
                "invocationNum": 1,
                "initialNumSteps": 13,
            }
        )
        self.assertIn("injectSteps", long_out0)
        self.assertEqual(
            json.loads((self.data_dir / "sessions" / "conv-long-trunc.json").read_text(encoding="utf-8"))["lastUserStepIndex"],
            -1,
        )
        _write_transcript(
            tr,
            [
                {"step_index": 10, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": f"<USER_REQUEST>\n{old_long}\n</USER_REQUEST>"},
                {"step_index": 12, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": f"<USER_REQUEST>\n{new_long}\n</USER_REQUEST>"},
            ],
        )
        long_out1 = self._run_hook(
            {
                "conversationId": "conv-long-trunc",
                "transcriptPath": str(tr),
                "workspacePaths": [str(self.workspace)],
                "lastUserInput": new_long,
                "invocationNum": 2,
                "initialNumSteps": 15,
            }
        )
        self.assertEqual(long_out1, {})
        self.assertEqual(
            json.loads((self.data_dir / "sessions" / "conv-long-trunc.json").read_text(encoding="utf-8"))["lastUserStepIndex"],
            12,
        )

        # 3) </USER_REQUEST> literal inside trailing <ADDITIONAL_METADATA> must still unwrap cleanly
        meta_lit_prompt = "++code unwrap metadata literal"
        wrapped_meta_lit = (
            f"<USER_REQUEST>\n{meta_lit_prompt}\n</USER_REQUEST>\n"
            "<ADDITIONAL_METADATA>\nExample tag: </USER_REQUEST>\n</ADDITIONAL_METADATA>"
        )
        _write_transcript(
            tr,
            [
                {"step_index": 60, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": wrapped_meta_lit},
            ],
        )
        ml_payload = {
            "conversationId": "conv-meta-literal",
            "transcriptPath": str(tr),
            "workspacePaths": [str(self.workspace)],
            "lastUserInput": meta_lit_prompt,
            "invocationNum": 1,
            "initialNumSteps": 61,
        }
        ml_out0 = self._run_hook(ml_payload)
        self.assertIn("injectSteps", ml_out0)
        self.assertEqual(
            json.loads((self.data_dir / "sessions" / "conv-meta-literal.json").read_text(encoding="utf-8"))["lastUserStepIndex"],
            60,
        )
        _write_transcript(
            tr,
            [
                {"step_index": 60, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": wrapped_meta_lit},
                {"step_index": 61, "type": "PLANNER_RESPONSE", "status": "DONE", "tool_calls": [{"name": "exit_plan_mode"}]},
                {"step_index": 62, "type": "SYSTEM_MESSAGE", "status": "DONE", "content": "Plan mode is off:"},
            ],
        )
        ml_handoff = self._run_hook({**ml_payload, "invocationNum": 1, "initialNumSteps": 64})
        self.assertEqual(ml_handoff, {})

        # 4) User prompt quoting a full </USER_REQUEST><ADDITIONAL_METADATA>...</ADDITIONAL_METADATA> example
        # must still match the real outer envelope boundary, not stop early at the quoted block.
        quoted_env_prompt = "++code explain </USER_REQUEST><ADDITIONAL_METADATA>x</ADDITIONAL_METADATA>"
        wrapped_quoted_env = (
            f"<USER_REQUEST>\n{quoted_env_prompt}\n</USER_REQUEST>\n"
            "<ADDITIONAL_METADATA>\ntime\n</ADDITIONAL_METADATA>"
        )
        _write_transcript(
            tr,
            [
                {"step_index": 70, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": wrapped_quoted_env},
            ],
        )
        qe_payload = {
            "conversationId": "conv-quoted-env",
            "transcriptPath": str(tr),
            "workspacePaths": [str(self.workspace)],
            "lastUserInput": quoted_env_prompt,
            "invocationNum": 1,
            "initialNumSteps": 71,
        }
        qe_out0 = self._run_hook(qe_payload)
        self.assertIn("injectSteps", qe_out0)
        self.assertEqual(
            json.loads((self.data_dir / "sessions" / "conv-quoted-env.json").read_text(encoding="utf-8"))["lastUserStepIndex"],
            70,
        )

        # 5) Truncated inside outer </USER_REQUEST> closing tag (all cut lengths) must still match intact prompt
        close_tag = "</USER_REQUEST>"
        for cut in range(1, len(close_tag)):
            step_obj = {
                "step_index": 75,
                "type": "USER_INPUT",
                "source": "USER_EXPLICIT",
                "content": f"<USER_REQUEST>\n++code fix parser\n{close_tag[:cut]}",
                "truncated_fields": ["content"],
            }
            self.assertTrue(hook._step_matches_input(step_obj, "++code fix parser"), f"cut={cut} failed")

        # 6) Visible trailing whitespace in truncated prompt must NOT be stripped before prefix match
        _write_transcript(
            tr,
            [
                {
                    "step_index": 60,
                    "type": "USER_INPUT",
                    "source": "USER_EXPLICIT",
                    "content": "<USER_REQUEST>\n++code rename foo ",
                    "truncated_fields": ["content"],
                },
            ],
        )
        ws_out0 = self._run_hook(
            {
                "conversationId": "conv-trunc-space",
                "transcriptPath": str(tr),
                "workspacePaths": [str(self.workspace)],
                "lastUserInput": "++code rename foobar",
                "invocationNum": 1,
                "initialNumSteps": 61,
            }
        )
        self.assertIn("injectSteps", ws_out0)
        self.assertEqual(
            json.loads((self.data_dir / "sessions" / "conv-trunc-space.json").read_text(encoding="utf-8"))["lastUserStepIndex"],
            -1,
        )

        # 7) Prompt quoting unclosed </USER_REQUEST><ADDITIONAL_METADATA> must still match outer boundary
        unclosed_quote_prompt = "++code explain </USER_REQUEST><ADDITIONAL_METADATA>"
        wrapped_unclosed_quote = (
            f"<USER_REQUEST>\n{unclosed_quote_prompt}\n</USER_REQUEST>\n"
            "<ADDITIONAL_METADATA>\ntime\n</ADDITIONAL_METADATA>"
        )
        _write_transcript(
            tr,
            [
                {"step_index": 80, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": wrapped_unclosed_quote},
            ],
        )
        uq_payload = {
            "conversationId": "conv-unclosed-quote",
            "transcriptPath": str(tr),
            "workspacePaths": [str(self.workspace)],
            "lastUserInput": unclosed_quote_prompt,
            "invocationNum": 1,
            "initialNumSteps": 81,
        }
        uq_out0 = self._run_hook(uq_payload)
        self.assertIn("injectSteps", uq_out0)
        self.assertEqual(
            json.loads((self.data_dir / "sessions" / "conv-unclosed-quote.json").read_text(encoding="utf-8"))["lastUserStepIndex"],
            80,
        )
        _write_transcript(
            tr,
            [
                {"step_index": 80, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": wrapped_unclosed_quote},
                {"step_index": 81, "type": "PLANNER_RESPONSE", "status": "DONE", "tool_calls": [{"name": "exit_plan_mode"}]},
                {"step_index": 82, "type": "SYSTEM_MESSAGE", "status": "DONE", "content": "Plan mode is off:"},
            ],
        )
        self.assertEqual(self._run_hook({**uq_payload, "invocationNum": 1, "initialNumSteps": 84}), {})

        # 8) Prompt with trailing space ("++code fix parser ") truncated across all 14 </USER_REQUEST> cuts
        for cut in range(1, len(close_tag)):
            step_obj = {
                "step_index": 85,
                "type": "USER_INPUT",
                "source": "USER_EXPLICIT",
                "content": f"<USER_REQUEST>\n++code fix parser \n{close_tag[:cut]}",
                "truncated_fields": ["content"],
            }
            self.assertTrue(hook._step_matches_input(step_obj, "++code fix parser "), f"space cut={cut} failed")

        # 9) Bare (unwrapped) truncated text ending in '<' must NOT falsely match without '<'
        bare_lt_step = {
            "step_index": 70,
            "type": "USER_INPUT",
            "source": "USER_EXPLICIT",
            "content": "++code compare x<",
            "truncated_fields": ["content"],
        }
        self.assertFalse(hook._step_matches_input(bare_lt_step, "++code compare x"))

        # 10) Closed truncated-metadata step must NOT prefix-match a follow-up prompt that includes closing tags
        closed_trunc_step = {
            "step_index": 80,
            "type": "USER_INPUT",
            "source": "USER_EXPLICIT",
            "content": "<USER_REQUEST>\n++code fix parser\n</USER_REQUEST>\n<ADDITIONAL_METADATA",
            "truncated_fields": ["content"],
        }
        self.assertFalse(
            hook._step_matches_input(
                closed_trunc_step,
                "++code fix parser\n</USER_REQUEST>\n<ADDITIONAL_METADATA>\nextra",
            )
        )

        # 11) Opening newline without closing newline (<USER_REQUEST>\n... </USER_REQUEST>) must match
        mixed_nl_step = {
            "step_index": 81,
            "type": "USER_INPUT",
            "source": "USER_EXPLICIT",
            "content": "<USER_REQUEST>\n++code fix parser</USER_REQUEST>",
        }
        self.assertTrue(hook._step_matches_input(mixed_nl_step, "++code fix parser"))

        # 12) Whitespace-only truncated raw step must NOT match arbitrary trigger prompts
        ws_only_step = {
            "step_index": 82,
            "type": "USER_INPUT",
            "source": "USER_EXPLICIT",
            "content": "   \n  ",
            "truncated_fields": ["content"],
        }
        self.assertFalse(hook._step_matches_input(ws_only_step, "++code anything"))

        # 13) Prompt literally equal to or one-newline shorter than closed truncated-metadata body must NOT match
        self.assertFalse(
            hook._step_matches_input(
                closed_trunc_step,
                "++code fix parser\n</USER_REQUEST>\n<ADDITIONAL_METADATA",
            )
        )
        closed_trunc_nl_step = {
            "step_index": 80,
            "type": "USER_INPUT",
            "source": "USER_EXPLICIT",
            "content": "<USER_REQUEST>\n++code fix parser\n</USER_REQUEST>\n<ADDITIONAL_METADATA>\n",
            "truncated_fields": ["content"],
        }
        self.assertFalse(
            hook._step_matches_input(
                closed_trunc_nl_step,
                "++code fix parser\n</USER_REQUEST>\n<ADDITIONAL_METADATA>",
            )
        )

        # 14) Two turns sharing identical 2407-char truncated prefix (differing only after truncation cut:
        # macOS vs Linux) must treat prior bound step 80 as stale on Turn 2 instead of re-binding step 80,
        # even if Turn 1 saved lastUserStepIndex=-1 when transcript wasn't flushed yet and Turn 2 sees ended step 80+81.
        common_2407 = "++code " + ("z" * 2400)
        p_macos = common_2407 + "macOS"
        p_linux = common_2407 + "Linux"
        _write_transcript(tr, [])
        t_mac_unflushed = self._run_hook(
            {
                "conversationId": "conv-2407-unflushed",
                "transcriptPath": str(tr),
                "workspacePaths": [str(self.workspace)],
                "lastUserInput": p_macos,
                "invocationNum": 1,
                "initialNumSteps": 79,
            }
        )
        self.assertIn("injectSteps", t_mac_unflushed)
        self.assertEqual(
            json.loads((self.data_dir / "sessions" / "conv-2407-unflushed.json").read_text(encoding="utf-8"))["lastUserStepIndex"],
            -1,
        )
        _write_transcript(
            tr,
            [
                {
                    "step_index": 80,
                    "type": "USER_INPUT",
                    "source": "USER_EXPLICIT",
                    "content": f"<USER_REQUEST>\n{common_2407}",
                    "truncated_fields": ["content"],
                },
                {"step_index": 81, "type": "PLANNER_RESPONSE", "status": "DONE", "tool_calls": []},
            ],
        )
        t_lin_u0 = self._run_hook(
            {
                "conversationId": "conv-2407-unflushed",
                "transcriptPath": str(tr),
                "workspacePaths": [str(self.workspace)],
                "lastUserInput": p_linux,
                "invocationNum": 1,
                "initialNumSteps": 83,
            }
        )
        self.assertIn("injectSteps", t_lin_u0)
        lin_u_sess = json.loads((self.data_dir / "sessions" / "conv-2407-unflushed.json").read_text(encoding="utf-8"))
        self.assertEqual(lin_u_sess["lastUserStepIndex"], -1)
        self.assertEqual(lin_u_sess["staleUserStepIndex"], 80)
        _write_transcript(
            tr,
            [
                {
                    "step_index": 80,
                    "type": "USER_INPUT",
                    "source": "USER_EXPLICIT",
                    "content": f"<USER_REQUEST>\n{common_2407}",
                    "truncated_fields": ["content"],
                },
                {"step_index": 81, "type": "PLANNER_RESPONSE", "status": "DONE", "tool_calls": []},
                {
                    "step_index": 84,
                    "type": "USER_INPUT",
                    "source": "USER_EXPLICIT",
                    "content": f"<USER_REQUEST>\n{p_linux}\n</USER_REQUEST>",
                },
            ],
        )
        self.assertEqual(
            self._run_hook(
                {
                    "conversationId": "conv-2407-unflushed",
                    "transcriptPath": str(tr),
                    "workspacePaths": [str(self.workspace)],
                    "lastUserInput": p_linux,
                    "invocationNum": 2,
                    "initialNumSteps": 85,
                }
            ),
            {},
        )
        self.assertEqual(
            json.loads((self.data_dir / "sessions" / "conv-2407-unflushed.json").read_text(encoding="utf-8"))["lastUserStepIndex"],
            84,
        )

        # 1) Turn 1 unflushed (init=79, 80, 81) -> Turn 2 sees ONLY Turn 1's truncated step 80 (no step 81 yet)
        for init_val in (79, 80, 81):
            cid_param = f"conv-2407-no-81-{init_val}"
            _write_transcript(tr, [])
            t_mac_only_step80 = self._run_hook(
                {
                    "conversationId": cid_param,
                    "transcriptPath": str(tr),
                    "workspacePaths": [str(self.workspace)],
                    "lastUserInput": p_macos,
                    "invocationNum": 1,
                    "initialNumSteps": init_val,
                }
            )
            self.assertIn("injectSteps", t_mac_only_step80)
            _write_transcript(
                tr,
                [
                    {
                        "step_index": 80,
                        "type": "USER_INPUT",
                        "source": "USER_EXPLICIT",
                        "content": f"<USER_REQUEST>\n{common_2407}",
                        "truncated_fields": ["content"],
                    },
                ],
            )
            t_lin_only80 = self._run_hook(
                {
                    "conversationId": cid_param,
                    "transcriptPath": str(tr),
                    "workspacePaths": [str(self.workspace)],
                    "lastUserInput": p_linux,
                    "invocationNum": 1,
                    "initialNumSteps": 83,
                }
            )
            self.assertIn("injectSteps", t_lin_only80)
            sess_no81 = json.loads((self.data_dir / "sessions" / f"{cid_param}.json").read_text(encoding="utf-8"))
            self.assertEqual(sess_no81["lastUserStepIndex"], -1)
            self.assertEqual(sess_no81["staleUserStepIndex"], 80)
            _write_transcript(
                tr,
                [
                    {
                        "step_index": 80,
                        "type": "USER_INPUT",
                        "source": "USER_EXPLICIT",
                        "content": f"<USER_REQUEST>\n{common_2407}",
                        "truncated_fields": ["content"],
                    },
                    {
                        "step_index": 84,
                        "type": "USER_INPUT",
                        "source": "USER_EXPLICIT",
                        "content": f"<USER_REQUEST>\n{p_linux}\n</USER_REQUEST>",
                    },
                ],
            )
            self.assertEqual(
                self._run_hook(
                    {
                        "conversationId": cid_param,
                        "transcriptPath": str(tr),
                        "workspacePaths": [str(self.workspace)],
                        "lastUserInput": p_linux,
                        "invocationNum": 2,
                        "initialNumSteps": 85,
                    }
                ),
                {},
            )
            self.assertEqual(
                json.loads((self.data_dir / "sessions" / f"{cid_param}.json").read_text(encoding="utf-8"))["lastUserStepIndex"],
                84,
            )

        # 2) Canceled handoff followed by trailing SYSTEM_MESSAGE/DONE must NOT look like active handoff
        _write_transcript(
            tr,
            [
                {"step_index": 0, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": wrapped_only},
            ],
        )
        cancel_sys_payload = {
            "conversationId": "conv-canceled-sysmsg",
            "transcriptPath": str(tr),
            "workspacePaths": [str(self.workspace)],
            "lastUserInput": only_prompt,
            "invocationNum": 1,
            "initialNumSteps": 1,
        }
        self.assertIn("MODE=only", self._run_hook(cancel_sys_payload)["injectSteps"][0]["systemMessage"]["systemMessage"])
        _write_transcript(
            tr,
            [
                {"step_index": 0, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": wrapped_only},
                {"step_index": 1, "type": "PLANNER_RESPONSE", "status": "DONE", "tool_calls": [{"name": "exit_plan_mode"}]},
                {"step_index": 2, "type": "GENERIC", "status": "CANCELED"},
                {"step_index": 3, "type": "SYSTEM_MESSAGE", "status": "DONE", "content": "Canceled"},
            ],
        )
        cs_next = self._run_hook({**cancel_sys_payload, "invocationNum": 1, "initialNumSteps": 5})
        self.assertIn("injectSteps", cs_next)
        self.assertIn("MODE=only", cs_next["injectSteps"][0]["systemMessage"]["systemMessage"])

        # 3) Quoted closed tags followed by truncated outer </USER_REQ must still match
        self.assertTrue(
            hook._step_matches_input(
                {
                    "step_index": 88,
                    "type": "USER_INPUT",
                    "source": "USER_EXPLICIT",
                    "content": (
                        "<USER_REQUEST>\n"
                        "++code explain </USER_REQUEST><ADDITIONAL_METADATA>x</ADDITIONAL_METADATA>\n"
                        "</USER_REQ"
                    ),
                    "truncated_fields": ["content"],
                },
                "++code explain </USER_REQUEST><ADDITIONAL_METADATA>x</ADDITIONAL_METADATA>",
            )
        )

        # 4) Turn A already bound step 80, finished at 81, Turn B arrives with truncated step 84 (also matching common_2407):
        # since Turn A already bound 80, step 84 MUST bind to Turn B immediately so exit_plan_mode handoff dedupes!
        _write_transcript(
            tr,
            [
                {
                    "step_index": 80,
                    "type": "USER_INPUT",
                    "source": "USER_EXPLICIT",
                    "content": f"<USER_REQUEST>\n{common_2407}",
                    "truncated_fields": ["content"],
                },
            ],
        )
        self.assertIn(
            "injectSteps",
            self._run_hook(
                {
                    "conversationId": "conv-2407-bound-80-to-84",
                    "transcriptPath": str(tr),
                    "workspacePaths": [str(self.workspace)],
                    "lastUserInput": p_macos,
                    "invocationNum": 1,
                    "initialNumSteps": 81,
                }
            ),
        )
        _write_transcript(
            tr,
            [
                {
                    "step_index": 80,
                    "type": "USER_INPUT",
                    "source": "USER_EXPLICIT",
                    "content": f"<USER_REQUEST>\n{common_2407}",
                    "truncated_fields": ["content"],
                },
                {"step_index": 81, "type": "PLANNER_RESPONSE", "status": "DONE", "tool_calls": []},
                {
                    "step_index": 84,
                    "type": "USER_INPUT",
                    "source": "USER_EXPLICIT",
                    "content": f"<USER_REQUEST>\n{common_2407}",
                    "truncated_fields": ["content"],
                },
            ],
        )
        self.assertIn(
            "injectSteps",
            self._run_hook(
                {
                    "conversationId": "conv-2407-bound-80-to-84",
                    "transcriptPath": str(tr),
                    "workspacePaths": [str(self.workspace)],
                    "lastUserInput": p_linux,
                    "invocationNum": 1,
                    "initialNumSteps": 85,
                }
            ),
        )
        self.assertEqual(
            json.loads((self.data_dir / "sessions" / "conv-2407-bound-80-to-84.json").read_text(encoding="utf-8"))["lastUserStepIndex"],
            84,
        )
        _write_transcript(
            tr,
            [
                {
                    "step_index": 80,
                    "type": "USER_INPUT",
                    "source": "USER_EXPLICIT",
                    "content": f"<USER_REQUEST>\n{common_2407}",
                    "truncated_fields": ["content"],
                },
                {"step_index": 81, "type": "PLANNER_RESPONSE", "status": "DONE", "tool_calls": []},
                {
                    "step_index": 84,
                    "type": "USER_INPUT",
                    "source": "USER_EXPLICIT",
                    "content": f"<USER_REQUEST>\n{common_2407}",
                    "truncated_fields": ["content"],
                },
                {"step_index": 85, "type": "PLANNER_RESPONSE", "status": "DONE", "tool_calls": [{"name": "exit_plan_mode"}]},
                {"step_index": 86, "type": "SYSTEM_MESSAGE", "status": "DONE", "content": "Plan mode is off:"},
            ],
        )
        self.assertEqual(
            self._run_hook(
                {
                    "conversationId": "conv-2407-bound-80-to-84",
                    "transcriptPath": str(tr),
                    "workspacePaths": [str(self.workspace)],
                    "lastUserInput": p_linux,
                    "invocationNum": 1,
                    "initialNumSteps": 88,
                }
            ),
            {},
        )

        # 5) Turn A unflushed (init=79), Turn B Inv 1 also sees empty transcript (init=83),
        # Turn B Inv 2 sees delayed truncated step 80 -> returns {} & marks 80 stale (NOT binding 80!),
        # Turn B Inv 3 sees step 84 -> returns {} & backfills 84!
        _write_transcript(tr, [])
        self.assertIn(
            "injectSteps",
            self._run_hook(
                {
                    "conversationId": "conv-2407-late-80",
                    "transcriptPath": str(tr),
                    "workspacePaths": [str(self.workspace)],
                    "lastUserInput": p_macos,
                    "invocationNum": 1,
                    "initialNumSteps": 79,
                }
            ),
        )
        self.assertIn(
            "injectSteps",
            self._run_hook(
                {
                    "conversationId": "conv-2407-late-80",
                    "transcriptPath": str(tr),
                    "workspacePaths": [str(self.workspace)],
                    "lastUserInput": p_linux,
                    "invocationNum": 1,
                    "initialNumSteps": 83,
                }
            ),
        )
        _write_transcript(
            tr,
            [
                {
                    "step_index": 80,
                    "type": "USER_INPUT",
                    "source": "USER_EXPLICIT",
                    "content": f"<USER_REQUEST>\n{common_2407}",
                    "truncated_fields": ["content"],
                },
            ],
        )
        self.assertEqual(
            self._run_hook(
                {
                    "conversationId": "conv-2407-late-80",
                    "transcriptPath": str(tr),
                    "workspacePaths": [str(self.workspace)],
                    "lastUserInput": p_linux,
                    "invocationNum": 2,
                    "initialNumSteps": 85,
                }
            ),
            {},
        )
        late_sess = json.loads((self.data_dir / "sessions" / "conv-2407-late-80.json").read_text(encoding="utf-8"))
        self.assertEqual(late_sess["lastUserStepIndex"], -1)
        self.assertEqual(late_sess["staleUserStepIndex"], 80)
        _write_transcript(
            tr,
            [
                {
                    "step_index": 80,
                    "type": "USER_INPUT",
                    "source": "USER_EXPLICIT",
                    "content": f"<USER_REQUEST>\n{common_2407}",
                    "truncated_fields": ["content"],
                },
                {
                    "step_index": 84,
                    "type": "USER_INPUT",
                    "source": "USER_EXPLICIT",
                    "content": f"<USER_REQUEST>\n{p_linux}\n</USER_REQUEST>",
                },
            ],
        )
        self.assertEqual(
            self._run_hook(
                {
                    "conversationId": "conv-2407-late-80",
                    "transcriptPath": str(tr),
                    "workspacePaths": [str(self.workspace)],
                    "lastUserInput": p_linux,
                    "invocationNum": 3,
                    "initialNumSteps": 87,
                }
            ),
            {},
        )
        self.assertEqual(
            json.loads((self.data_dir / "sessions" / "conv-2407-late-80.json").read_text(encoding="utf-8"))["lastUserStepIndex"],
            84,
        )

        # 6) Turn A unflushed (init=79, 80, 81), then [step 80 (A), step 81 (A ended), step 84 (B truncated)]
        # appears together at either Turn B Inv 1 or Inv 2: step 84 MUST bind to B so exit_plan_mode handoff returns {}.
        for a_init in (79, 80, 81):
            for first_seen_inv in (1, 2):
                cid_both = f"conv-2407-both-init{a_init}-inv{first_seen_inv}"
                _write_transcript(tr, [])
                self.assertIn(
                    "injectSteps",
                    self._run_hook(
                        {
                            "conversationId": cid_both,
                            "transcriptPath": str(tr),
                            "workspacePaths": [str(self.workspace)],
                            "lastUserInput": p_macos,
                            "invocationNum": 1,
                            "initialNumSteps": a_init,
                        }
                    ),
                )
                if first_seen_inv == 2:
                    self.assertIn(
                        "injectSteps",
                        self._run_hook(
                            {
                                "conversationId": cid_both,
                                "transcriptPath": str(tr),
                                "workspacePaths": [str(self.workspace)],
                                "lastUserInput": p_linux,
                                "invocationNum": 1,
                                "initialNumSteps": 83,
                            }
                        ),
                    )
                _write_transcript(
                    tr,
                    [
                        {
                            "step_index": 80,
                            "type": "USER_INPUT",
                            "source": "USER_EXPLICIT",
                            "content": f"<USER_REQUEST>\n{p_macos}\n</USER_REQUEST>",
                        },
                        {"step_index": 81, "type": "PLANNER_RESPONSE", "status": "DONE", "tool_calls": []},
                        {
                            "step_index": 84,
                            "type": "USER_INPUT",
                            "source": "USER_EXPLICIT",
                            "content": f"<USER_REQUEST>\n{common_2407}",
                            "truncated_fields": ["content"],
                        },
                    ],
                )
                res_seen = self._run_hook(
                    {
                        "conversationId": cid_both,
                        "transcriptPath": str(tr),
                        "workspacePaths": [str(self.workspace)],
                        "lastUserInput": p_linux,
                        "invocationNum": first_seen_inv,
                        "initialNumSteps": 85 if first_seen_inv == 1 else 87,
                    }
                )
                if first_seen_inv == 1:
                    self.assertIn("injectSteps", res_seen)
                else:
                    self.assertEqual(res_seen, {})
                self.assertEqual(
                    json.loads((self.data_dir / "sessions" / f"{cid_both}.json").read_text(encoding="utf-8"))["lastUserStepIndex"],
                    84,
                )
                _write_transcript(
                    tr,
                    [
                        {
                            "step_index": 80,
                            "type": "USER_INPUT",
                            "source": "USER_EXPLICIT",
                            "content": f"<USER_REQUEST>\n{p_macos}\n</USER_REQUEST>",
                        },
                        {"step_index": 81, "type": "PLANNER_RESPONSE", "status": "DONE", "tool_calls": []},
                        {
                            "step_index": 84,
                            "type": "USER_INPUT",
                            "source": "USER_EXPLICIT",
                            "content": f"<USER_REQUEST>\n{common_2407}",
                            "truncated_fields": ["content"],
                        },
                        {"step_index": 85, "type": "PLANNER_RESPONSE", "status": "DONE", "tool_calls": [{"name": "exit_plan_mode"}]},
                        {"step_index": 86, "type": "SYSTEM_MESSAGE", "status": "DONE", "content": "Plan mode is off:"},
                    ],
                )
                self.assertEqual(
                    self._run_hook(
                        {
                            "conversationId": cid_both,
                            "transcriptPath": str(tr),
                            "workspacePaths": [str(self.workspace)],
                            "lastUserInput": p_linux,
                            "invocationNum": 1,
                            "initialNumSteps": 88,
                        }
                    ),
                    {},
                )

        # 9) Concurrent append between initial read (seeing step 80) and wider read (appending step 84)
        # must still evaluate target step 80 rather than jumping to newly appended step 84.
        toctou_steps1 = [
            {"step_index": 80, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": f"<USER_REQUEST>\n{common_2407}", "truncated_fields": ["content"]},
        ]
        toctou_steps2 = toctou_steps1 + [
            {"step_index": 81, "type": "PLANNER_RESPONSE", "status": "DONE", "tool_calls": []},
            {"step_index": 84, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": f"<USER_REQUEST>\n{p_linux}\n</USER_REQUEST>"},
        ]
        with mock.patch.object(hook, "_load_recent_transcript_steps", return_value=toctou_steps2):
            self.assertFalse(
                hook._has_prior_explicit_step_matching(
                    {},
                    toctou_steps1,
                    0,
                    p_macos,
                    target_step_index=80,
                    after_step_index=None,
                )
            )

        # 7) Stale prior step (staleUserStepIndex=70 from A1->A2 repeat) must NOT count as fresh prior evidence:
        # Turn B Inv 1 seeing [step 70 (A1), step 80 (A2 truncated)] must mark 80 stale, then Inv 2 backfills 84.
        _write_transcript(
            tr,
            [
                {"step_index": 70, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": f"<USER_REQUEST>\n{p_macos}\n</USER_REQUEST>"},
            ],
        )
        self.assertIn(
            "injectSteps",
            self._run_hook(
                {
                    "conversationId": "conv-2407-stale-70",
                    "transcriptPath": str(tr),
                    "workspacePaths": [str(self.workspace)],
                    "lastUserInput": p_macos,
                    "invocationNum": 1,
                    "initialNumSteps": 71,
                }
            ),
        )
        _write_transcript(
            tr,
            [
                {"step_index": 70, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": f"<USER_REQUEST>\n{p_macos}\n</USER_REQUEST>"},
                {"step_index": 71, "type": "PLANNER_RESPONSE", "status": "DONE", "tool_calls": []},
            ],
        )
        self.assertIn(
            "injectSteps",
            self._run_hook(
                {
                    "conversationId": "conv-2407-stale-70",
                    "transcriptPath": str(tr),
                    "workspacePaths": [str(self.workspace)],
                    "lastUserInput": p_macos,
                    "invocationNum": 1,
                    "initialNumSteps": 75,
                }
            ),
        )
        self.assertEqual(
            json.loads((self.data_dir / "sessions" / "conv-2407-stale-70.json").read_text(encoding="utf-8"))["staleUserStepIndex"],
            70,
        )
        _write_transcript(
            tr,
            [
                {"step_index": 70, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": f"<USER_REQUEST>\n{p_macos}\n</USER_REQUEST>"},
                {"step_index": 71, "type": "PLANNER_RESPONSE", "status": "DONE", "tool_calls": []},
                {"step_index": 80, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": f"<USER_REQUEST>\n{common_2407}", "truncated_fields": ["content"]},
            ],
        )
        self.assertIn(
            "injectSteps",
            self._run_hook(
                {
                    "conversationId": "conv-2407-stale-70",
                    "transcriptPath": str(tr),
                    "workspacePaths": [str(self.workspace)],
                    "lastUserInput": p_linux,
                    "invocationNum": 1,
                    "initialNumSteps": 83,
                }
            ),
        )
        sess_stale70 = json.loads((self.data_dir / "sessions" / "conv-2407-stale-70.json").read_text(encoding="utf-8"))
        self.assertEqual(sess_stale70["lastUserStepIndex"], -1)
        self.assertEqual(sess_stale70["staleUserStepIndex"], 80)
        _write_transcript(
            tr,
            [
                {"step_index": 70, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": f"<USER_REQUEST>\n{p_macos}\n</USER_REQUEST>"},
                {"step_index": 71, "type": "PLANNER_RESPONSE", "status": "DONE", "tool_calls": []},
                {"step_index": 80, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": f"<USER_REQUEST>\n{common_2407}", "truncated_fields": ["content"]},
                {"step_index": 84, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": f"<USER_REQUEST>\n{p_linux}\n</USER_REQUEST>"},
            ],
        )
        self.assertEqual(
            self._run_hook(
                {
                    "conversationId": "conv-2407-stale-70",
                    "transcriptPath": str(tr),
                    "workspacePaths": [str(self.workspace)],
                    "lastUserInput": p_linux,
                    "invocationNum": 2,
                    "initialNumSteps": 85,
                }
            ),
            {},
        )
        self.assertEqual(
            json.loads((self.data_dir / "sessions" / "conv-2407-stale-70.json").read_text(encoding="utf-8"))["lastUserStepIndex"],
            84,
        )

        # 8) Transcript > 128 KiB where Turn A (step 80) sits outside the initial tail window while Turn B (step 150)
        # sits inside the last 128 KiB must expand window on demand and bind step 150 to Turn B.
        big_tr = self.root / "transcript_big.jsonl"
        filler_padding = "x" * 18000
        big_steps = [
            {"step_index": 80, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": f"<USER_REQUEST>\n{p_macos}\n</USER_REQUEST>"},
        ]
        for pad_idx in range(81, 90):
            big_steps.append({"step_index": pad_idx, "type": "GENERIC", "status": "DONE", "content": filler_padding})
        big_steps.append(
            {"step_index": 150, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": f"<USER_REQUEST>\n{common_2407}", "truncated_fields": ["content"]}
        )
        _write_transcript(big_tr, [])
        self.assertIn(
            "injectSteps",
            self._run_hook(
                {
                    "conversationId": "conv-big-window",
                    "transcriptPath": str(big_tr),
                    "workspacePaths": [str(self.workspace)],
                    "lastUserInput": p_macos,
                    "invocationNum": 1,
                    "initialNumSteps": 79,
                }
            ),
        )
        _write_transcript(big_tr, big_steps)
        self.assertGreater(big_tr.stat().st_size, 140000)
        self.assertIn(
            "injectSteps",
            self._run_hook(
                {
                    "conversationId": "conv-big-window",
                    "transcriptPath": str(big_tr),
                    "workspacePaths": [str(self.workspace)],
                    "lastUserInput": p_linux,
                    "invocationNum": 1,
                    "initialNumSteps": 151,
                }
            ),
        )
        self.assertEqual(
            json.loads((self.data_dir / "sessions" / "conv-big-window.json").read_text(encoding="utf-8"))["lastUserStepIndex"],
            150,
        )
        _write_transcript(
            big_tr,
            big_steps
            + [
                {"step_index": 151, "type": "PLANNER_RESPONSE", "status": "DONE", "tool_calls": [{"name": "exit_plan_mode"}]},
                {"step_index": 152, "type": "SYSTEM_MESSAGE", "status": "DONE", "content": "Plan mode is off:"},
            ],
        )
        self.assertEqual(
            self._run_hook(
                {
                    "conversationId": "conv-big-window",
                    "transcriptPath": str(big_tr),
                    "workspacePaths": [str(self.workspace)],
                    "lastUserInput": p_linux,
                    "invocationNum": 1,
                    "initialNumSteps": 154,
                }
            ),
            {},
        )

        # 10) A1 binds truncated 70 -> A2 repeats (init=79, sees 70+71, marks 70 stale) -> B Inv 1 still sees 70+71
        # -> must preserve staleUserStepIndex=70, priorInitBound=78, and A2's priorUnboundUserInput so B Inv 2
        # rejects delayed A2 step 80 and B Inv 3 backfills real step 84 without re-injecting.
        cid_a1_a2_b = "conv-2407-a1-70-a2-79-b-late-80"
        _write_transcript(
            tr,
            [{"step_index": 70, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": f"<USER_REQUEST>\n{common_2407}", "truncated_fields": ["content"]}],
        )
        self.assertIn(
            "injectSteps",
            self._run_hook(
                {
                    "conversationId": cid_a1_a2_b,
                    "transcriptPath": str(tr),
                    "workspacePaths": [str(self.workspace)],
                    "lastUserInput": p_macos,
                    "invocationNum": 1,
                    "initialNumSteps": 71,
                }
            ),
        )
        _write_transcript(
            tr,
            [
                {"step_index": 70, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": f"<USER_REQUEST>\n{common_2407}", "truncated_fields": ["content"]},
                {"step_index": 71, "type": "PLANNER_RESPONSE", "status": "DONE", "tool_calls": []},
            ],
        )
        self.assertIn(
            "injectSteps",
            self._run_hook(
                {
                    "conversationId": cid_a1_a2_b,
                    "transcriptPath": str(tr),
                    "workspacePaths": [str(self.workspace)],
                    "lastUserInput": p_macos,
                    "invocationNum": 1,
                    "initialNumSteps": 79,
                }
            ),
        )
        self.assertIn(
            "injectSteps",
            self._run_hook(
                {
                    "conversationId": cid_a1_a2_b,
                    "transcriptPath": str(tr),
                    "workspacePaths": [str(self.workspace)],
                    "lastUserInput": p_linux,
                    "invocationNum": 1,
                    "initialNumSteps": 83,
                }
            ),
        )
        sess_b1 = json.loads((self.data_dir / "sessions" / f"{cid_a1_a2_b}.json").read_text(encoding="utf-8"))
        self.assertEqual(sess_b1["staleUserStepIndex"], 70)
        self.assertEqual(sess_b1["priorInitBound"], 78)
        self.assertEqual(sess_b1["priorUnboundUserInput"], p_macos)

        _write_transcript(
            tr,
            [
                {"step_index": 70, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": f"<USER_REQUEST>\n{common_2407}", "truncated_fields": ["content"]},
                {"step_index": 71, "type": "PLANNER_RESPONSE", "status": "DONE", "tool_calls": []},
                {"step_index": 80, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": f"<USER_REQUEST>\n{common_2407}", "truncated_fields": ["content"]},
            ],
        )
        self.assertEqual(
            self._run_hook(
                {
                    "conversationId": cid_a1_a2_b,
                    "transcriptPath": str(tr),
                    "workspacePaths": [str(self.workspace)],
                    "lastUserInput": p_linux,
                    "invocationNum": 2,
                    "initialNumSteps": 85,
                }
            ),
            {},
        )
        self.assertEqual(
            json.loads((self.data_dir / "sessions" / f"{cid_a1_a2_b}.json").read_text(encoding="utf-8"))["staleUserStepIndex"],
            80,
        )
        _write_transcript(
            tr,
            [
                {"step_index": 70, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": f"<USER_REQUEST>\n{common_2407}", "truncated_fields": ["content"]},
                {"step_index": 71, "type": "PLANNER_RESPONSE", "status": "DONE", "tool_calls": []},
                {"step_index": 80, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": f"<USER_REQUEST>\n{common_2407}", "truncated_fields": ["content"]},
                {"step_index": 84, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": f"<USER_REQUEST>\n{p_linux}\n</USER_REQUEST>"},
            ],
        )
        self.assertEqual(
            self._run_hook(
                {
                    "conversationId": cid_a1_a2_b,
                    "transcriptPath": str(tr),
                    "workspacePaths": [str(self.workspace)],
                    "lastUserInput": p_linux,
                    "invocationNum": 3,
                    "initialNumSteps": 87,
                }
            ),
            {},
        )
        self.assertEqual(
            json.loads((self.data_dir / "sessions" / f"{cid_a1_a2_b}.json").read_text(encoding="utf-8"))["lastUserStepIndex"],
            84,
        )

        # Also verify bound step 80 from Turn 1 blocks re-binding step 80 on Turn 2 even without step 81.
        _write_transcript(
            tr,
            [
                {
                    "step_index": 80,
                    "type": "USER_INPUT",
                    "source": "USER_EXPLICIT",
                    "content": f"<USER_REQUEST>\n{common_2407}",
                    "truncated_fields": ["content"],
                },
            ],
        )
        t_mac = self._run_hook(
            {
                "conversationId": "conv-2407-trunc",
                "transcriptPath": str(tr),
                "workspacePaths": [str(self.workspace)],
                "lastUserInput": p_macos,
                "invocationNum": 1,
                "initialNumSteps": 81,
            }
        )
        self.assertIn("injectSteps", t_mac)
        self.assertEqual(
            json.loads((self.data_dir / "sessions" / "conv-2407-trunc.json").read_text(encoding="utf-8"))["lastUserStepIndex"],
            80,
        )
        # Turn 2 sends Linux while transcript still only shows truncated step 80 -> step 80 belongs to Turn 1!
        t_lin0 = self._run_hook(
            {
                "conversationId": "conv-2407-trunc",
                "transcriptPath": str(tr),
                "workspacePaths": [str(self.workspace)],
                "lastUserInput": p_linux,
                "invocationNum": 1,
                "initialNumSteps": 83,
            }
        )
        self.assertIn("injectSteps", t_lin0)
        lin_sess0 = json.loads((self.data_dir / "sessions" / "conv-2407-trunc.json").read_text(encoding="utf-8"))
        self.assertEqual(lin_sess0["lastUserStepIndex"], -1)
        self.assertEqual(lin_sess0["staleUserStepIndex"], 80)

        _write_transcript(
            tr,
            [
                {
                    "step_index": 80,
                    "type": "USER_INPUT",
                    "source": "USER_EXPLICIT",
                    "content": f"<USER_REQUEST>\n{common_2407}",
                    "truncated_fields": ["content"],
                },
                {
                    "step_index": 84,
                    "type": "USER_INPUT",
                    "source": "USER_EXPLICIT",
                    "content": f"<USER_REQUEST>\n{p_linux}\n</USER_REQUEST>",
                },
            ],
        )
        t_lin1 = self._run_hook(
            {
                "conversationId": "conv-2407-trunc",
                "transcriptPath": str(tr),
                "workspacePaths": [str(self.workspace)],
                "lastUserInput": p_linux,
                "invocationNum": 2,
                "initialNumSteps": 85,
            }
        )
        self.assertEqual(t_lin1, {})
        self.assertEqual(
            json.loads((self.data_dir / "sessions" / "conv-2407-trunc.json").read_text(encoding="utf-8"))["lastUserStepIndex"],
            84,
        )

        # Two consecutive identical prompts where Turn A saw an empty transcript (lastUserStepIndex=-1,
        # initialNumSteps=81) and Turn A's step 80 later appears without its ending step 81 at Turn B Inv 1 or Inv 2.
        for first_seen_inv in (1, 2):
            cid_same = f"conv-same-hash-unbound-inv{first_seen_inv}"
            _write_transcript(tr, [])
            same_a = self._run_hook(
                {
                    "conversationId": cid_same,
                    "transcriptPath": str(tr),
                    "workspacePaths": [str(self.workspace)],
                    "lastUserInput": only_prompt,
                    "invocationNum": 1,
                    "initialNumSteps": 81,
                }
            )
            self.assertIn("MODE=only", same_a["injectSteps"][0]["systemMessage"]["systemMessage"])
            self.assertEqual(
                json.loads((self.data_dir / "sessions" / f"{cid_same}.json").read_text(encoding="utf-8"))["lastUserStepIndex"],
                -1,
            )

            same_b_payload = {
                "conversationId": cid_same,
                "transcriptPath": str(tr),
                "workspacePaths": [str(self.workspace)],
                "lastUserInput": only_prompt,
            }
            if first_seen_inv == 1:
                _write_transcript(
                    tr,
                    [
                        {"step_index": 80, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": wrapped_only},
                    ],
                )
                same_b1 = self._run_hook({**same_b_payload, "invocationNum": 1, "initialNumSteps": 83})
                self.assertIn("MODE=only", same_b1["injectSteps"][0]["systemMessage"]["systemMessage"])
                same_sess1 = json.loads((self.data_dir / "sessions" / f"{cid_same}.json").read_text(encoding="utf-8"))
                self.assertEqual(same_sess1["lastUserStepIndex"], -1)
                self.assertEqual(same_sess1["staleUserStepIndex"], 80)
                self.assertEqual(same_sess1["priorInitBound"], 80)
            else:
                same_b1 = self._run_hook({**same_b_payload, "invocationNum": 1, "initialNumSteps": 83})
                self.assertIn("MODE=only", same_b1["injectSteps"][0]["systemMessage"]["systemMessage"])
                same_sess1 = json.loads((self.data_dir / "sessions" / f"{cid_same}.json").read_text(encoding="utf-8"))
                self.assertEqual(same_sess1["lastUserStepIndex"], -1)
                self.assertEqual(same_sess1["priorInitBound"], 80)

                _write_transcript(
                    tr,
                    [
                        {"step_index": 80, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": wrapped_only},
                    ],
                )
                same_b2 = self._run_hook({**same_b_payload, "invocationNum": 2, "initialNumSteps": 85})
                self.assertEqual(same_b2, {})
                same_sess2 = json.loads((self.data_dir / "sessions" / f"{cid_same}.json").read_text(encoding="utf-8"))
                self.assertEqual(same_sess2["lastUserStepIndex"], -1)
                self.assertEqual(same_sess2["priorInitBound"], 80)

            _write_transcript(
                tr,
                [
                    {"step_index": 80, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": wrapped_only},
                    {"step_index": 81, "type": "PLANNER_RESPONSE", "status": "DONE", "tool_calls": []},
                    {"step_index": 84, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": wrapped_only},
                ],
            )
            same_final = self._run_hook(
                {
                    **same_b_payload,
                    "invocationNum": first_seen_inv + 1,
                    "initialNumSteps": 85 if first_seen_inv == 1 else 87,
                }
            )
            self.assertEqual(same_final, {})
            same_sess_final = json.loads((self.data_dir / "sessions" / f"{cid_same}.json").read_text(encoding="utf-8"))
            self.assertEqual(same_sess_final["lastUserStepIndex"], 84)
            self.assertNotIn("staleUserStepIndex", same_sess_final)
            self.assertNotIn("priorInitBound", same_sess_final)

        # Canceled exit_plan_mode handoff followed by identical ++only prompt on next turn
        # must NOT treat the stale canceled handoff as active: Inv 1 injects MODE=only, Inv 2 dedupes.
        _write_transcript(
            tr,
            [
                {"step_index": 0, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": wrapped_only},
            ],
        )
        cancel_payload = {
            "conversationId": "conv-canceled-handoff",
            "transcriptPath": str(tr),
            "workspacePaths": [str(self.workspace)],
            "lastUserInput": only_prompt,
            "invocationNum": 1,
            "initialNumSteps": 1,
        }
        c_out0 = self._run_hook(cancel_payload)
        self.assertIn("MODE=only", c_out0["injectSteps"][0]["systemMessage"]["systemMessage"])

        _write_transcript(
            tr,
            [
                {"step_index": 0, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": wrapped_only},
                {"step_index": 1, "type": "PLANNER_RESPONSE", "status": "DONE", "tool_calls": [{"name": "exit_plan_mode"}]},
                {"step_index": 2, "type": "GENERIC", "status": "CANCELED"},
            ],
        )
        c_next_out0 = self._run_hook({**cancel_payload, "invocationNum": 1, "initialNumSteps": 4})
        self.assertIn("injectSteps", c_next_out0)
        self.assertIn("MODE=only", c_next_out0["injectSteps"][0]["systemMessage"]["systemMessage"])

        _write_transcript(
            tr,
            [
                {"step_index": 0, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": wrapped_only},
                {"step_index": 1, "type": "PLANNER_RESPONSE", "status": "DONE", "tool_calls": [{"name": "exit_plan_mode"}]},
                {"step_index": 2, "type": "GENERIC", "status": "CANCELED"},
                {"step_index": 3, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": wrapped_only},
            ],
        )
        c_next_out1 = self._run_hook({**cancel_payload, "invocationNum": 2, "initialNumSteps": 6})
        self.assertEqual(c_next_out1, {})
        c_sess = json.loads((self.data_dir / "sessions" / "conv-canceled-handoff.json").read_text(encoding="utf-8"))
        self.assertEqual(c_sess["lastUserStepIndex"], 3)

        # Hook failure before packet build must NOT persist dedupe state (retry still succeeds).
        fail_payload = {
            "conversationId": "conv-fail-retry",
            "transcriptPath": str(tr),
            "workspacePaths": [str(self.workspace)],
            "lastUserInput": only_prompt,
            "invocationNum": 1,
            "initialNumSteps": 13,
        }
        buf1 = io.StringIO()
        with mock.patch("sys.stdin", io.StringIO(json.dumps(fail_payload))), mock.patch.object(
            core, "collect_workspace_context", side_effect=RuntimeError("boom")
        ), contextlib.redirect_stdout(buf1):
            hook.main()
        self.assertEqual(json.loads(buf1.getvalue()), {})
        self.assertFalse((self.data_dir / "sessions" / "conv-fail-retry.json").exists())

        buf2 = io.StringIO()
        with mock.patch("sys.stdin", io.StringIO(json.dumps(fail_payload))), contextlib.redirect_stdout(buf2):
            hook.main()
        self.assertIn("injectSteps", json.loads(buf2.getvalue()))

        # New user turn with the exact same prompt text -> cur_user_step advances -> MUST re-trigger
        _write_transcript(
            tr,
            [
                {"step_index": 184, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": wrapped_prompt},
                {"step_index": 196, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": wrapped_prompt},
            ],
        )
        out_new_turn = self._run_hook({**base_payload, "invocationNum": 1, "initialNumSteps": 197})
        self.assertIn("injectSteps", out_new_turn)

    def test_hook_dedupe_unflushed_transcript_race_and_backfill(self) -> None:
        tr = self.root / "transcript_race.jsonl"
        old_prompt = "++ old message"
        new_prompt = "++code check race condition"
        # At Inv 0, transcript still only contains the older turn (unflushed race)
        _write_transcript(
            tr,
            [
                {"step_index": 2, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": old_prompt},
            ],
        )
        payload0 = {
            "conversationId": "conv-race",
            "transcriptPath": str(tr),
            "workspacePaths": [str(self.workspace)],
            "lastUserInput": new_prompt,
            "invocationNum": 0,
            "initialNumSteps": 10,
        }
        out0 = self._run_hook(payload0)
        self.assertIn("injectSteps", out0)
        session_file = self.data_dir / "sessions" / "conv-race.json"
        self.assertEqual(json.loads(session_file.read_text(encoding="utf-8"))["lastUserStepIndex"], -1)

        # Before Inv 1, transcript flushes step 10 with the new prompt
        _write_transcript(
            tr,
            [
                {"step_index": 2, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": old_prompt},
                {"step_index": 10, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": new_prompt},
            ],
        )
        out1 = self._run_hook({**payload0, "invocationNum": 1, "initialNumSteps": 10})
        self.assertEqual(out1, {})
        # Step index should now be backfilled to 10
        self.assertEqual(json.loads(session_file.read_text(encoding="utf-8"))["lastUserStepIndex"], 10)

        # Subsequent exit_plan_mode handoff (inv=0, initialNumSteps=15) still dedupes via backfilled step 10
        _write_transcript(
            tr,
            [
                {"step_index": 2, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": old_prompt},
                {"step_index": 10, "type": "USER_INPUT", "source": "USER_EXPLICIT", "content": new_prompt},
                {"step_index": 11, "type": "PLANNER_RESPONSE", "status": "DONE", "tool_calls": [{"name": "exit_plan_mode"}]},
                {"step_index": 12, "type": "GENERIC", "status": "DONE"},
                {"step_index": 13, "type": "SYSTEM_MESSAGE", "status": "DONE", "content": "Plan mode is off:"},
            ],
        )
        out_handoff = self._run_hook({**payload0, "invocationNum": 0, "initialNumSteps": 15})
        self.assertEqual(out_handoff, {})

    # 3. Workspace Context Collection
    def test_workspace_context_git_secrets_apfs_and_cap(self) -> None:
        (self.workspace / "src").mkdir()
        (self.workspace / "src" / "app_runner.py").write_text("print('ok')\n", encoding="utf-8")
        (self.workspace / ".env").write_text("API_KEY=supersecret\n", encoding="utf-8")
        (self.workspace / "secret.key").write_text("private\n", encoding="utf-8")
        (self.workspace / "AGENTS.md").write_text("# Repo Rules\n- Keep diffs minimal.\n", encoding="utf-8")

        # Initialize git repo
        subprocess.run(["git", "init"], cwd=str(self.workspace), check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(self.workspace), check=True)
        subprocess.run(["git", "config", "user.name", "Tester"], cwd=str(self.workspace), check=True)
        subprocess.run(["git", "add", "src/app_runner.py", "AGENTS.md"], cwd=str(self.workspace), check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=str(self.workspace), check=True, capture_output=True)

        ctx = core.collect_workspace_context(
            [str(self.workspace)],
            "fix `app_runner.py` and check .env and secret.key",
            max_chars=4000,
        )
        self.assertTrue(ctx["git"]["branch"])
        self.assertIn("src/app_runner.py", ctx["mentioned_paths"])
        self.assertNotIn(".env", ctx["mentioned_paths"])
        self.assertNotIn("secret.key", ctx["mentioned_paths"])
        self.assertNotIn(".env", ctx["snapshot_text"])
        self.assertNotIn("secret.key", ctx["snapshot_text"])

        # APFS / inode dedupe: AGENTS.md must appear at most once in rules
        rule_files = [r["file"] for r in ctx["rules"]]
        self.assertEqual(len(rule_files), len(set(f.lower() for f in rule_files)))
        self.assertEqual(len(rule_files), 1)

        # Strict max_chars enforcement
        tight_ctx = core.collect_workspace_context([str(self.workspace)], "app_runner.py", max_chars=220)
        self.assertLessEqual(len(tight_ctx["snapshot_text"]), 220)

        # Track .env in git, enable color.status=always, then modify unstaged so `git status`
        # would emit ANSI codes under --short; --porcelain=v1 must still filter .env cleanly.
        subprocess.run(["git", "add", "-f", ".env"], cwd=str(self.workspace), check=True)
        subprocess.run(["git", "commit", "-m", "track env for filter test"], cwd=str(self.workspace), check=True, capture_output=True)
        subprocess.run(["git", "config", "color.status", "always"], cwd=str(self.workspace), check=True)
        (self.workspace / ".env").write_text("API_KEY=changed\n", encoding="utf-8")
        filtered_ctx = core.collect_workspace_context([str(self.workspace)], "check status", max_chars=4000)
        self.assertEqual(filtered_ctx["git"]["status_state"], "filtered_only")
        self.assertEqual(filtered_ctx["git"]["status_lines"], [])
        self.assertIn("Git Status: only ignored/secret files changed", filtered_ctx["snapshot_text"])
        self.assertNotIn(".env", filtered_ctx["snapshot_text"])

        # Unavailable git status must not report clean
        orig_run_git = core._run_git
        def fake_run_git(root: Path, args: list[str]) -> tuple[bool, str]:
            if args and args[0] == "status":
                return False, ""
            return orig_run_git(root, args)

        with mock.patch.object(core, "_run_git", side_effect=fake_run_git):
            unavail_ctx = core.collect_workspace_context([str(self.workspace)], "check status", max_chars=4000)
        self.assertEqual(unavail_ctx["git"]["status_state"], "unavailable")
        self.assertIn("Git Status: unavailable", unavail_ctx["snapshot_text"])

    def test_non_git_workspace_context(self) -> None:
        (self.workspace / "notes.md").write_text("# Notes\n", encoding="utf-8")
        ctx = core.collect_workspace_context([str(self.workspace)], "summarize notes.md", max_chars=4000)
        self.assertEqual(ctx["git"], {})
        self.assertIn("notes.md", ctx["mentioned_paths"])
        self.assertIn("notes.md", ctx["snapshot_text"])

    def test_scenario_inference_boundaries_and_false_positives(self) -> None:
        templates = core.load_all_templates()
        self.assertEqual(core.infer_scenario("测试一下", templates), "general")
        self.assertNotEqual(core.infer_scenario("排一下docker容器启动慢", templates), "doc")
        self.assertEqual(core.infer_scenario("explain how it works", templates), "general")
        self.assertEqual(core.infer_scenario("帮我把人类样本分类", templates), "general")
        self.assertEqual(core.infer_scenario("写个api接口并补单测", templates), "code")
        self.assertEqual(core.infer_scenario("write a python parser", templates), "code")

    def test_snapshot_preserves_rules_and_balances_large_subdirs_and_escapes_tags(self) -> None:
        alpha = self.workspace / "alpha"
        alpha.mkdir()
        for i in range(1, 16):
            (alpha / f"file_{i:02d}.py").write_text(f"x = {i}\n", encoding="utf-8")
        zeta = self.workspace / "zeta"
        zeta.mkdir()
        (zeta / "target_mod.py").write_text("ok = True\n", encoding="utf-8")
        (self.workspace / "AGENTS.md").write_text(
            "# Repo Rules\n- CRITICAL_RULE_KEEP_MINIMAL: never touch unrelated files.\n",
            encoding="utf-8",
        )

        ctx_full = core.collect_workspace_context([str(self.workspace)], "check file_14.py", max_chars=4000)
        self.assertIn("alpha/... (+7 more)", ctx_full["snapshot_text"])
        self.assertIn("zeta/target_mod.py", ctx_full["snapshot_text"])
        self.assertIn("alpha/file_14.py", ctx_full["mentioned_paths"])

        # Add extra top-level files so tree+rules exceed 650 chars, verifying rules survive while tree trims first.
        extra_dir = self.workspace / "beta"
        extra_dir.mkdir()
        for j in range(1, 10):
            (extra_dir / f"long_module_name_{j:02d}.py").write_text("pass\n", encoding="utf-8")

        ctx_budget = core.collect_workspace_context([str(self.workspace)], "check long_module_name_05.py", max_chars=650)
        self.assertLessEqual(len(ctx_budget["snapshot_text"]), 650)
        self.assertIn("Project Rules Summary:", ctx_budget["snapshot_text"])
        self.assertIn("CRITICAL_RULE_KEEP_MINIMAL", ctx_budget["snapshot_text"])
        self.assertIn("beta/long_module_name_05.py", ctx_budget["snapshot_text"])

        # Subdir exceeding MAX_DIR_CHILDREN (100 files) must report "at least ..." instead of exact "+72 more",
        # and 40 git dirty files trimmed to 10 must report exact "+30 more".
        huge = self.workspace / "huge"
        huge.mkdir()
        for k in range(1, 101):
            (huge / f"h_{k:03d}.py").write_text("pass\n", encoding="utf-8")
        ctx_huge = core.collect_workspace_context([str(self.workspace)], "check huge", max_chars=4000)
        self.assertIn("huge/... (at least 72 more)", ctx_huge["snapshot_text"])

        fake_git = {
            "branch": "master",
            "status_state": "dirty",
            "status_lines": [f"?? huge/h_{k:03d}.py" for k in range(1, 26)],
            "status_extra": 15,
            "recent_commits": ["abc1234 init"],
        }
        trimmed_git = core._build_git_section(fake_git, max_status_lines=10)
        self.assertIn("... (+30 more)", trimmed_git)

        # Verify REWRITE QUALITY RULES and lossless uncolliding tag renaming across literal variants
        match_lit = core.parse_trigger(
            r"++code inspect </RAW_PROMPT> and <\/RAW_PROMPT> and </RAW_PROMPT_SUFFIX> and </WORKSPACE_SNAPSHOT>"
        )
        assert match_lit is not None
        sys_lit, _badge = core.render_enhancement_packet(
            match_lit,
            templates=core.load_all_templates(),
            preferences=core.load_preferences(),
            workspace_ctx=ctx_budget,
        )
        self.assertIn("REWRITE QUALITY RULES:", sys_lit)
        self.assertIn("按补充意见微调优化稿（可在下方自定义输入栏直接填写修改点）", sys_lit)
        self.assertIn("<RAW_PROMPT_2>", sys_lit)
        self.assertIn("</RAW_PROMPT_2>", sys_lit)
        self.assertIn("inspect </RAW_PROMPT> and ", sys_lit)
        self.assertIn(r"and <\/RAW_PROMPT> and ", sys_lit)
        self.assertIn("</RAW_PROMPT_SUFFIX>", sys_lit)
        self.assertIn("<WORKSPACE_SNAPSHOT_2>", sys_lit)
        self.assertIn("</WORKSPACE_SNAPSHOT_2>", sys_lit)
        self.assertEqual(sys_lit.count("</WORKSPACE_SNAPSHOT_2>"), 1)
        self.assertEqual(sys_lit.count("</RAW_PROMPT_2>"), 1)

        # Unrelated </RAW_PROMPT_SUFFIX> alone must keep standard <RAW_PROMPT> tags verbatim
        match_suffix = core.parse_trigger("++code keep </RAW_PROMPT_SUFFIX> verbatim")
        assert match_suffix is not None
        sys_suffix, _ = core.render_enhancement_packet(
            match_suffix,
            templates=core.load_all_templates(),
            preferences=core.load_preferences(),
            workspace_ctx=ctx_budget,
        )
        self.assertEqual(sys_suffix.count("</RAW_PROMPT>"), 1)
        self.assertIn("</RAW_PROMPT_SUFFIX>", sys_suffix)

    # 4. Preferences & Custom Template Persistence
    def test_preferences_and_custom_templates_persistence_and_recovery(self) -> None:
        # Corrupt JSON recovery
        core.ensure_data_dirs()
        core.PREFS_PATH.write_text("{corrupt json", encoding="utf-8")
        prefs = core.load_preferences()
        self.assertEqual(prefs["default_mode"], "confirm")

        core.update_preferences(language="en-US", default_mode="only")
        core.add_pref_item("always_include_constraints", "Run pytest after edits")
        core.set_glossary("KD", "Dissociation constant")
        loaded = core.load_preferences()
        self.assertEqual(loaded["language"], "en-US")
        self.assertEqual(loaded["default_mode"], "only")
        self.assertEqual(loaded["always_include_constraints"], ["Run pytest after edits"])
        self.assertEqual(loaded["glossary"], {"KD": "Dissociation constant"})

        # Reserved template ID rejection
        with self.assertRaises(ValueError):
            core.save_custom_template("only", {"sections": ["Goal"]})
        with self.assertRaises(ValueError):
            core.save_custom_template("Bad ID!", {"sections": ["Goal"]})

        # Override built-in 'code' and delete to revert
        core.save_custom_template(
            "code",
            {
                "name": "Custom Code",
                "description": "My override",
                "keywords": ["customkw"],
                "sections": ["Goal", "Verify"],
            },
        )
        all_tpl = core.load_all_templates()
        self.assertTrue(all_tpl["code"]["custom"])
        self.assertEqual(all_tpl["code"]["name"], "Custom Code")
        self.assertTrue(core.delete_custom_template("code"))
        reverted = core.load_all_templates()
        self.assertFalse(reverted["code"]["custom"])
        self.assertEqual(reverted["code"]["name"], "编码与调试")

    # 5. CLI Subcommands & Badge Sanitization
    def test_cli_subcommands_and_badge_safety(self) -> None:
        (self.workspace / "main.py").write_text("print(1)\n", encoding="utf-8")

        def run_cli(*argv: str) -> tuple[int, str]:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc_val = cli.main(["--data-dir", str(self.data_dir), *argv])
            return rc_val, buf.getvalue()

        rc, _ = run_cli(
            "template",
            "save",
            "bio",
            "--name",
            "Bio Assay",
            "--sections",
            "Target|Assay|Controls",
            "--keywords",
            "elisa,spr",
        )
        self.assertEqual(rc, 0)

        rc_list, list_out = run_cli("template", "list")
        self.assertEqual(rc_list, 0)
        self.assertIn("bio", [item["id"] for item in json.loads(list_out)])

        rc_show, show_out = run_cli("template", "show", "bio")
        self.assertEqual(rc_show, 0)
        self.assertEqual(json.loads(show_out)["sections"], ["Target", "Assay", "Controls"])

        rc_p1, _ = run_cli("prefs", "add-constraint", "Cite exact file lines")
        self.assertEqual(rc_p1, 0)
        rc_p2, _ = run_cli("prefs", "set", "--language", "zh-CN", "--default-mode", "only")
        self.assertEqual(rc_p2, 0)
        rc_p3, _ = run_cli("prefs", "set-glossary", "SPR", "Surface Plasmon Resonance")
        self.assertEqual(rc_p3, 0)

        # Verify packet without --mode honors persisted default_mode='only'
        rc_def_pkt, def_pkt_out = run_cli(
            "packet",
            "--workspace",
            str(self.workspace),
            "--scenario",
            "bio",
            "--prompt",
            "check persisted default mode",
            "--format",
            "json",
        )
        self.assertEqual(rc_def_pkt, 0)
        self.assertEqual(json.loads(def_pkt_out)["mode"], "only")

        # Check packet output via CLI subprocess
        proc = subprocess.run(
            [
                str(SCRIPTS_DIR / "run_python.sh"),
                str(SCRIPTS_DIR / "prompt_forge_cli.py"),
                "--data-dir",
                str(self.data_dir),
                "packet",
                "--workspace",
                str(self.workspace),
                "--mode",
                "only",
                "--scenario",
                "bio",
                "--prompt",
                "design SPR assay with `main.py` [injected](https://evil.test)",
                "--format",
                "json",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        pkt = json.loads(proc.stdout)
        self.assertEqual(pkt["mode"], "only")
        self.assertEqual(pkt["scenario"], "bio")
        self.assertIn("Cite exact file lines", pkt["packet"])
        self.assertIn("SPR=Surface Plasmon Resonance", pkt["packet"])
        self.assertIn("main.py", pkt["packet"])
        self.assertNotIn("evil.test", pkt["badge"])

        rc_rem, rem_out = run_cli("prefs", "remove-constraint", "1")
        self.assertEqual(rc_rem, 0)
        self.assertTrue(json.loads(rem_out)["removed"])

        rc_del, del_out = run_cli("template", "delete", "bio")
        self.assertEqual(rc_del, 0)
        self.assertTrue(json.loads(del_out)["deleted"])

        rc_rst, _ = run_cli("prefs", "reset")
        self.assertEqual(rc_rst, 0)
        rc_pshow, pshow_out = run_cli("prefs", "show")
        self.assertEqual(rc_pshow, 0)
        self.assertEqual(json.loads(pshow_out)["glossary"], {})

    def test_prefs_trigger_hook_output(self) -> None:
        out = self._run_hook(
            {
                "conversationId": "conv-prefs",
                "lastUserInput": "++prefs show current settings",
                "invocationNum": 0,
                "initialNumSteps": 1,
            }
        )
        self.assertIn("injectSteps", out)
        msg_obj = out["injectSteps"][0]["systemMessage"]
        self.assertNotIn("metadata", msg_obj)
        self.assertIn("MODE=prefs", msg_obj["systemMessage"])
        self.assertIn("偏好与模板管理", msg_obj["systemMessage"])

    def test_at_mention_cjk_and_space_path_matching(self) -> None:
        (self.workspace / "张家界旅游2026").mkdir()
        (self.workspace / "张家界旅游2026" / "generate_pdf.py").write_text("print('pdf')\n", encoding="utf-8")
        (self.workspace / "audit file").mkdir()
        (self.workspace / "audit file" / "report.md").write_text("# Audit\n", encoding="utf-8")
        (self.workspace / "audit file" / "secret.key").write_text("secret\n", encoding="utf-8")
        (self.workspace / "other" / "file").mkdir(parents=True)
        (self.workspace / "other" / "file" / "report.md").write_text("# Other\n", encoding="utf-8")
        (self.workspace / ".env").write_text("TOKEN=123\n", encoding="utf-8")
        outside = self.root / "outside_secret.txt"
        outside.write_text("nope\n", encoding="utf-8")

        ctx = core.collect_workspace_context(
            [str(self.workspace)],
            "核对 @[张家界旅游2026/generate_pdf.py]、@[audit file/report.md]、@[.env]、@[audit file/secret.key]、@[../outside_secret.txt]",
            max_chars=4000,
        )
        self.assertIn("张家界旅游2026/generate_pdf.py", ctx["mentioned_paths"])
        self.assertIn("audit file/report.md", ctx["mentioned_paths"])
        self.assertNotIn("other/file/report.md", ctx["mentioned_paths"])
        self.assertNotIn(".env", ctx["mentioned_paths"])
        self.assertNotIn("audit file/secret.key", ctx["mentioned_paths"])
        self.assertNotIn("outside_secret.txt", ctx["mentioned_paths"])
        self.assertIn("张家界旅游2026/generate_pdf.py", ctx["snapshot_text"])
        self.assertIn("audit file/report.md", ctx["snapshot_text"])

        ctx_cjk_plain = core.collect_workspace_context(
            [str(self.workspace)],
            "修改 张家界旅游2026/generate_pdf.py，然后验证输出",
            max_chars=4000,
        )
        self.assertIn("张家界旅游2026/generate_pdf.py", ctx_cjk_plain["mentioned_paths"])

        # Absolute path inside workspace must resolve relative to workspace root (not lstrip("/"))
        (self.workspace / "plugins" / "next-step" / "scripts").mkdir(parents=True)
        abs_target = self.workspace / "plugins" / "next-step" / "scripts" / "next_step_core.py"
        abs_target.write_text("# core\n", encoding="utf-8")
        ext_dir = self.root / "external_repo" / "audit file"
        ext_dir.mkdir(parents=True)
        ext_report = ext_dir / "report.md"
        ext_report.write_text("# external\n", encoding="utf-8")
        ctx_abs = core.collect_workspace_context(
            [str(self.workspace)],
            f"查看 @[{abs_target}] 和 @[{ext_report}] 以及 @[/audit file/report.md]",
            max_chars=4000,
        )
        self.assertIn("plugins/next-step/scripts/next_step_core.py", ctx_abs["mentioned_paths"])
        self.assertIn("audit file/report.md", ctx_abs["mentioned_paths"])
        self.assertFalse(any("external_repo" in p for p in ctx_abs["mentioned_paths"]))

        # Leading dots on dotfiles (@[.gitignore]) must be preserved, while parent traversal (@[../docs/build_docx.js]) must not strip '..'
        (self.workspace / ".gitignore").write_text(".venv/\n", encoding="utf-8")
        (self.workspace / "docs").mkdir(exist_ok=True)
        (self.workspace / "docs" / "build_docx.js").write_text("// doc\n", encoding="utf-8")
        ext_docs = self.root / "docs"
        ext_docs.mkdir(exist_ok=True)
        (ext_docs / "build_docx.js").write_text("// outside\n", encoding="utf-8")

        ctx_dot = core.collect_workspace_context(
            [str(self.workspace)],
            "检查 @[.gitignore] 和 @[../docs/build_docx.js]",
            max_chars=4000,
        )
        self.assertIn(".gitignore", ctx_dot["mentioned_paths"])
        self.assertNotIn("docs/build_docx.js", ctx_dot["mentioned_paths"])

    def test_claude_rules_scoped_matching_and_exclusion(self) -> None:
        (self.workspace / "docs").mkdir()
        (self.workspace / "docs" / "build_docx.js").write_text("// doc builder\n", encoding="utf-8")
        (self.workspace / "张家界旅游2026").mkdir()
        (self.workspace / "张家界旅游2026" / "generate_pdf.py").write_text("print('ok')\n", encoding="utf-8")
        (self.workspace / "scripts").mkdir()
        (self.workspace / "scripts" / "app.py").write_text("print(1)\n", encoding="utf-8")

        # Long root AGENTS.md (>2000 chars) must not starve matching .claude/rules/*.md
        long_agents = ["# Root Rules"] + [f"- Top item {i}: keep code clean and concise." for i in range(45)]
        (self.workspace / "AGENTS.md").write_text("\n".join(long_agents) + "\n", encoding="utf-8")

        rules_dir = self.workspace / ".claude" / "rules"
        rules_dir.mkdir(parents=True)
        (rules_dir / "docs-build.md").write_text(
            '---\npaths:\n  - "docs/**"\n---\n# Docs Rule\n- Never edit generated .docx directly.\n',
            encoding="utf-8",
        )
        (rules_dir / "travel.md").write_text(
            '---\npaths:\n  - "张家界旅游2026/**"\n---\n# Travel Rule\n- Keep generate_pdf.py single-file.\n',
            encoding="utf-8",
        )
        (rules_dir / "always-on.md").write_text(
            "# Always Rule\n- Always verify outputs.\n",
            encoding="utf-8",
        )
        (rules_dir / "py-only.md").write_text(
            '---\npaths: ["scripts/*.py"]\n---\n# Scripts Rule\n- Run unittest.\n',
            encoding="utf-8",
        )
        (rules_dir / "doc-boundary.md").write_text(
            '---\npaths:\n  - "doc/**"\n---\n# Doc Boundary\n- Should not match docker.\n',
            encoding="utf-8",
        )
        (rules_dir / "secret.md").write_text("# Secret Rule\n- Hidden.\n", encoding="utf-8")
        try:
            (rules_dir / "symlink_rule.md").symlink_to(rules_dir / "always-on.md")
        except OSError:
            pass

        ctx_docs_file = core.collect_workspace_context([str(self.workspace)], "修改 docs/build_docx.js", max_chars=4000)
        files_docs = [r["file"] for r in ctx_docs_file["rules"]]
        self.assertIn("AGENTS.md", files_docs)
        self.assertIn(".claude/rules/docs-build.md", files_docs)
        self.assertIn(".claude/rules/always-on.md", files_docs)
        self.assertNotIn(".claude/rules/travel.md", files_docs)
        self.assertNotIn(".claude/rules/py-only.md", files_docs)
        self.assertNotIn(".claude/rules/secret.md", files_docs)
        self.assertNotIn(".claude/rules/symlink_rule.md", files_docs)
        self.assertLessEqual(
            sum(len(r["summary"]) + len(r["file"]) + 8 for r in ctx_docs_file["rules"]),
            core.MAX_RULES_CHARS,
        )

        # Directory prefix mention without concrete filename
        ctx_docs_dir = core.collect_workspace_context([str(self.workspace)], "更新 docs/ 里的说明", max_chars=4000)
        self.assertIn(".claude/rules/docs-build.md", [r["file"] for r in ctx_docs_dir["rules"]])

        # CJK @mention path triggers travel.md and excludes docs-build.md
        ctx_travel = core.collect_workspace_context(
            [str(self.workspace)],
            "检查 @[张家界旅游2026/generate_pdf.py]",
            max_chars=4000,
        )
        files_travel = [r["file"] for r in ctx_travel["rules"]]
        self.assertIn(".claude/rules/travel.md", files_travel)
        self.assertIn(".claude/rules/always-on.md", files_travel)
        self.assertNotIn(".claude/rules/docs-build.md", files_travel)

        # Word boundary prevents "docker" from triggering "doc/**"
        ctx_docker = core.collect_workspace_context([str(self.workspace)], "排一下docker容器启动慢", max_chars=4000)
        self.assertNotIn(".claude/rules/doc-boundary.md", [r["file"] for r in ctx_docker["rules"]])

        # Segment-aware glob: "src/*.py" matches "src/app.py", NOT "src/nested/app.py";
        # "src/**/*.py" matches newly created uncreated "@[src/new_file.py]";
        # "docs/**" does NOT match "vendor/docs/a.md".
        (self.workspace / "src" / "nested").mkdir(parents=True)
        (self.workspace / "src" / "app.py").write_text("x = 1\n", encoding="utf-8")
        (self.workspace / "src" / "nested" / "app.py").write_text("x = 2\n", encoding="utf-8")
        (self.workspace / "vendor" / "docs").mkdir(parents=True)
        (self.workspace / "vendor" / "docs" / "a.md").write_text("# Vendor\n", encoding="utf-8")
        (rules_dir / "src-top.md").write_text(
            '---\npaths:\n  - "src/*.py"\n---\n# Src Top Only\n- Keep top-level src clean.\n',
            encoding="utf-8",
        )
        (rules_dir / "src-deep.md").write_text(
            '---\npaths:\n  - "src/**/*.py"\n---\n# Src Deep\n- Add type hints.\n',
            encoding="utf-8",
        )

        ctx_nested = core.collect_workspace_context([str(self.workspace)], "edit src/nested/app.py", max_chars=4000)
        files_nested = [r["file"] for r in ctx_nested["rules"]]
        self.assertNotIn(".claude/rules/src-top.md", files_nested)
        self.assertIn(".claude/rules/src-deep.md", files_nested)

        ctx_new_file = core.collect_workspace_context([str(self.workspace)], "create @[src/new_file.py]", max_chars=4000)
        files_new = [r["file"] for r in ctx_new_file["rules"]]
        self.assertIn(".claude/rules/src-top.md", files_new)
        self.assertIn(".claude/rules/src-deep.md", files_new)

        ctx_vendor = core.collect_workspace_context(
            [str(self.workspace)],
            "update vendor/docs/a.md and vendor/docs/ notes",
            max_chars=4000,
        )
        self.assertNotIn(".claude/rules/docs-build.md", [r["file"] for r in ctx_vendor["rules"]])

        # Uncreated file via absolute path inside workspace matches "src/**/*.py", while external path does not
        abs_new = self.workspace / "src" / "brand_new.py"
        ctx_abs_new = core.collect_workspace_context([str(self.workspace)], f"create @[{abs_new}]", max_chars=4000)
        self.assertIn(".claude/rules/src-deep.md", [r["file"] for r in ctx_abs_new["rules"]])
        ext_new = self.root / "other_ws" / "src" / "ext_new.py"
        ctx_ext_new = core.collect_workspace_context([str(self.workspace)], f"create @[{ext_new}]", max_chars=4000)
        self.assertNotIn(".claude/rules/src-deep.md", [r["file"] for r in ctx_ext_new["rules"]])

        # Explicit file reference resolving outside docs/ (e.g. @[docs/../agents.md] or bare docs/../agents.md) must NOT trigger docs/**
        (self.workspace / "agents.md").write_text("# Root\n- Rule.\n", encoding="utf-8")
        ctx_dotdot_at = core.collect_workspace_context(
            [str(self.workspace)],
            "检查 @[docs/../agents.md]",
            max_chars=4000,
        )
        self.assertNotIn(".claude/rules/docs-build.md", [r["file"] for r in ctx_dotdot_at["rules"]])
        ctx_dotdot_bare = core.collect_workspace_context(
            [str(self.workspace)],
            "检查 docs/../agents.md",
            max_chars=4000,
        )
        self.assertNotIn(".claude/rules/docs-build.md", [r["file"] for r in ctx_dotdot_bare["rules"]])

    def test_tail_redlines_preserved_in_long_agents_md_and_compact_budget(self) -> None:
        lines = ["# Repo Guidelines", "- Primary rule: keep replies concise."]
        for i in range(1, 40):
            lines.append(f"- Background guideline {i}: follow standard conventions across modules in this workspace.")
        lines.extend(
            [
                "## Safety Boundaries",
                "- Prefer YAGNI and minimal-diff changes.",
                "**NEVER:**",
                "- Modify `.env`, lockfiles, or CI secrets without explicit approval",
                "- Delete user data or reset databases without confirmation",
                "- Force-push to master or rewrite pushed history",
                "**ALWAYS:**",
                "- Show diff before committing",
                "- Ask before making destructive or irreversible operations",
                "- 严禁直接修改生成产物",
            ]
        )
        long_text = "\n".join(lines)
        self.assertGreater(len(long_text), 2500)

        summary = core._summarize_rule_text(long_text, 1200)
        self.assertLessEqual(len(summary), 1200)
        self.assertIn("# Repo Guidelines", summary)
        self.assertIn("**NEVER:**", summary)
        self.assertIn("Delete user data or reset databases", summary)
        self.assertIn("Force-push to master", summary)
        self.assertIn("**ALWAYS:**", summary)
        self.assertIn("Show diff before committing", summary)
        self.assertIn("严禁直接修改生成产物", summary)
        self.assertLess(summary.index("# Repo Guidelines"), summary.index("**NEVER:**"))
        self.assertLess(summary.index("**NEVER:**"), summary.index("Force-push to master"))
        self.assertLess(summary.index("Force-push to master"), summary.index("Show diff before committing"))
        self.assertLess(summary.index("Show diff before committing"), summary.index("严禁直接修改生成产物"))

        # In tight max_chars=650 with many tree files, Project Rules Summary compacts to <=240 and survives
        (self.workspace / "AGENTS.md").write_text(long_text + "\n", encoding="utf-8")
        pkg = self.workspace / "modules"
        pkg.mkdir()
        for j in range(1, 20):
            (pkg / f"module_long_name_{j:02d}.py").write_text("pass\n", encoding="utf-8")

        ctx_650 = core.collect_workspace_context([str(self.workspace)], "check module_long_name_01.py", max_chars=650)
        self.assertLessEqual(len(ctx_650["snapshot_text"]), 650)
        self.assertIn("Project Rules Summary:", ctx_650["snapshot_text"])

        ctx_220 = core.collect_workspace_context([str(self.workspace)], "check module_long_name_01.py", max_chars=220)
        self.assertLessEqual(len(ctx_220["snapshot_text"]), 220)

    def test_multi_rule_repo_budget_preserves_all_never_always_bullets(self) -> None:
        # Replicate real ~2.5KB AGENTS.md + 2 matched .claude/rules/*.md files loaded simultaneously
        agents_lines = [
            "# AGENTS.md",
            "本仓库所有 Agent 共用准则。",
            "**Tradeoff**：谨慎优先于速度；明显拼写错误直接执行。",
            "## 1. 交互与行文（最高优先级）",
            "- **称呼规则**：每次回复必须称呼用户为「绳匠」。无例外。",
            "- **语言规则**：无论用户使用何种语言，**必须始终使用中文回复**。无例外。",
            "- **反报告腔**：直出结论，随后附具体文件/行号/数据证据。",
        ]
        for i in range(1, 24):
            agents_lines.append(f"- Context rule {i}: explain subsystem conventions and read-only constraints in detail.")
        agents_lines.extend(
            [
                "## 4. 不可逆操作与 Git 安全红线",
                "默认分支：`master`。敏感信息绝不入库。",
                "**NEVER:**",
                "- Modify `.env`, lockfiles, or CI secrets without explicit approval",
                "- Auto-fix validation errors without asking user first",
                "- Delete user data or reset databases without confirmation",
                "- Force-push to `master` / `main` branch or rewrite pushed Git history",
                "**ALWAYS:**",
                "- Show diff before committing",
                "- Ask before making destructive, paid, public-release, privilege-escalation, or irreversible operations",
                "**上下文压缩保留项**：长会话执行 `/compact` 时，必须保留已拍板决策、验证状态及本节安全红线。",
            ]
        )
        agents_text = "\n".join(agents_lines) + "\n"
        self.assertGreater(len(agents_text), 2400)
        (self.workspace / "AGENTS.md").write_text(agents_text, encoding="utf-8")

        (self.workspace / "docs").mkdir()
        (self.workspace / "docs" / "build_docx.js").write_text("// doc\n", encoding="utf-8")
        (self.workspace / "张家界旅游2026").mkdir()
        (self.workspace / "张家界旅游2026" / "generate_pdf.py").write_text("print(1)\n", encoding="utf-8")

        rules_dir = self.workspace / ".claude" / "rules"
        rules_dir.mkdir(parents=True)
        (rules_dir / "docs-build.md").write_text(
            '---\npaths:\n  - "docs/**"\n---\n# docs/ Word 文档构建规则\n- 改源文件/脚本，不改生成产物。\n- "生成文档" -> 跑 `node docs/build_docx.js`。\n',
            encoding="utf-8",
        )
        (rules_dir / "travel-2026.md").write_text(
            '---\npaths:\n  - "张家界旅游2026/**"\n---\n# 张家界旅游2026 子项目规则\n- Python 脚本保持单文件、直接跑。\n- 改源文件/脚本，不改产物。\n',
            encoding="utf-8",
        )

        ctx = core.collect_workspace_context(
            [str(self.workspace)],
            "更新 docs/ 和 张家界旅游2026/",
            max_chars=4000,
        )
        rule_files = [r["file"] for r in ctx["rules"]]
        self.assertEqual(
            rule_files,
            ["AGENTS.md", ".claude/rules/docs-build.md", ".claude/rules/travel-2026.md"],
        )
        snap = ctx["snapshot_text"]
        self.assertIn("Modify `.env`, lockfiles, or CI secrets without explicit approval", snap)
        self.assertIn("Auto-fix validation errors without asking user first", snap)
        self.assertIn("Delete user data or reset databases without confirmation", snap)
        self.assertIn("Force-push to `master` / `main` branch or rewrite pushed Git history", snap)
        self.assertIn("Show diff before committing", snap)
        self.assertIn("Ask before making destructive", snap)

    def test_git_head_direct_read_and_concurrent_git_calls(self) -> None:
        (self.workspace / "main.py").write_text("print(1)\n", encoding="utf-8")
        subprocess.run(["git", "init"], cwd=str(self.workspace), check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(self.workspace), check=True)
        subprocess.run(["git", "config", "user.name", "Tester"], cwd=str(self.workspace), check=True)
        subprocess.run(["git", "checkout", "-b", "feat/fast-head"], cwd=str(self.workspace), check=True, capture_output=True)
        subprocess.run(["git", "add", "main.py"], cwd=str(self.workspace), check=True)
        subprocess.run(["git", "commit", "-m", "initial commit"], cwd=str(self.workspace), check=True, capture_output=True)

        calls: list[str] = []
        orig_run_git = core._run_git

        def spy_run_git(root: Path, args: list[str]) -> tuple[bool, str]:
            if args:
                calls.append(args[0])
            return orig_run_git(root, args)

        with mock.patch.object(core, "_run_git", side_effect=spy_run_git):
            info = core._collect_git_info(self.workspace)

        self.assertEqual(info["branch"], "feat/fast-head")
        self.assertEqual(info["status_state"], "clean")
        self.assertEqual(len(info["recent_commits"]), 1)
        self.assertEqual(set(calls), {"status", "log"})

        # _run_git must invoke git with --no-optional-locks so .git/index stays read-only
        sub_cmds: list[list[str]] = []
        orig_sub_run = subprocess.run

        def spy_sub_run(cmd: list[str], *args: object, **kwargs: object) -> subprocess.CompletedProcess:
            sub_cmds.append(list(cmd))
            return orig_sub_run(cmd, *args, **kwargs)

        with mock.patch("prompt_forge_core.subprocess.run", side_effect=spy_sub_run):
            ok_st, _ = core._run_git(self.workspace, ["status", "--porcelain=v1"])
        self.assertTrue(ok_st)
        self.assertEqual(sub_cmds[0][:4], ["git", "-C", str(self.workspace), "--no-optional-locks"])

        # Detached 40-hex HEAD is also read directly without invoking git rev-parse
        subprocess.run(["git", "checkout", "--detach", "HEAD"], cwd=str(self.workspace), check=True, capture_output=True)
        calls.clear()
        with mock.patch.object(core, "_run_git", side_effect=spy_run_git):
            detached_branch = core._read_git_head_branch(self.workspace)
        self.assertEqual(detached_branch, "HEAD")
        self.assertEqual(calls, [])

        # Worktree / gitfile fallback calls git rev-parse
        wt_dir = self.root / "wt"
        subprocess.run(["git", "worktree", "add", "-b", "wt-branch", str(wt_dir)], cwd=str(self.workspace), check=True, capture_output=True)
        calls.clear()
        with mock.patch.object(core, "_run_git", side_effect=spy_run_git):
            wt_branch = core._read_git_head_branch(wt_dir)
        self.assertEqual(wt_branch, "wt-branch")
        self.assertIn("rev-parse", calls)

    def test_expanded_scenario_keywords(self) -> None:
        templates = core.load_all_templates()
        self.assertEqual(core.infer_scenario("优化 next-step 插件性能", templates), "code")
        self.assertEqual(core.infer_scenario("optimize typescript ui component perf", templates), "code")
        self.assertEqual(core.infer_scenario("做一次技术评审并输出对比方案演进路线", templates), "plan")

        match = core.parse_trigger("++ 优化 next-step 插件性能")
        assert match is not None
        sys_msg, badge = core.render_enhancement_packet(match, templates=templates, preferences=core.load_preferences())
        self.assertIn("SCENARIO=code (inferred)", sys_msg)
        self.assertIn("[code]", badge)

    def test_raw_stdin_fast_path_and_session_prune_on_new_only(self) -> None:
        buf = io.StringIO()
        with mock.patch("sys.stdin", io.StringIO('{"conversationId": "c1", "lastUserInput": "explain code"}')), mock.patch(
            "json.loads", side_effect=AssertionError("json.loads must be skipped on non-++ raw stdin")
        ), contextlib.redirect_stdout(buf):
            hook.main()
        self.assertEqual(json.loads(buf.getvalue()), {})

        # Fresh python process confirms prompt_forge_core is not imported on fast-path
        proc = subprocess.run(
            [
                str(SCRIPTS_DIR / "run_python.sh"),
                "-c",
                (
                    "import io, sys; "
                    f"sys.path.insert(0, {str(SCRIPTS_DIR)!r}); "
                    "import prompt_forge_hook as h; "
                    "sys.stdin = io.StringIO('{\"lastUserInput\":\"hello\"}'); "
                    "h.main(); "
                    "assert 'prompt_forge_core' not in sys.modules"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        self.assertEqual(proc.stdout.strip(), "{}")

        # ASCII-escaped fullwidth & '+' triggers (\\uff0b\\uff0b, \\u002b\\u002b, +\\u002b) pass pre-filter and trigger
        for raw_json_line in (
            json.dumps({"conversationId": "conv-fw-esc", "lastUserInput": "＋＋ 查看偏好", "workspacePaths": [str(self.workspace)]}, ensure_ascii=True),
            '{"conversationId":"conv-u002b-both","lastUserInput":"\\u002b\\u002bonly review","workspacePaths":' + json.dumps([str(self.workspace)]) + "}",
            '{"conversationId":"conv-u002b-mixed","lastUserInput":"+\\u002bonly review","workspacePaths":' + json.dumps([str(self.workspace)]) + "}",
        ):
            esc_out = subprocess.run(
                [str(SCRIPTS_DIR / "run_python.sh"), str(SCRIPTS_DIR / "prompt_forge_hook.py")],
                input=raw_json_line,
                capture_output=True,
                text=True,
                env={**os.environ, "PLUGIN_DATA": str(self.data_dir)},
                cwd=str(PLUGIN_ROOT),
                timeout=5,
                check=True,
            )
            self.assertIn("injectSteps", json.loads(esc_out.stdout.strip()))

        # Non-prefix C\\u002b\\u002b skips prompt_forge_core import after JSON decode
        cpp_proc = subprocess.run(
            [
                str(SCRIPTS_DIR / "run_python.sh"),
                "-c",
                (
                    "import io, sys; "
                    f"sys.path.insert(0, {str(SCRIPTS_DIR)!r}); "
                    "import prompt_forge_hook as h; "
                    "sys.stdin = io.StringIO('{\"lastUserInput\":\"C\\\\u002b\\\\u002b code\"}'); "
                    "h.main(); "
                    "assert 'prompt_forge_core' not in sys.modules"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        self.assertEqual(cpp_proc.stdout.strip(), "{}")

        # save_session calls _prune_sessions only when creating a new session file
        with mock.patch.object(core, "_prune_sessions") as mock_prune:
            core.save_session("conv-prune-check", {"lastUserStepIndex": -1})
            self.assertEqual(mock_prune.call_count, 1)
            core.save_session("conv-prune-check", {"lastUserStepIndex": 5})
            self.assertEqual(mock_prune.call_count, 1)

    # 6. Execution Tool Routing
    def test_execution_tool_recommendation_heuristics_and_guards(self) -> None:
        # Simple / localized / general tasks -> none (YAGNI default)
        for simple in (
            "",
            "修复 auth.py 第 42 行的空指针报错",
            "更新 README.md 中的安装说明",
            "解释一下这个函数的参数含义",
            "解释 JSON Schema 是什么",
            "仅把 README 中的 pipeline 改为大写",
            "解释 fan-out 是什么",
            "写一段 map-reduce 概念说明",
            "对比两个蛋白结合筛选策略",
            "修复 CI 流水线报错",
            "validate JSON schema for plugin.json",
            "use boost::asio to fix socket timeout",
            "use swarm plot in seaborn",
            "批量扫描 2 个文件并汇总",
            "批量处理三个文件",
            "遍历所有 3 个文件并汇总",
            "批量处理 a.py、b.py 和 c.py",
            "batch process three files and summarize",
            "逐个扫描 2 个文件并汇总",
            "在 README 中添加 5 个示例",
        ):
            rec = core.recommend_execution_tool(simple)
            self.assertEqual(rec["tool"], "none", f"Expected none for {simple!r}, got {rec}")
            self.assertEqual(rec["execution_scope"], "in_session")

        # Batch fan-out -> /run-workflow (in-session)
        for wf_prompt in (
            "遍历 20 个模块文件提取接口签名并汇总成 JSON 表",
            "批量遍历 10 个文件并按 JSON Schema 汇总",
            "用流水线批量处理 8 篇文献并汇总",
            "遍历所有十三个文件并汇总",
            "批量处理二十四篇文献并汇总",
        ):
            rec_wf = core.recommend_execution_tool(wf_prompt)
            self.assertEqual(rec_wf["tool"], "/run-workflow")
            self.assertEqual(rec_wf["execution_scope"], "in_session")
            self.assertEqual(rec_wf["confidence"], "high")

        # Deep concurrency / root cause -> /boost (host_slash)
        rec_boost = core.recommend_execution_tool("排查偶发死锁与竞态条件根因并做严密验证")
        self.assertEqual(rec_boost["tool"], "/boost")
        self.assertEqual(rec_boost["execution_scope"], "host_slash")
        self.assertEqual(rec_boost["confidence"], "high")

        # Parallel multi-module + verifier -> /swarm (host_slash)
        rec_swarm = core.recommend_execution_tool(
            "将 parser、lexer、codegen 三个模块并行重构并配 verifier 交叉验证"
        )
        self.assertEqual(rec_swarm["tool"], "/swarm")
        self.assertEqual(rec_swarm["execution_scope"], "host_slash")
        self.assertEqual(rec_swarm["confidence"], "high")

        # Multi-file workspace_ctx paths (>=3) + parallel cue + verifier -> /swarm
        rec_swarm_ctx = core.recommend_execution_tool(
            "并行重构这几个文件并做独立复核",
            workspace_ctx={"mentioned_paths": ["a.py", "b.py", "c.py"]},
        )
        self.assertEqual(rec_swarm_ctx["tool"], "/swarm")

        # End-to-end 0->1 project -> /teamwork (host_slash)
        rec_team = core.recommend_execution_tool("从零搭建完整全栈实验管理平台并端到端交付")
        self.assertEqual(rec_team["tool"], "/teamwork")
        self.assertEqual(rec_team["execution_scope"], "host_slash")

        # Meta-Q&A comparing tools -> none
        rec_meta = core.recommend_execution_tool("对比 /swarm、/teamwork、/run-workflow、/boost 有什么区别")
        self.assertEqual(rec_meta["tool"], "none")
        self.assertIn("meta_qa", rec_meta["signals"])

        rec_howto = core.recommend_execution_tool("怎么用 /swarm 做并行？")
        self.assertEqual(rec_howto["tool"], "none")
        self.assertIn("meta_qa", rec_howto["signals"])

        rec_howto_trailing = core.recommend_execution_tool("/swarm 怎么用？")
        self.assertEqual(rec_howto_trailing["tool"], "none")
        self.assertIn("meta_qa", rec_howto_trailing["signals"])

        for q_enable in ("/run-workflow 怎么启用？", "/swarm 怎么启用？", "How to use /swarm?"):
            rec_enable = core.recommend_execution_tool(q_enable)
            self.assertEqual(rec_enable["tool"], "none")
            self.assertIn("meta_qa", rec_enable["signals"])

        # Explicit 2-file workspace_ctx list with 逐一 must NOT trigger /run-workflow
        rec_two_ctx = core.recommend_execution_tool(
            "逐一检查 @[README.md] 和 @[plugin.json]",
            workspace_ctx={"mentioned_paths": ["README.md", "plugin.json"]},
        )
        self.assertEqual(rec_two_ctx["tool"], "none")
        rec_two_all_ctx = core.recommend_execution_tool(
            "遍历所有列出的文件（仅 @[README.md] 和 @[plugin.json]）并汇总",
            workspace_ctx={"mentioned_paths": ["README.md", "plugin.json"]},
        )
        self.assertEqual(rec_two_all_ctx["tool"], "none")
        rec_dir_to_single_out = core.recommend_execution_tool(
            "遍历目录下所有文件并汇总到 report.md",
            workspace_ctx={"mentioned_paths": ["report.md"]},
        )
        self.assertEqual(rec_dir_to_single_out["tool"], "/run-workflow")
        rec_all_with_out = core.recommend_execution_tool(
            "遍历所有插件文件，逐个计算 SHA-256 并汇总为 JSON，结果写入 @[checksums.json]",
            workspace_ctx={"mentioned_paths": []},
        )
        self.assertEqual(rec_all_with_out["tool"], "/run-workflow")
        rec_spec_all = core.recommend_execution_tool(
            "按照 @[README.md] 中的规范，遍历所有插件文件，逐个计算 SHA-256 并汇总为 JSON",
            workspace_ctx={"mentioned_paths": ["README.md"]},
        )
        self.assertEqual(rec_spec_all["tool"], "/run-workflow")
        rec_spec_all_out = core.recommend_execution_tool(
            "按照 @[README.md] 中的规范，遍历所有插件文件，逐个计算 SHA-256 并汇总为 JSON，结果写入 @[checksums.json]",
            workspace_ctx={"mentioned_paths": ["README.md"]},
        )
        self.assertEqual(rec_spec_all_out["tool"], "/run-workflow")

        # Per-tool negation & global opt-out guards (including CJK-adjacent /swarm并行 and backticks)
        rec_neg = core.recommend_execution_tool("不用 /swarm，分三个模块并行重构并交叉验证")
        self.assertNotEqual(rec_neg["tool"], "/swarm")
        rec_neg_cjk = core.recommend_execution_tool("不要用/swarm并行修改三个模块")
        self.assertEqual(rec_neg_cjk["tool"], "none")
        rec_neg_cjk2 = core.recommend_execution_tool("不使用swarm同时修改三个模块并交叉验证")
        self.assertEqual(rec_neg_cjk2["tool"], "none")
        rec_neg_please_dont = core.recommend_execution_tool("请勿使用 /swarm，并行修改三个模块")
        self.assertEqual(rec_neg_please_dont["tool"], "none")
        rec_neg_backtick = core.recommend_execution_tool("请勿使用 `/swarm`，并行修改三个模块")
        self.assertEqual(rec_neg_backtick["tool"], "none")
        rec_neg_enable = core.recommend_execution_tool("不要启用 /swarm，并行修改三个模块")
        self.assertEqual(rec_neg_enable["tool"], "none")
        rec_neg_polite = core.recommend_execution_tool("请不要再启用 /swarm，并行修改三个模块")
        self.assertEqual(rec_neg_polite["tool"], "none")
        rec_optout = core.recommend_execution_tool("不使用任何编排工具，直接修 parser")
        self.assertEqual(rec_optout["tool"], "none")
        self.assertIn("explicit_opt_out", rec_optout["signals"])
        rec_optout_please = core.recommend_execution_tool("请勿使用任何编排工具，并行修改三个模块")
        self.assertEqual(rec_optout_please["tool"], "none")
        self.assertIn("explicit_opt_out", rec_optout_please["signals"])

        # Explicit single-tool adoption vs comparative multi-tool fallthrough
        rec_explicit = core.recommend_execution_tool("用 /boost 排查这个缓存问题")
        self.assertEqual(rec_explicit["tool"], "/boost")
        self.assertIn("explicit_request", rec_explicit["signals"])

        rec_explicit_diff = core.recommend_execution_tool("用 /boost 深入分析两种算法的区别")
        self.assertEqual(rec_explicit_diff["tool"], "/boost")
        self.assertIn("explicit_request", rec_explicit_diff["signals"])

        rec_explicit_xgb = core.recommend_execution_tool("用 /boost 分析 XGBoost 的优势与性能瓶颈")
        self.assertEqual(rec_explicit_xgb["tool"], "/boost")
        self.assertIn("explicit_request", rec_explicit_xgb["signals"])

        rec_meta_pair = core.recommend_execution_tool("用 /boost 和 /swarm 有什么区别")
        self.assertEqual(rec_meta_pair["tool"], "none")
        self.assertIn("meta_qa", rec_meta_pair["signals"])

        rec_cmp = core.recommend_execution_tool("看看适合 /swarm 还是 /run-workflow：批量扫描 30 个文件汇总")
        self.assertEqual(rec_cmp["tool"], "/run-workflow")

    def test_tool_routing_preferences_cli_and_packet_modes(self) -> None:
        (self.workspace / "main.py").write_text("print(1)\n", encoding="utf-8")

        def run_cli(*argv: str) -> tuple[int, str]:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc_val = cli.main(["--data-dir", str(self.data_dir), *argv])
            return rc_val, buf.getvalue()

        # Backward compatibility: legacy preferences.json missing tool_routing defaults to auto
        core.ensure_data_dirs()
        core.PREFS_PATH.write_text(
            json.dumps({"version": 1, "language": "zh-CN", "default_mode": "confirm"}),
            encoding="utf-8",
        )
        self.assertEqual(core.load_preferences()["tool_routing"], "auto")

        # prefs set --tool-routing off persists and packet omits routing block
        rc_off, off_out = run_cli("prefs", "set", "--tool-routing", "off")
        self.assertEqual(rc_off, 0)
        self.assertEqual(json.loads(off_out)["tool_routing"], "off")

        rc_pkt_off, pkt_off_json = run_cli(
            "packet",
            "--workspace",
            str(self.workspace),
            "--prompt",
            "遍历 20 个模块文件并汇总",
            "--format",
            "json",
        )
        self.assertEqual(rc_pkt_off, 0)
        pkt_off = json.loads(pkt_off_json)
        self.assertEqual(pkt_off["tool_routing"], "off")
        self.assertIsNone(pkt_off["tool_recommendation"])
        self.assertNotIn("EXECUTION TOOL ROUTING", pkt_off["packet"])
        self.assertIn("按补充意见微调优化稿（可在下方自定义输入栏直接填写修改点）", pkt_off["packet"])

        # CLI --tool-routing auto override restores routing block and JSON recommendation
        rc_pkt_auto, pkt_auto_json = run_cli(
            "packet",
            "--workspace",
            str(self.workspace),
            "--tool-routing",
            "auto",
            "--prompt",
            "遍历 20 个模块文件并汇总",
            "--format",
            "json",
        )
        self.assertEqual(rc_pkt_auto, 0)
        pkt_auto = json.loads(pkt_auto_json)
        self.assertEqual(pkt_auto["tool_routing"], "auto")
        self.assertEqual(pkt_auto["tool_recommendation"]["tool"], "/run-workflow")
        self.assertIn("EXECUTION TOOL ROUTING", pkt_auto["packet"])
        self.assertIn("Heuristic hint (overridable): tool=/run-workflow", pkt_auto["packet"])

        # render_enhancement_packet in auto mode keeps badge unchanged, preserves edit option string,
        # and documents confirm branching for none, /run-workflow, and host slash commands
        match = core.parse_trigger("++code 排查偶发死锁与竞态条件根因并做严密验证")
        assert match is not None
        sys_msg, badge = core.render_enhancement_packet(
            match,
            templates=core.load_all_templates(),
            preferences={**core.load_preferences(), "tool_routing": "auto"},
            workspace_ctx={"workspace_name": "repo", "snapshot_text": "Workspace: repo"},
        )
        self.assertNotIn("/boost", badge)
        self.assertIn("EXECUTION TOOL ROUTING", sys_msg)
        self.assertIn("tool=/boost", sys_msg)
        self.assertIn("保留带 /<cmd> 前缀的优化稿并结束本轮", sys_msg)
        self.assertIn("直接在当前会话调用 run_workflow 按优化稿执行", sys_msg)
        self.assertIn("按补充意见微调优化稿（可在下方自定义输入栏直接填写修改点）", sys_msg)


if __name__ == "__main__":
    unittest.main()
