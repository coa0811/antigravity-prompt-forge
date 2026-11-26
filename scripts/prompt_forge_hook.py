#!/usr/bin/env python3
"""Antigravity PreInvocation hook for Prompt Forge (explicit ++ / ＋＋ prefix trigger)."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

TRANSCRIPT_INITIAL_TAIL_BYTES = 131072
_HOST_ENVELOPE_TAGS = ("ADDITIONAL_METADATA", "USER_SETTINGS_CHANGE", "CONTEXT_SUMMARY")


def _resolve_transcript_path(payload: dict[str, Any]) -> Path | None:
    from pathlib import Path

    raw_tr = str(payload.get("transcriptPath") or "").strip()
    if raw_tr:
        p = Path(raw_tr).expanduser()
        if p.is_file():
            return p
    cid = str(payload.get("conversationId") or "").strip()
    if cid:
        for app_dir in ("antigravity", "antigravity-cli", "antigravity-ide"):
            cand = (
                Path.home()
                / ".gemini"
                / app_dir
                / "brain"
                / cid
                / ".system_generated"
                / "logs"
                / "transcript.jsonl"
            )
            if cand.is_file():
                return cand
    return None


def _load_recent_transcript_steps(
    tr_path: Path | None,
    *,
    min_user_steps: int = 1,
) -> list[dict[str, Any]]:
    if tr_path is None:
        return []
    import json
    import os

    try:
        with tr_path.open("rb") as fh:
            file_size = fh.seek(0, os.SEEK_END)
            if file_size <= 0:
                return []
            window = TRANSCRIPT_INITIAL_TAIL_BYTES
            while True:
                offset = max(0, file_size - window)
                fh.seek(offset)
                raw = fh.read(file_size - offset)
                if offset > 0:
                    nl = raw.find(b"\n")
                    raw = raw[nl + 1 :] if nl >= 0 else b""
                steps: list[dict[str, Any]] = []
                for line in raw.decode("utf-8", errors="replace").splitlines():
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        obj = json.loads(s)
                        if isinstance(obj, dict):
                            steps.append(obj)
                    except json.JSONDecodeError:
                        continue
                if offset == 0:
                    return steps
                seen_user_steps = 0
                for st in reversed(steps):
                    if st.get("type") == "USER_INPUT" and st.get("source") == "USER_EXPLICIT":
                        seen_user_steps += 1
                        if seen_user_steps >= min_user_steps:
                            return steps
                window *= 4
    except OSError:
        return []


def _consume_balanced_host_tag(text: str, pos: int) -> tuple[str | None, int, bool]:
    for tag in _HOST_ENVELOPE_TAGS:
        open_tok = f"<{tag}>"
        if not text.startswith(open_tok, pos):
            continue
        close_tok = f"</{tag}>"
        depth = 1
        cur = pos + len(open_tok)
        while depth > 0:
            next_open = text.find(open_tok, cur)
            next_close = text.find(close_tok, cur)
            if next_close < 0:
                return tag, len(text), False
            if 0 <= next_open < next_close:
                depth += 1
                cur = next_open + len(open_tok)
            else:
                depth -= 1
                cur = next_close + len(close_tok)
        return tag, cur, True
    return None, pos, False


def _match_host_envelope_tail(tail: str, *, truncated: bool) -> bool:
    pos = 0
    n = len(tail)
    while pos < n:
        while pos < n and tail[pos].isspace():
            pos += 1
        if pos >= n:
            return True
        tag, next_pos, closed = _consume_balanced_host_tag(tail, pos)
        if tag is not None:
            if not closed:
                return truncated
            pos = next_pos
            continue
        if truncated:
            rem = tail[pos:]
            if any(f"<{t}>".startswith(rem) for t in _HOST_ENVELOPE_TAGS):
                return True
        return False
    return True


def _has_closed_host_envelope(body: str, *, truncated: bool) -> bool:
    close_token = "</USER_REQUEST>"
    pos = 0
    while True:
        close_idx = body.find(close_token, pos)
        if close_idx < 0:
            return False
        tail = body[close_idx + len(close_token) :]
        if _match_host_envelope_tail(tail, truncated=truncated):
            return True
        pos = close_idx + len(close_token)


def _match_wrapped_after_body(body: str, target_len: int, *, truncated: bool) -> bool:
    after = body[target_len:]
    if not after:
        return truncated and not _has_closed_host_envelope(body[:target_len], truncated=True)
    for close_prefix in ("\r\n</USER_REQUEST>", "\n</USER_REQUEST>", "</USER_REQUEST>"):
        if after.startswith(close_prefix):
            return _match_host_envelope_tail(after[len(close_prefix) :], truncated=truncated)
    if truncated:
        if after in ("\r\n", "\n"):
            return not _has_closed_host_envelope(body[:target_len], truncated=True)
        if (
            "\r\n</USER_REQUEST>".startswith(after)
            or "\n</USER_REQUEST>".startswith(after)
            or (len(after) >= 2 and "</USER_REQUEST>".startswith(after))
        ):
            has_unclosed_meta = _has_closed_host_envelope(
                body[:target_len], truncated=True
            ) and not _has_closed_host_envelope(body[:target_len], truncated=False)
            return not has_unclosed_meta
    return False


def _step_matches_input(step: dict[str, Any], raw_input: str) -> bool:
    content = str(step.get("content") or "")
    if not content:
        return False
    trunc = step.get("truncated_fields") or []
    is_truncated = isinstance(trunc, list) and "content" in trunc
    targets = [t for t in dict.fromkeys((raw_input, raw_input.strip())) if t]
    if not targets:
        return False

    left = content.lstrip()
    if not left:
        return False
    if left.startswith("<") and not left.startswith("<USER_REQUEST>"):
        cand_left = left
        while True:
            tag, next_pos, closed = _consume_balanced_host_tag(cand_left, 0)
            if tag is None or not closed:
                break
            cand_left = cand_left[next_pos:].lstrip()
            if cand_left.startswith("<USER_REQUEST>"):
                left = cand_left
                break
    if not left.startswith("<USER_REQUEST>"):
        if not is_truncated:
            return content in targets or content.strip() in targets
        return any(t.startswith(content) or t.startswith(left) for t in targets)

    inner = left[len("<USER_REQUEST>") :]
    body_candidates: list[str] = []
    if inner.startswith("\r\n"):
        body_candidates.append(inner[2:])
    elif inner.startswith("\n"):
        body_candidates.append(inner[1:])
    else:
        body_candidates.append(inner)
        if inner.lstrip() != inner:
            body_candidates.append(inner.lstrip())

    for body in body_candidates:
        if not body:
            continue
        for target in targets:
            if body.startswith(target):
                if _match_wrapped_after_body(body, len(target), truncated=is_truncated):
                    return True
            elif (
                is_truncated
                and target.startswith(body)
                and not _has_closed_host_envelope(body, truncated=True)
            ):
                return True
    return False


def _verified_user_step(payload: dict[str, Any], raw_input: str) -> tuple[int, list[dict[str, Any]], int]:
    tr_path = _resolve_transcript_path(payload)
    steps = _load_recent_transcript_steps(tr_path)
    for idx in range(len(steps) - 1, -1, -1):
        st = steps[idx]
        if st.get("type") == "USER_INPUT" and st.get("source") == "USER_EXPLICIT":
            if _step_matches_input(st, raw_input):
                try:
                    return int(st.get("step_index", idx)), steps, idx
                except (TypeError, ValueError):
                    return idx, steps, idx
            return -1, steps, -1
    return -1, steps, -1


def _step_index_of(st: dict[str, Any], fallback_pos: int) -> int:
    try:
        return int(st.get("step_index", fallback_pos))
    except (TypeError, ValueError):
        return fallback_pos


def _has_prior_explicit_step_matching(
    payload: dict[str, Any],
    steps: list[dict[str, Any]],
    before_pos: int,
    prev_input: str,
    *,
    target_step_index: int | None = None,
    after_step_index: int | None = None,
) -> bool:
    if not prev_input:
        return False

    def _scan_prior(step_list: list[dict[str, Any]], pos: int) -> tuple[bool, bool]:
        for idx in range(pos - 1, -1, -1):
            st = step_list[idx]
            if st.get("type") == "USER_INPUT" and st.get("source") == "USER_EXPLICIT":
                if after_step_index is not None:
                    st_idx = _step_index_of(st, idx)
                    if st_idx <= after_step_index:
                        return False, True
                return _step_matches_input(st, prev_input), True
        return False, False

    matched, found_any = _scan_prior(steps, before_pos)
    if found_any:
        return matched
    tr_path = _resolve_transcript_path(payload)
    wider_steps = _load_recent_transcript_steps(tr_path, min_user_steps=2)
    if len(wider_steps) > len(steps):
        wider_pos = -1
        for idx in range(len(wider_steps) - 1, -1, -1):
            st = wider_steps[idx]
            if st.get("type") == "USER_INPUT" and st.get("source") == "USER_EXPLICIT":
                if target_step_index is None or _step_index_of(st, idx) == target_step_index:
                    wider_pos = idx
                    break
        if wider_pos > 0:
            matched, _ = _scan_prior(wider_steps, wider_pos)
            return matched
    return False


def _is_same_turn_handoff(steps: list[dict[str, Any]], user_pos: int) -> bool:
    """Return True only when steps after user_pos show an active plan handoff or Stop continuation."""
    if user_pos < 0 or user_pos >= len(steps) - 1:
        return False
    if _transcript_turn_ended(steps, user_pos):
        return False
    after_user = steps[user_pos + 1 :]
    last_planner_pos = -1
    for i, st in enumerate(after_user):
        if st.get("type") == "PLANNER_RESPONSE":
            last_planner_pos = i
    if last_planner_pos < 0:
        return False
    last_planner = after_user[last_planner_pos]
    last_planner_status = str(last_planner.get("status") or "").upper()
    if last_planner_status in ("ERROR", "CANCELED", "CANCELLED"):
        return False
    for st in after_user[last_planner_pos + 1 :]:
        if st.get("type") == "SYSTEM_MESSAGE":
            return True
    for tc in last_planner.get("tool_calls") or []:
        tname = str(tc.get("name") or "").lower()
        if tname in ("exit_plan_mode", "enter_plan_mode"):
            return True
    return False


def _transcript_turn_ended(steps: list[dict[str, Any]], user_pos: int) -> bool:
    if user_pos < 0 or user_pos >= len(steps) - 1:
        return False
    after_user = steps[user_pos + 1 :]
    last_step = steps[-1]
    stype = str(last_step.get("type") or "")
    status = str(last_step.get("status") or "").upper()
    if status in ("CANCELED", "CANCELLED"):
        return True
    if stype == "PLANNER_RESPONSE":
        if not (last_step.get("tool_calls") or []):
            return True
        if status == "ERROR":
            return True
    last_planner_pos = -1
    for i, st in enumerate(after_user):
        if st.get("type") == "PLANNER_RESPONSE":
            last_planner_pos = i
    if last_planner_pos >= 0:
        for st in after_user[last_planner_pos + 1 :]:
            if str(st.get("status") or "").upper() in ("CANCELED", "CANCELLED"):
                return True
    return False


def _parse_optional_int(val: Any) -> int | None:
    if val is None or val == "":
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def main() -> None:
    try:
        raw_stdin = sys.stdin.read()
    except Exception:
        print("{}")
        return
    low_stdin = raw_stdin.lower()
    if not any(tok in low_stdin for tok in ("++", "＋＋", "\\uff0b", "\\u002b")):
        print("{}")
        return

    import json

    try:
        payload = json.loads(raw_stdin)
        if not isinstance(payload, dict):
            payload = {}
    except Exception:
        print("{}")
        return

    raw_input = str(payload.get("lastUserInput") or "")
    if not raw_input.lstrip().startswith(("++", "＋＋")):
        print("{}")
        return

    import hashlib
    from pathlib import Path

    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    import prompt_forge_core as core  # noqa: E402

    try:
        prefs = core.load_preferences()
        templates = core.load_all_templates()
        match = core.parse_trigger(
            raw_input,
            default_mode=prefs.get("default_mode", "confirm"),
            custom_template_ids=set(templates.keys()),
        )
        if match is None:
            print("{}")
            return

        cid = str(payload.get("conversationId") or "default").strip() or "default"
        inv_num = _parse_optional_int(payload.get("invocationNum"))
        inv_num_val = inv_num if inv_num is not None else 0
        initial_num_steps = _parse_optional_int(payload.get("initialNumSteps"))
        cur_user_step, steps, user_pos = _verified_user_step(payload, raw_input)
        input_hash = hashlib.sha256(raw_input.encode("utf-8")).hexdigest()

        state = core.load_session(cid)
        stale_user_step: int | None = None
        prior_init_bound: int | None = None
        prior_unbound_input: str | None = None
        prev_step = _parse_optional_int(state.get("lastUserStepIndex"))
        prev_stale_step = _parse_optional_int(state.get("staleUserStepIndex"))
        prev_init_bound = _parse_optional_int(state.get("priorInitBound"))
        prev_init = _parse_optional_int(state.get("lastInitialNumSteps"))
        if state.get("lastUserInputHash") == input_hash:
            prev_inv = _parse_optional_int(state.get("lastInvocationNum"))
            prior_unbound = str(state.get("priorUnboundUserInput") or "")
            stale_bound = max(
                (x for x in (prev_stale_step, prev_init_bound) if x is not None and x >= 0),
                default=-1,
            )

            if cur_user_step >= 0 and stale_bound >= 0 and cur_user_step <= stale_bound:
                # Still reading the old finished/canceled turn's USER_INPUT while waiting for flush.
                cur_user_step = -1
                user_pos = -1

            if (
                cur_user_step >= 0
                and (prev_step is None or prev_step < 0)
                and prior_unbound
                and _step_matches_input(steps[user_pos], prior_unbound)
                and not _has_prior_explicit_step_matching(
                    payload,
                    steps,
                    user_pos,
                    prior_unbound,
                    target_step_index=cur_user_step,
                    after_step_index=prev_stale_step,
                )
            ):
                # Delayed truncated step from an earlier unbound turn appeared during retry; mark stale once.
                stale_user_step = cur_user_step
                cur_user_step = -1
                user_pos = -1

            ended = _transcript_turn_ended(steps, user_pos)
            handoff = _is_same_turn_handoff(steps, user_pos)
            same_init = initial_num_steps is not None and prev_init == initial_num_steps
            inv_increased = prev_inv is not None and inv_num_val > prev_inv

            if (
                cur_user_step >= 0
                and ended
                and (prev_step is None or prev_step < 0)
                and inv_increased
            ):
                # Turn 2 Inv 1 ran before Turn 1 flushed (saved prev_step=-1); on Turn 2 Inv 2,
                # Turn 1's finished step appeared for the first time while Turn 2 is still unflushed.
                stale_user_step = cur_user_step
                cur_user_step = -1
                user_pos = -1
                ended = False
                handoff = False

            same_step_or_unflushed = (
                cur_user_step < 0
                or prev_step is None
                or prev_step < 0
                or cur_user_step == prev_step
            )

            if same_step_or_unflushed and (same_init or (inv_increased and not ended)):
                if (
                    (cur_user_step >= 0 and prev_step != cur_user_step)
                    or stale_user_step is not None
                    or initial_num_steps != prev_init
                    or inv_num_val != prev_inv
                ):
                    if cur_user_step >= 0:
                        state["lastUserStepIndex"] = cur_user_step
                        state.pop("staleUserStepIndex", None)
                        state.pop("priorInitBound", None)
                        state.pop("priorUnboundUserInput", None)
                    elif stale_user_step is not None:
                        state["staleUserStepIndex"] = stale_user_step
                        state.pop("priorInitBound", None)
                        state.pop("priorUnboundUserInput", None)
                    state["lastInitialNumSteps"] = initial_num_steps
                    state["lastInvocationNum"] = inv_num_val
                    core.save_session(cid, state)
                print("{}")
                return

            if same_step_or_unflushed and handoff:
                if (
                    (cur_user_step >= 0 and prev_step != cur_user_step)
                    or stale_user_step is not None
                    or initial_num_steps != prev_init
                    or inv_num_val != prev_inv
                ):
                    if cur_user_step >= 0:
                        state["lastUserStepIndex"] = cur_user_step
                        state.pop("staleUserStepIndex", None)
                        state.pop("priorInitBound", None)
                        state.pop("priorUnboundUserInput", None)
                    elif stale_user_step is not None:
                        state["staleUserStepIndex"] = stale_user_step
                        state.pop("priorInitBound", None)
                        state.pop("priorUnboundUserInput", None)
                    state["lastInitialNumSteps"] = initial_num_steps
                    state["lastInvocationNum"] = inv_num_val
                    core.save_session(cid, state)
                print("{}")
                return

            prev_bound = max(
                (x for x in (prev_step, prev_stale_step) if x is not None and x >= 0),
                default=-1,
            )
            init_bound = (
                (prev_init - 1)
                if ((prev_step is None or prev_step < 0) and prev_init is not None and prev_init > 0)
                else -1
            )
            effective_init_bound = max(
                (x for x in (prev_init_bound, init_bound) if x is not None and x >= 0),
                default=-1,
            )
            effective_bound = max(prev_bound, effective_init_bound)
            if cur_user_step >= 0 and (
                (effective_bound >= 0 and cur_user_step <= effective_bound)
                or ended
            ):
                # Stale USER_INPUT from a previous finished/canceled/unbound turn; mark stale until new step flushes.
                stale_user_step = max(prev_bound, cur_user_step)
                if effective_init_bound >= 0:
                    prior_init_bound = effective_init_bound
                cur_user_step = -1
            elif cur_user_step < 0:
                if prior_unbound and stale_user_step is None:
                    prior_unbound_input = prior_unbound
                carried_stale = max(
                    (x for x in (stale_user_step, prev_bound) if x is not None and x >= 0),
                    default=-1,
                )
                if carried_stale >= 0:
                    stale_user_step = carried_stale
                if effective_init_bound >= 0:
                    prior_init_bound = effective_init_bound
        else:
            prev_bound = max(
                (x for x in (prev_step, prev_stale_step) if x is not None and x >= 0),
                default=-1,
            )
            init_bound = (prev_init - 1) if (prev_init is not None and prev_init > 0) else -1
            effective_bound = max(prev_bound, init_bound)
            prev_user_input = str(state.get("lastUserInput") or "")
            matches_prev_input = (
                cur_user_step >= 0
                and (prev_step is None or prev_step < 0)
                and bool(prev_user_input)
                and _step_matches_input(steps[user_pos], prev_user_input)
                and not _has_prior_explicit_step_matching(
                    payload,
                    steps,
                    user_pos,
                    prev_user_input,
                    target_step_index=cur_user_step,
                    after_step_index=prev_stale_step,
                )
            )
            if cur_user_step >= 0 and (
                (effective_bound >= 0 and cur_user_step <= effective_bound)
                or _transcript_turn_ended(steps, user_pos)
                or matches_prev_input
            ):
                # Keep concrete stale steps separate from inferred init_bound; if the rejected step
                # was <= prev_bound while the prior turn remained unbound, preserve its text too.
                was_older_than_prev = prev_bound >= 0 and cur_user_step <= prev_bound
                stale_user_step = max(prev_bound, cur_user_step)
                if init_bound >= 0:
                    prior_init_bound = init_bound
                if (prev_step is None or prev_step < 0) and prev_user_input and was_older_than_prev:
                    prior_unbound_input = prev_user_input
                cur_user_step = -1
            elif cur_user_step < 0:
                if prev_bound >= 0:
                    stale_user_step = prev_bound
                if init_bound >= 0:
                    prior_init_bound = init_bound
                if (prev_step is None or prev_step < 0) and prev_user_input:
                    prior_unbound_input = prev_user_input

        workspace_ctx = None
        if match.mode != "prefs":
            ws_paths = [str(p) for p in (payload.get("workspacePaths") or []) if p]
            if not ws_paths and payload.get("cwd"):
                ws_paths = [str(payload["cwd"])]
            workspace_ctx = core.collect_workspace_context(ws_paths, match.raw_prompt)

        system_msg, _user_badge = core.render_enhancement_packet(
            match,
            templates=templates,
            preferences=prefs,
            workspace_ctx=workspace_ctx,
        )
        out = {
            "injectSteps": [
                {
                    "systemMessage": {
                        "systemMessage": system_msg,
                    }
                }
            ]
        }
        new_state: dict[str, Any] = {
            "lastUserStepIndex": cur_user_step,
            "lastUserInputHash": input_hash,
            "lastUserInput": raw_input[:16384],
            "lastInitialNumSteps": initial_num_steps,
            "lastInvocationNum": inv_num_val,
            "updatedAt": core._now_iso(),
        }
        if stale_user_step is not None and cur_user_step < 0:
            new_state["staleUserStepIndex"] = stale_user_step
        if prior_init_bound is not None and cur_user_step < 0:
            new_state["priorInitBound"] = prior_init_bound
        if prior_unbound_input and cur_user_step < 0:
            new_state["priorUnboundUserInput"] = prior_unbound_input[:16384]
        core.save_session(cid, new_state)
        print(json.dumps(out, ensure_ascii=False))
    except Exception as exc:
        core.append_error_log(f"pre_invocation:{payload.get('conversationId', 'default')}", exc)
        print("{}")


if __name__ == "__main__":
    main()
