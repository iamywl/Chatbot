from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from backend.config import DEFAULT_GEN_PARAMS, MLX_BASE_URL

# Regex patterns for OpenAI Harmony channels, tool calls, and <think> tags
TOOL_HEADER_A_RE = re.compile(
    r"(?:<\|start\|>assistant)?\s*to=functions\.([a-zA-Z0-9_-]+)\s*<\|channel\|>(?:commentary|analysis)(?:\s*<\|constrain\|>)?(?:\s*[a-zA-Z0-9_+-]+)?\s*<\|message\|>"
)
TOOL_HEADER_B_RE = re.compile(
    r"(?:<\|start\|>assistant)?\s*<\|channel\|>commentary\s+to=functions\.([a-zA-Z0-9_-]+)(?:\s*<\|constrain\|>)?(?:\s*[a-zA-Z0-9_+-]+)?\s*<\|message\|>"
)
REASONING_HEADER_RE = re.compile(
    r"(?:<\|start\|>assistant)?\s*<\|channel\|>analysis\s*<\|message\|>|<think>"
)
FINAL_HEADER_RE = re.compile(
    r"(?:<\|start\|>assistant)?\s*<\|channel\|>final\s*<\|message\|>|</think>"
)
CONTROL_END_RE = re.compile(r"<\|call\|>|<\|end\|>|<\|return\|>")

# Prefixes that might initiate any Harmony / think tag or tool header
HOLD_START_RE = re.compile(r"(<\|?|<t|</t|\s*to=functions\.|\s*to=f|\s*to=)")
PARTIAL_PREFIXES = (
    "<|",
    "<think>",
    "</think>",
    "to=functions.",
    " to=functions.",
)
MAX_HEADER_HOLD = 140


def _safe_parse_json_args(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}
    except json.JSONDecodeError:
        # Try extracting first JSON object if trailing garbage exists
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        return {"_raw": text}


class HarmonyStreamParser:
    """
    Incremental streaming state machine for gpt-oss-120b (OpenAI Harmony) + <think> models:
      - `analysis` channel / `<think>` -> ("reasoning", str)
      - `final` channel / plain text -> ("content", str)
      - `commentary` tool calls (`to=functions.<name>`) -> ("tool_call", {"name": str, "arguments": dict})
    Handles arbitrary chunk boundaries (even 1 character at a time).
    """

    def __init__(self) -> None:
        self.mode: str = "content"  # "content" | "reasoning" | "tool" | "control"
        self.buffer: str = ""
        self._tool_name: str | None = None
        self._tool_args_buf: list[str] = []

    def feed(
        self, content_chunk: str | None = None, reasoning_chunk: str | None = None
    ) -> list[tuple[str, Any]]:
        events: list[tuple[str, Any]] = []
        if reasoning_chunk:
            events.append(("reasoning", reasoning_chunk))
        if not content_chunk:
            return events

        self.buffer += content_chunk
        events.extend(self._drain(final=False))
        return events

    def flush(self) -> list[tuple[str, Any]]:
        events = self._drain(final=True)
        if self.mode == "tool" and self._tool_name:
            events.append(self._finalize_tool_call())
        return events

    def _finalize_tool_call(self) -> tuple[str, dict[str, Any]]:
        name = self._tool_name or "unknown"
        raw_args = "".join(self._tool_args_buf)
        self._tool_name = None
        self._tool_args_buf = []
        self.mode = "control"
        return ("tool_call", {"name": name, "arguments": _safe_parse_json_args(raw_args)})

    def _emit_text(self, events: list[tuple[str, Any]], chunk: str) -> None:
        if not chunk:
            return
        if self.mode == "reasoning":
            events.append(("reasoning", chunk))
        elif self.mode == "content":
            events.append(("content", chunk))
        elif self.mode == "tool":
            self._tool_args_buf.append(chunk)
        elif self.mode == "control":
            # Drop control whitespace/stray role markers between <|end|> and next header
            cleaned = chunk.strip()
            if cleaned and cleaned not in ("assistant", "analysis", "final", "commentary"):
                events.append(("content", chunk))

    def _find_earliest_match(self, text: str) -> tuple[int, int, str, str | None] | None:
        candidates: list[tuple[int, int, str, str | None]] = []

        for rx in (TOOL_HEADER_A_RE, TOOL_HEADER_B_RE):
            m = rx.search(text)
            if m:
                candidates.append((m.start(), m.end(), "tool", m.group(1)))

        m = REASONING_HEADER_RE.search(text)
        if m:
            candidates.append((m.start(), m.end(), "reasoning", None))

        m = FINAL_HEADER_RE.search(text)
        if m:
            candidates.append((m.start(), m.end(), "content", None))

        m = CONTROL_END_RE.search(text)
        if m:
            candidates.append((m.start(), m.end(), "end", m.group(0)))

        if not candidates:
            return None
        # Earliest start position wins; on tie, longest span wins
        candidates.sort(key=lambda item: (item[0], -(item[1] - item[0])))
        return candidates[0]

    def _compute_safe_emit_len(self, text: str) -> int:
        """Hold back any tail that could be the beginning of an incomplete Harmony/tool/think tag."""
        # 1) If `<|` or `to=functions.` already started in the buffer, hold from its start
        #    (up to MAX_HEADER_HOLD chars so malformed plain text doesn't stall forever)
        tail_window_start = max(0, len(text) - MAX_HEADER_HOLD)
        window = text[tail_window_start:]

        hold_idx = len(text)

        for marker in ("<|", "<think", "</think", "to=functions."):
            pos = window.find(marker)
            if pos != -1:
                hold_idx = min(hold_idx, tail_window_start + pos)

        # Also check short trailing suffixes that could start any marker (e.g. `<`, ` to=f`, `to=`)
        suffix_scan_start = max(0, len(text) - 16)
        for i in range(suffix_scan_start, len(text)):
            suf = text[i:]
            for pref in PARTIAL_PREFIXES:
                if pref.startswith(suf):
                    hold_idx = min(hold_idx, i)
                    break

        return hold_idx

    def _drain(self, final: bool) -> list[tuple[str, Any]]:
        events: list[tuple[str, Any]] = []
        while self.buffer:
            match = self._find_earliest_match(self.buffer)
            if match is not None:
                start, end, kind, extra = match
                if start > 0:
                    self._emit_text(events, self.buffer[:start])
                self.buffer = self.buffer[end:]

                if kind == "tool":
                    if self.mode == "tool" and self._tool_name:
                        events.append(self._finalize_tool_call())
                    self.mode = "tool"
                    self._tool_name = extra
                    self._tool_args_buf = []
                elif kind == "reasoning":
                    if self.mode == "tool" and self._tool_name:
                        events.append(self._finalize_tool_call())
                    self.mode = "reasoning"
                elif kind == "content":
                    if self.mode == "tool" and self._tool_name:
                        events.append(self._finalize_tool_call())
                    self.mode = "content"
                elif kind == "end":
                    if self.mode == "tool" and self._tool_name:
                        events.append(self._finalize_tool_call())
                    self.mode = "control"
                continue

            if final:
                self._emit_text(events, self.buffer)
                self.buffer = ""
                break

            safe_len = self._compute_safe_emit_len(self.buffer)
            if safe_len > 0:
                self._emit_text(events, self.buffer[:safe_len])
                self.buffer = self.buffer[safe_len:]
            break

        return events


def build_agent_system_instructions(agent_mode: str, workspace_root: str) -> str:
    if agent_mode == "off":
        return ""
    if agent_mode == "read_only":
        return (
            f"\n\n[로컬 워크스페이스 에이전트 - 읽기 전용]\n"
            f"- 현재 활성 워크스페이스 경로: `{workspace_root}`\n"
            f"- `list_dir`, `read_file`, `search_files` 도구를 사용해 실제 파일을 확인한 뒤 사실에 근거해 답변하세요.\n"
            f"- 파일 수정이나 명령 실행은 현재 읽기 전용 모드라 비활성화되어 있습니다."
        )
    mode_note = (
        "파일 생성·수정(`write_file`, `edit_file`) 및 터미널 명령(`run_command`)은 사용자 승인 후 실행됩니다."
        if agent_mode == "approve"
        else "파일 읽기·수정 및 안전한 명령을 자율적으로 실행할 수 있습니다."
    )
    return (
        f"\n\n[로컬 워크스페이스 코딩 에이전트 (Antigravity 모드)]\n"
        f"- 현재 활성 워크스페이스 경로: `{workspace_root}`\n"
        f"- 사용 가능한 도구: `list_dir`, `read_file`, `search_files`, `write_file`, `edit_file`, `run_command`.\n"
        f"- {mode_note}\n"
        f"- 코드를 수정하기 전에는 반드시 `read_file`이나 `search_files`로 실제 파일을 확인하세요.\n"
        f"- 수정 후에는 가능하면 테스트나 검증 명령을 실행해 결과를 확인하고, 한국어로 명확하게 요약하세요."
    )


def build_llm_messages(
    system_prompt: str,
    history_messages: list[dict[str, Any]],
    extra_system: str = "",
) -> list[dict[str, Any]]:
    """
    Build OpenAI messages[] payload.
    CRITICAL for gpt-oss-120b: only include final assistant `content` in multi-turn context,
    NEVER re-inject past internal reasoning/CoT tokens.
    """
    messages: list[dict[str, Any]] = []
    full_system = (system_prompt.strip() + extra_system).strip()
    if full_system:
        messages.append({"role": "system", "content": full_system})

    for msg in history_messages:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    return messages


async def stream_chat_completion(
    messages: list[dict[str, Any]],
    params: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
    cancel_event: asyncio.Event | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Stream one completion step from local mlx_lm.server, yielding structured events:
      - {"type": "reasoning", "text": "..."}
      - {"type": "content", "text": "..."}
      - {"type": "tool_call", "name": "...", "arguments": {...}}
      - {"type": "keepalive", "detail": "..."}
      - {"type": "done", "metrics": {...}, "cancelled": bool}
    """
    cfg = {**DEFAULT_GEN_PARAMS, **(params or {})}
    reasoning_effort = cfg.get("reasoning_effort", "medium")
    if reasoning_effort not in ("low", "medium", "high"):
        reasoning_effort = "medium"

    payload: dict[str, Any] = {
        "model": "default_model",
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": float(cfg["temperature"]),
        "top_p": float(cfg["top_p"]),
        "presence_penalty": float(cfg["presence_penalty"]),
        "max_tokens": int(cfg["max_tokens"]),
        "chat_template_kwargs": {"reasoning_effort": reasoning_effort},
    }

    if tools:
        payload["tools"] = tools
        # Stop cleanly at Harmony <|call|> so the model doesn't hallucinate tool output
        payload["stop"] = ["<|call|>"]

    parser = HarmonyStreamParser()
    t_start = time.monotonic()
    t_first_token: float | None = None
    completion_tokens = 0
    prompt_tokens = 0
    chunk_count = 0
    cancelled = False

    timeout = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST", f"{MLX_BASE_URL}/v1/chat/completions", json=payload
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise RuntimeError(
                    f"mlx_lm.server 오류 (HTTP {resp.status_code}): {body.decode('utf-8', errors='replace')[:400]}"
                )

            async for raw_line in resp.aiter_lines():
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = True
                    break

                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith(": keepalive"):
                    yield {"type": "keepalive", "detail": line[2:].strip()}
                    continue
                if not line.startswith("data:"):
                    continue

                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                usage = data.get("usage")
                if isinstance(usage, dict):
                    prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
                    completion_tokens = int(
                        usage.get("completion_tokens") or completion_tokens
                    )

                choices = data.get("choices") or []
                if not choices:
                    continue

                delta = choices[0].get("delta") or {}
                content_piece = delta.get("content")
                reasoning_piece = delta.get("reasoning") or delta.get("reasoning_content")
                native_tool_calls = delta.get("tool_calls") or []

                if content_piece or reasoning_piece or native_tool_calls:
                    if t_first_token is None:
                        t_first_token = time.monotonic()
                    chunk_count += 1

                for tc in native_tool_calls:
                    fn = tc.get("function") or {}
                    t_name = fn.get("name")
                    t_args = fn.get("arguments", {})
                    if isinstance(t_args, str):
                        t_args = _safe_parse_json_args(t_args)
                    if t_name:
                        yield {"type": "tool_call", "name": t_name, "arguments": t_args}

                for ev_type, val in parser.feed(content_piece, reasoning_piece):
                    if ev_type == "tool_call":
                        yield {"type": "tool_call", **val}
                    else:
                        yield {"type": ev_type, "text": val}

    for ev_type, val in parser.flush():
        if ev_type == "tool_call":
            yield {"type": "tool_call", **val}
        else:
            yield {"type": ev_type, "text": val}

    elapsed = max(time.monotonic() - t_start, 0.001)
    ttft = round(t_first_token - t_start, 2) if t_first_token else round(elapsed, 2)
    gen_time = (
        max(time.monotonic() - t_first_token, 0.001)
        if t_first_token
        else elapsed
    )
    tokens_out = completion_tokens if completion_tokens > 0 else chunk_count
    tok_per_sec = round(tokens_out / gen_time, 1) if tokens_out > 0 else 0.0

    yield {
        "type": "done",
        "cancelled": cancelled,
        "metrics": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": tokens_out,
            "ttft_sec": ttft,
            "duration_sec": round(elapsed, 2),
            "tok_per_sec": tok_per_sec,
        },
    }
