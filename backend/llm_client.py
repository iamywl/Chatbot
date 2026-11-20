from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from backend.config import DEFAULT_GEN_PARAMS, MLX_BASE_URL

# Control/channel tokens emitted by gpt-oss-120b (Harmony format) & standard <think> models
CONTROL_TOKENS: list[tuple[str, str | None]] = [
    ("<|channel|>analysis<|message|>", "reasoning"),
    ("<|channel|>commentary<|message|>", "reasoning"),
    ("<|channel|>final<|message|>", "content"),
    ("<|start|>assistant<|channel|>analysis<|message|>", "reasoning"),
    ("<|start|>assistant<|channel|>commentary<|message|>", "reasoning"),
    ("<|start|>assistant<|channel|>final<|message|>", "content"),
    ("<|start|>assistant", "control"),
    ("<|end|>", "control"),
    ("<|return|>", "control"),
    ("<|call|>", "control"),
    ("<|start|>", "control"),
    ("<|message|>", "control"),
    ("<|channel|>", "control"),
    ("<think>", "reasoning"),
    ("</think>", "content"),
]

# Precompute longest control token for tail buffering
MAX_TAG_LEN = max(len(t[0]) for t in CONTROL_TOKENS)
# Special prefixes that can start any control token
TAG_START_MARKERS = ("<|", "<think", "</think")


class HarmonyStreamParser:
    """
    Incremental streaming state machine that cleanly separates:
      - OpenAI Harmony tags (`<|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>...`)
      - Standard `<think>...</think>` tags
      - Native OpenAI/mlx_lm `delta.reasoning` fields
    even when tags are split across arbitrary token/SSE chunks.
    """

    def __init__(self) -> None:
        self.mode: str = "content"  # "reasoning" | "content" | "control"
        self.buffer: str = ""
        self._saw_harmony: bool = False

    def feed(
        self, content_chunk: str | None = None, reasoning_chunk: str | None = None
    ) -> list[tuple[str, str]]:
        events: list[tuple[str, str]] = []
        if reasoning_chunk:
            events.append(("reasoning", reasoning_chunk))
        if not content_chunk:
            return events

        self.buffer += content_chunk
        events.extend(self._drain(final=False))
        return events

    def flush(self) -> list[tuple[str, str]]:
        return self._drain(final=True)

    def _find_next_tag(self, text: str) -> tuple[int, str, str | None] | None:
        best_pos = -1
        best_tok = ""
        best_action: str | None = None
        for tok, action in CONTROL_TOKENS:
            pos = text.find(tok)
            if pos != -1:
                # Pick earliest match; on tie pick longest token
                if best_pos == -1 or pos < best_pos or (
                    pos == best_pos and len(tok) > len(best_tok)
                ):
                    best_pos = pos
                    best_tok = tok
                    best_action = action
        if best_pos == -1:
            return None
        return best_pos, best_tok, best_action

    def _safe_prefix_len(self, text: str) -> int:
        """Return index up to which text is guaranteed not to contain the start of a partial tag."""
        tail_start = max(0, len(text) - MAX_TAG_LEN)
        tail = text[tail_start:]
        # Check if any suffix of `tail` is a proper prefix of any known CONTROL_TOKEN
        earliest_hold = len(text)
        for i in range(len(tail)):
            suffix = tail[i:]
            if not suffix.startswith("<"):
                continue
            for tok, _ in CONTROL_TOKENS:
                if tok.startswith(suffix):
                    earliest_hold = min(earliest_hold, tail_start + i)
                    break
        return earliest_hold

    def _emit(self, events: list[tuple[str, str]], chunk: str) -> None:
        if not chunk:
            return
        if self.mode == "reasoning":
            events.append(("reasoning", chunk))
        elif self.mode == "content":
            events.append(("content", chunk))
        # In "control" mode (between <|end|> and next <|channel|>...<|message|>),
        # ignore whitespace/stray channel header fragments unless plain text resumes.
        elif self.mode == "control":
            stripped = chunk.strip()
            if stripped and stripped not in ("assistant", "analysis", "final", "commentary"):
                events.append(("content", chunk))

    def _drain(self, final: bool) -> list[tuple[str, str]]:
        events: list[tuple[str, str]] = []
        while self.buffer:
            match = self._find_next_tag(self.buffer)
            if match is not None:
                pos, tok, action = match
                if pos > 0:
                    self._emit(events, self.buffer[:pos])
                # Advance past matched token
                self.buffer = self.buffer[pos + len(tok) :]
                if tok.startswith("<|"):
                    self._saw_harmony = True
                if action == "reasoning":
                    self.mode = "reasoning"
                elif action == "content":
                    self.mode = "content"
                elif action == "control":
                    # Wait for next channel tag; if we matched `<|channel|>` alone, check
                    # whether the remaining buffer might complete `<|channel|>final<|message|>`.
                    if tok in ("<|channel|>", "<|start|>"):
                        # Only consume bare <|channel|> / <|start|> when final or followed by unknown text
                        if not final and len(self.buffer) < 30:
                            # Put it back and wait for more tokens to form the full compound tag
                            self.buffer = tok + self.buffer
                            break
                    self.mode = "control"
                continue

            if final:
                self._emit(events, self.buffer)
                self.buffer = ""
                break

            safe_len = self._safe_prefix_len(self.buffer)
            if safe_len > 0:
                self._emit(events, self.buffer[:safe_len])
                self.buffer = self.buffer[safe_len:]
            break

        return events


def build_llm_messages(
    system_prompt: str,
    history_messages: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """
    Build OpenAI messages[] payload.
    CRITICAL for gpt-oss-120b: only include final assistant `content` in multi-turn context,
    NEVER re-inject past internal reasoning/CoT tokens.
    """
    messages: list[dict[str, str]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})

    for msg in history_messages:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    return messages


async def stream_chat_completion(
    messages: list[dict[str, str]],
    params: dict[str, Any] | None = None,
    cancel_event: asyncio.Event | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Stream chat completion from local mlx_lm.server, yielding structured events:
      - {"type": "reasoning", "text": "..."}
      - {"type": "content", "text": "..."}
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

                if content_piece or reasoning_piece:
                    if t_first_token is None:
                        t_first_token = time.monotonic()
                    chunk_count += 1

                for ev_type, text in parser.feed(content_piece, reasoning_piece):
                    yield {"type": ev_type, "text": text}

    for ev_type, text in parser.flush():
        yield {"type": ev_type, "text": text}

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
