from __future__ import annotations

import asyncio
import json
import secrets
import urllib.parse
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from backend.config import DEFAULT_GEN_PARAMS, DEFAULT_MODEL_ID, FRONTEND_DIR
from backend.crypto_vault import (
    CryptoVault,
    VaultAuthError,
    VaultDecryptError,
    VaultError,
    VaultLockedError,
)
from backend.git_sync import GitSyncWorker
from backend.llm_client import (
    build_agent_system_instructions,
    build_llm_messages,
    stream_chat_completion,
)
from backend.model_manager import ModelManager
from backend.storage import EncryptedSessionStorage, new_message_id, now_iso
from backend.workspace_tools import (
    WorkspaceManager,
    WorkspaceSecurityError,
    get_tool_schemas,
)

MAX_AGENT_STEPS = 12


@dataclass
class PendingApproval:
    approval_id: str
    session_id: str
    tool_name: str
    event: asyncio.Event = field(default_factory=asyncio.Event)
    approved: bool | None = None


# Singletons for local personal desktop app
vault = CryptoVault()
storage = EncryptedSessionStorage(vault)
model_manager = ModelManager()
git_sync = GitSyncWorker()
workspace_manager = WorkspaceManager()

# Track active generation cancellation events & pending tool approvals
active_streams: dict[str, asyncio.Event] = {}
pending_approvals: dict[str, PendingApproval] = {}


def _cancel_session_streams(session_id: str | None = None) -> None:
    if session_id is None:
        for ev in active_streams.values():
            ev.set()
        for pa in list(pending_approvals.values()):
            pa.approved = False
            pa.event.set()
    else:
        if session_id in active_streams:
            active_streams[session_id].set()
        for pa in list(pending_approvals.values()):
            if pa.session_id == session_id:
                pa.approved = False
                pa.event.set()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    model_manager.start_background_tasks()
    try:
        yield
    finally:
        _cancel_session_streams()
        await git_sync.shutdown()
        await model_manager.shutdown()
        vault.lock()


app = FastAPI(title="Private MLX Chatbot Vault", lifespan=lifespan)


# ----- Error handlers -----

@app.exception_handler(VaultLockedError)
async def handle_vault_locked(_req: Request, exc: VaultLockedError) -> JSONResponse:
    return JSONResponse(status_code=423, content={"detail": str(exc), "code": "VAULT_LOCKED"})


@app.exception_handler(VaultAuthError)
async def handle_vault_auth(_req: Request, exc: VaultAuthError) -> JSONResponse:
    return JSONResponse(status_code=401, content={"detail": str(exc), "code": "VAULT_AUTH"})


@app.exception_handler(VaultDecryptError)
async def handle_vault_decrypt(_req: Request, exc: VaultDecryptError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc), "code": "VAULT_DECRYPT"})


@app.exception_handler(VaultError)
async def handle_vault_error(_req: Request, exc: VaultError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc), "code": "VAULT_ERROR"})


@app.exception_handler(WorkspaceSecurityError)
async def handle_workspace_security(_req: Request, exc: WorkspaceSecurityError) -> JSONResponse:
    return JSONResponse(status_code=403, content={"detail": str(exc), "code": "WORKSPACE_FORBIDDEN"})


@app.exception_handler(FileNotFoundError)
async def handle_not_found(_req: Request, exc: FileNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(ValueError)
async def handle_value_error(_req: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# ----- Request Schemas -----

class VaultInitRequest(BaseModel):
    mode: str = Field(default="passphrase", pattern="^(passphrase|keyfile)$")
    passphrase: str | None = None


class VaultUnlockRequest(BaseModel):
    passphrase: str | None = None
    key_text: str | None = None


class VaultSettingsRequest(BaseModel):
    auto_lock_minutes: int = Field(ge=0, le=1440)


class ModelLoadRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str | None = None


class ModelSettingsRequest(BaseModel):
    idle_unload_minutes: int = Field(ge=0, le=1440)


class CreateSessionRequest(BaseModel):
    title: str = "새 대화"
    persona: str = "default"
    params: dict[str, Any] | None = None


class UpdateSessionRequest(BaseModel):
    title: str | None = None
    persona: str | None = None
    params: dict[str, Any] | None = None


class ForkSessionRequest(BaseModel):
    up_to_message_id: str | None = None


class ChatRequest(BaseModel):
    content: str = ""
    params: dict[str, Any] | None = None
    regenerate: bool = False
    agent_mode: str = Field(default="approve", pattern="^(off|read_only|approve|auto)$")
    attached_files: list[str] = Field(default_factory=list)


class PersonaSaveRequest(BaseModel):
    content: str


class GitSettingsRequest(BaseModel):
    auto_sync: bool


class WorkspaceSelectRequest(BaseModel):
    path: str


class ApprovalDecisionRequest(BaseModel):
    approved: bool


# ----- Vault API -----

@app.get("/api/vault/status")
async def get_vault_status() -> dict[str, Any]:
    st = vault.status()
    st["encrypted_sessions_count"] = storage.count_encrypted_files()
    return st


@app.post("/api/vault/init")
async def init_vault(req: VaultInitRequest) -> dict[str, Any]:
    if req.mode == "passphrase":
        if not req.passphrase:
            raise VaultAuthError("패스프레이즈를 입력하세요.")
        backup_key = vault.init_with_passphrase(req.passphrase)
    else:
        backup_key = vault.init_with_random_key()
    git_sync.schedule_sync()
    st = vault.status()
    st["encrypted_sessions_count"] = storage.count_encrypted_files()
    return {"status": st, "backup_key": backup_key}


@app.post("/api/vault/unlock")
async def unlock_vault(req: VaultUnlockRequest) -> dict[str, Any]:
    if req.key_text and req.key_text.strip():
        vault.unlock_with_key_text(req.key_text)
    elif req.passphrase is not None and req.passphrase != "":
        vault.unlock_with_passphrase(req.passphrase)
    else:
        raise VaultAuthError("패스프레이즈 또는 개인 키(.key)를 입력하세요.")
    st = vault.status()
    st["encrypted_sessions_count"] = storage.count_encrypted_files()
    return st


@app.post("/api/vault/lock")
async def lock_vault() -> dict[str, Any]:
    _cancel_session_streams()
    vault.lock()
    st = vault.status()
    st["encrypted_sessions_count"] = storage.count_encrypted_files()
    return st


@app.get("/api/vault/export-key")
async def export_vault_key() -> dict[str, str]:
    return {"key_text": vault.export_key_text()}


@app.post("/api/vault/settings")
async def update_vault_settings(req: VaultSettingsRequest) -> dict[str, Any]:
    vault.set_auto_lock_minutes(req.auto_lock_minutes)
    st = vault.status()
    st["encrypted_sessions_count"] = storage.count_encrypted_files()
    return st


# ----- Model Lifecycle API -----

@app.get("/api/model/status")
async def get_model_status() -> dict[str, Any]:
    return model_manager.get_state()


@app.get("/api/model/logs")
async def get_model_logs(limit: int = Query(default=100, ge=1, le=250)) -> dict[str, Any]:
    return {"logs": model_manager.get_logs(limit=limit)}


@app.post("/api/model/load")
async def load_model(req: ModelLoadRequest | None = None) -> dict[str, Any]:
    model_id = req.model_id if req else None
    return await model_manager.load(model_id=model_id)


@app.post("/api/model/unload")
async def unload_model() -> dict[str, Any]:
    _cancel_session_streams()
    return await model_manager.unload()


@app.post("/api/model/settings")
async def update_model_settings(req: ModelSettingsRequest) -> dict[str, Any]:
    model_manager.set_idle_unload_minutes(req.idle_unload_minutes)
    return model_manager.get_state()


# ----- Workspace & Local Filesystem API (Antigravity Mode) -----

@app.get("/api/workspace/status")
async def get_workspace_status() -> dict[str, Any]:
    return workspace_manager.get_status()


@app.post("/api/workspace/select")
async def select_workspace(req: WorkspaceSelectRequest) -> dict[str, Any]:
    return workspace_manager.set_workspace(req.path)


@app.post("/api/workspace/pick-native")
async def pick_workspace_native() -> dict[str, Any]:
    return await workspace_manager.pick_native_folder()


@app.get("/api/workspace/tree")
async def get_workspace_tree(path: str = Query(default=".")) -> dict[str, Any]:
    return {
        "root": str(workspace_manager.root),
        "path": path,
        "entries": workspace_manager.list_tree(path),
    }


@app.get("/api/workspace/file")
async def get_workspace_file(
    path: str = Query(...),
    start_line: int = Query(default=1, ge=1),
    end_line: int | None = Query(default=None, ge=1),
) -> dict[str, Any]:
    return workspace_manager.read_file_for_context(
        path, start_line=start_line, end_line=end_line
    )


@app.post("/api/agent/approvals/{approval_id}")
async def resolve_agent_approval(
    approval_id: str, req: ApprovalDecisionRequest
) -> dict[str, Any]:
    pa = pending_approvals.get(approval_id)
    if pa is None:
        raise HTTPException(status_code=404, detail="만료되었거나 존재하지 않는 승인 요청입니다.")
    pa.approved = req.approved
    pa.event.set()
    return {"approval_id": approval_id, "approved": req.approved}


# ----- Sessions API -----

@app.get("/api/sessions")
async def list_sessions(
    q: str = Query(default=""), persona: str = Query(default="")
) -> dict[str, Any]:
    if not vault.is_unlocked():
        raise VaultLockedError("Vault가 잠겨 있습니다.")
    return {"sessions": storage.list_sessions(query=q, persona=persona)}


@app.post("/api/sessions")
async def create_session(req: CreateSessionRequest) -> dict[str, Any]:
    if not vault.is_unlocked():
        raise VaultLockedError("Vault가 잠겨 있습니다.")
    session = storage.create_session(
        title=req.title,
        persona=req.persona,
        model=model_manager.model_id or DEFAULT_MODEL_ID,
        params=req.params,
    )
    git_sync.schedule_sync()
    return session


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str) -> dict[str, Any]:
    if not vault.is_unlocked():
        raise VaultLockedError("Vault가 잠겨 있습니다.")
    return storage.get_session(session_id)


@app.patch("/api/sessions/{session_id}")
async def update_session(session_id: str, req: UpdateSessionRequest) -> dict[str, Any]:
    if not vault.is_unlocked():
        raise VaultLockedError("Vault가 잠겨 있습니다.")
    session = storage.update_session_meta(
        session_id=session_id,
        title=req.title,
        persona=req.persona,
        params=req.params,
    )
    git_sync.schedule_sync()
    return session


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str) -> dict[str, Any]:
    if not vault.is_unlocked():
        raise VaultLockedError("Vault가 잠겨 있습니다.")
    _cancel_session_streams(session_id)
    deleted = storage.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    git_sync.schedule_sync()
    return {"deleted": True, "session_id": session_id}


@app.post("/api/sessions/{session_id}/fork")
async def fork_session(session_id: str, req: ForkSessionRequest) -> dict[str, Any]:
    if not vault.is_unlocked():
        raise VaultLockedError("Vault가 잠겨 있습니다.")
    forked = storage.fork_session(session_id, up_to_message_id=req.up_to_message_id)
    git_sync.schedule_sync()
    return forked


@app.get("/api/sessions/{session_id}/export")
async def export_session(session_id: str) -> PlainTextResponse:
    if not vault.is_unlocked():
        raise VaultLockedError("Vault가 잠겨 있습니다.")
    filename, md = storage.export_session_markdown(session_id)
    quoted = urllib.parse.quote(filename)
    return PlainTextResponse(
        content=md,
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{quoted}"
        },
    )


# ----- Chat & Agentic Streaming API -----

def _auto_title(text: str) -> str:
    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "새 대화")
    cleaned = first_line.lstrip("#>-* ").strip()
    if len(cleaned) > 38:
        return cleaned[:38].rstrip() + "…"
    return cleaned or "새 대화"


def _sse(event_type: str, payload: dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _build_attached_files_context(attached_files: list[str]) -> str:
    if not attached_files:
        return ""
    blocks: list[str] = ["\n\n---\n[사용자가 첨부한 로컬 워크스페이스 파일]"]
    for rel in attached_files[:10]:
        try:
            info = workspace_manager.read_file_for_context(rel, start_line=1, end_line=350)
            blocks.append(
                f"\n### File: `{info['path']}` (lines {info['start_line']}-{info['end_line']}/{info['total_lines']})\n```\n{info['raw_content']}\n```"
            )
        except Exception as exc:
            blocks.append(f"\n### File: `{rel}`\n(읽기 실패: {exc})")
    return "\n".join(blocks)


@app.post("/api/sessions/{session_id}/stop")
async def stop_generation(session_id: str) -> dict[str, Any]:
    had = session_id in active_streams
    _cancel_session_streams(session_id)
    return {"stopping": had}


@app.post("/api/sessions/{session_id}/chat")
async def chat_stream(session_id: str, req: ChatRequest) -> StreamingResponse:
    if not vault.is_unlocked():
        raise VaultLockedError("Vault가 잠겨 있습니다.")

    state = model_manager.get_state()
    if state["status"] != "ready":
        raise HTTPException(
            status_code=503,
            detail="로컬 모델이 메모리에 올라가 있지 않습니다. 상단의 'Load' 버튼을 눌러 모델을 적재해 주세요.",
        )

    session = storage.get_session(session_id)
    attached_files = [f.strip() for f in req.attached_files if f.strip()]

    if req.regenerate:
        while session["messages"] and session["messages"][-1].get("role") == "assistant":
            session["messages"].pop()
        if not session["messages"] or session["messages"][-1].get("role") != "user":
            raise HTTPException(status_code=400, detail="재생성할 사용자 메시지가 없습니다.")
        if not attached_files:
            attached_files = list(session["messages"][-1].get("attached_files") or [])
    else:
        content = req.content.strip()
        if not content:
            raise HTTPException(status_code=400, detail="메시지 내용을 입력하세요.")
        user_msg: dict[str, Any] = {
            "id": new_message_id(),
            "role": "user",
            "content": content,
            "reasoning": "",
            "ts": now_iso(),
        }
        if attached_files:
            user_msg["attached_files"] = attached_files
        session["messages"].append(user_msg)
        if session.get("title") in ("새 대화", "", "제목 없는 대화"):
            session["title"] = _auto_title(content)

    merged_params = {**DEFAULT_GEN_PARAMS, **(session.get("params") or {}), **(req.params or {})}
    session["params"] = merged_params
    session["model"] = model_manager.model_id or session.get("model") or DEFAULT_MODEL_ID

    storage.save_session(session)

    agent_mode = req.agent_mode
    active_tools = get_tool_schemas(agent_mode)
    extra_sys = build_agent_system_instructions(agent_mode, str(workspace_manager.root))
    system_prompt = storage.get_persona_prompt(session.get("persona", "default"))
    llm_messages = build_llm_messages(system_prompt, session["messages"], extra_system=extra_sys)

    # Inject attached file contents into the final user turn sent to the LLM
    if attached_files and llm_messages and llm_messages[-1]["role"] == "user":
        llm_messages[-1]["content"] += _build_attached_files_context(attached_files)

    _cancel_session_streams(session_id)
    cancel_event = asyncio.Event()
    active_streams[session_id] = cancel_event

    async def event_generator() -> AsyncGenerator[str, None]:
        assistant_id = new_message_id()
        reasoning_buf: list[str] = []
        content_buf: list[str] = []
        tool_events: list[dict[str, Any]] = []
        last_metrics: dict[str, Any] = {}
        total_completion_tokens = 0
        cancelled = False

        model_manager.begin_generation()
        try:
            yield _sse(
                "meta",
                {
                    "session_id": session_id,
                    "title": session["title"],
                    "assistant_id": assistant_id,
                    "agent_mode": agent_mode,
                    "workspace": str(workspace_manager.root),
                },
            )

            for step in range(MAX_AGENT_STEPS):
                if cancel_event.is_set():
                    cancelled = True
                    break

                step_tool_calls: list[dict[str, Any]] = []
                step_content_buf: list[str] = []

                async for ev in stream_chat_completion(
                    messages=llm_messages,
                    params=merged_params,
                    tools=active_tools if active_tools else None,
                    cancel_event=cancel_event,
                ):
                    ev_type = ev["type"]
                    if ev_type == "reasoning":
                        reasoning_buf.append(ev["text"])
                        yield _sse("reasoning", {"text": ev["text"]})
                    elif ev_type == "content":
                        step_content_buf.append(ev["text"])
                        content_buf.append(ev["text"])
                        yield _sse("content", {"text": ev["text"]})
                    elif ev_type == "tool_call":
                        step_tool_calls.append(
                            {
                                "id": f"call_{secrets.token_hex(4)}",
                                "name": ev["name"],
                                "arguments": ev.get("arguments") or {},
                            }
                        )
                    elif ev_type == "keepalive":
                        yield _sse("keepalive", {"detail": ev["detail"]})
                    elif ev_type == "done":
                        last_metrics = ev.get("metrics") or {}
                        total_completion_tokens += int(
                            last_metrics.get("completion_tokens") or 0
                        )
                        cancelled = bool(ev.get("cancelled", False))

                if cancelled or not step_tool_calls or not active_tools:
                    break

                # Append assistant's tool call message (gpt-oss Harmony template expects dict arguments!)
                llm_messages.append(
                    {
                        "role": "assistant",
                        "content": "".join(step_content_buf).strip(),
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": tc["arguments"],
                                },
                            }
                            for tc in step_tool_calls
                        ],
                    }
                )

                # Execute each requested tool (with human-in-the-loop approval when needed)
                for tc in step_tool_calls:
                    if cancel_event.is_set():
                        cancelled = True
                        break

                    tc_id = tc["id"]
                    t_name = tc["name"]
                    t_args = tc["arguments"]
                    preview = workspace_manager.prepare_tool_preview(t_name, t_args)
                    needs_approval = workspace_manager.requires_approval(
                        agent_mode, t_name, t_args
                    )

                    yield _sse(
                        "tool_start",
                        {
                            "id": tc_id,
                            "name": t_name,
                            "arguments": t_args,
                            "summary": preview["summary"],
                            "diff": preview["diff"],
                            "is_write": preview["is_write"],
                            "high_risk": preview["high_risk"],
                            "requires_approval": needs_approval,
                        },
                    )

                    approved = True
                    if needs_approval:
                        approval_id = f"apr_{secrets.token_hex(6)}"
                        pa = PendingApproval(
                            approval_id=approval_id,
                            session_id=session_id,
                            tool_name=t_name,
                        )
                        pending_approvals[approval_id] = pa
                        yield _sse(
                            "approval_required",
                            {
                                "approval_id": approval_id,
                                "tool_call_id": tc_id,
                                "name": t_name,
                                "arguments": t_args,
                                "summary": preview["summary"],
                                "diff": preview["diff"],
                                "high_risk": preview["high_risk"],
                            },
                        )
                        try:
                            await pa.event.wait()
                            approved = bool(pa.approved) and not cancel_event.is_set()
                        finally:
                            pending_approvals.pop(approval_id, None)

                    if cancel_event.is_set():
                        cancelled = True
                        break

                    if approved:
                        result = await workspace_manager.execute_tool(
                            agent_mode, t_name, t_args
                        )
                    else:
                        result = {
                            "ok": False,
                            "output": "사용자가 이 도구 실행을 거절했습니다. 수정/실행하지 말고 사용자에게 설명하거나 대안을 물어보세요.",
                            "diff": preview["diff"],
                        }

                    tool_record = {
                        "id": tc_id,
                        "name": t_name,
                        "arguments": t_args,
                        "summary": preview["summary"],
                        "ok": bool(result["ok"]),
                        "approved": approved,
                        "output": str(result["output"])[:4000],
                        "diff": str(result.get("diff") or "")[:6000],
                    }
                    tool_events.append(tool_record)

                    yield _sse("tool_result", tool_record)

                    llm_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "name": t_name,
                            "content": result["output"],
                        }
                    )

                if cancelled:
                    break
                if content_buf and not content_buf[-1].endswith("\n"):
                    content_buf.append("\n\n")
                    yield _sse("content", {"text": "\n\n"})

            final_content = "".join(content_buf).strip()
            final_reasoning = "".join(reasoning_buf).strip()
            if not final_content and (final_reasoning or tool_events) and cancelled:
                final_content = "*(생성이 중단되었습니다)*"

            if total_completion_tokens > 0:
                last_metrics["completion_tokens"] = total_completion_tokens

            if final_content or final_reasoning or tool_events:
                assistant_msg: dict[str, Any] = {
                    "id": assistant_id,
                    "role": "assistant",
                    "content": final_content,
                    "reasoning": final_reasoning,
                    "ts": now_iso(),
                    "metrics": last_metrics,
                }
                if tool_events:
                    assistant_msg["tool_events"] = tool_events

                if vault.is_unlocked():
                    latest = storage.get_session(session_id)
                    latest["messages"].append(assistant_msg)
                    storage.save_session(latest)
                    git_sync.schedule_sync()

            yield _sse(
                "done",
                {
                    "session_id": session_id,
                    "title": session["title"],
                    "assistant_id": assistant_id,
                    "cancelled": cancelled,
                    "metrics": last_metrics,
                    "tool_count": len(tool_events),
                },
            )

        except Exception as exc:
            yield _sse("error", {"detail": str(exc)})
        finally:
            model_manager.end_generation()
            if active_streams.get(session_id) is cancel_event:
                active_streams.pop(session_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ----- Personas API -----

@app.get("/api/personas")
async def list_personas() -> dict[str, Any]:
    return {"personas": storage.list_personas()}


@app.put("/api/personas/{name}")
async def save_persona(name: str, req: PersonaSaveRequest) -> dict[str, Any]:
    persona = storage.save_persona(name, req.content)
    git_sync.schedule_sync()
    return persona


@app.delete("/api/personas/{name}")
async def delete_persona(name: str) -> dict[str, Any]:
    deleted = storage.delete_persona(name)
    if not deleted:
        raise HTTPException(status_code=404, detail="페르소나를 찾을 수 없습니다.")
    git_sync.schedule_sync()
    return {"deleted": True, "name": name}


# ----- Git Sync API -----

@app.get("/api/git/status")
async def get_git_status() -> dict[str, Any]:
    return await git_sync.get_status()


@app.post("/api/git/sync")
async def trigger_git_sync() -> dict[str, Any]:
    return await git_sync.sync_now(reason="manual")


@app.post("/api/git/pull")
async def trigger_git_pull() -> dict[str, Any]:
    return await git_sync.pull_now()


@app.post("/api/git/settings")
async def update_git_settings(req: GitSettingsRequest) -> dict[str, Any]:
    git_sync.auto_sync = req.auto_sync
    return await git_sync.get_status()


# ----- Frontend Static Files -----

@app.get("/")
async def serve_index() -> FileResponse:
    return FileResponse(
        FRONTEND_DIR / "index.html",
        headers={"Cache-Control": "no-cache"},
    )


if FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
