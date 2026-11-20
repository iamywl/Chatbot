from __future__ import annotations

import asyncio
import json
import urllib.parse
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
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
from backend.llm_client import build_llm_messages, stream_chat_completion
from backend.model_manager import ModelManager
from backend.storage import EncryptedSessionStorage, new_message_id, now_iso

# Singletons for local personal desktop app
vault = CryptoVault()
storage = EncryptedSessionStorage(vault)
model_manager = ModelManager()
git_sync = GitSyncWorker()

# Track active generation cancellation events per session
active_streams: dict[str, asyncio.Event] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    model_manager.start_background_tasks()
    try:
        yield
    finally:
        for ev in active_streams.values():
            ev.set()
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


class PersonaSaveRequest(BaseModel):
    content: str


class GitSettingsRequest(BaseModel):
    auto_sync: bool


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
    # Stop any ongoing chat streams before wiping key
    for ev in active_streams.values():
        ev.set()
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
    for ev in active_streams.values():
        ev.set()
    return await model_manager.unload()


@app.post("/api/model/settings")
async def update_model_settings(req: ModelSettingsRequest) -> dict[str, Any]:
    model_manager.set_idle_unload_minutes(req.idle_unload_minutes)
    return model_manager.get_state()


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
    if session_id in active_streams:
        active_streams[session_id].set()
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


# ----- Chat Streaming API -----

def _auto_title(text: str) -> str:
    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "새 대화")
    cleaned = first_line.lstrip("#>-* ").strip()
    if len(cleaned) > 38:
        return cleaned[:38].rstrip() + "…"
    return cleaned or "새 대화"


def _sse(event_type: str, payload: dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/api/sessions/{session_id}/stop")
async def stop_generation(session_id: str) -> dict[str, Any]:
    ev = active_streams.get(session_id)
    if ev is not None:
        ev.set()
        return {"stopping": True}
    return {"stopping": False}


@app.post("/api/sessions/{session_id}/chat")
async def chat_stream(session_id: str, req: ChatRequest) -> StreamingResponse:
    if not vault.is_unlocked():
        raise VaultLockedError("Vault가 잠겨 있습니다.")

    state = model_manager.get_state()
    if state["status"] != "ready":
        raise HTTPException(
            status_code=503,
            detail="로컬 모델이 메모리에 올라가 있지 않습니다. 상단의 'Load Model' 버튼을 눌러 모델을 적재해 주세요.",
        )

    session = storage.get_session(session_id)

    if req.regenerate:
        # Remove trailing assistant message(s) so we regenerate from the last user prompt
        while session["messages"] and session["messages"][-1].get("role") == "assistant":
            session["messages"].pop()
        if not session["messages"] or session["messages"][-1].get("role") != "user":
            raise HTTPException(status_code=400, detail="재생성할 사용자 메시지가 없습니다.")
    else:
        content = req.content.strip()
        if not content:
            raise HTTPException(status_code=400, detail="메시지 내용을 입력하세요.")
        user_msg = {
            "id": new_message_id(),
            "role": "user",
            "content": content,
            "reasoning": "",
            "ts": now_iso(),
        }
        session["messages"].append(user_msg)
        if session.get("title") in ("새 대화", "", "제목 없는 대화"):
            session["title"] = _auto_title(content)

    merged_params = {**DEFAULT_GEN_PARAMS, **(session.get("params") or {}), **(req.params or {})}
    session["params"] = merged_params
    session["model"] = model_manager.model_id or session.get("model") or DEFAULT_MODEL_ID

    # Save user prompt immediately in encrypted storage
    storage.save_session(session)

    system_prompt = storage.get_persona_prompt(session.get("persona", "default"))
    llm_messages = build_llm_messages(system_prompt, session["messages"])

    # Cancel any existing stream on the same session
    if session_id in active_streams:
        active_streams[session_id].set()
    cancel_event = asyncio.Event()
    active_streams[session_id] = cancel_event

    async def event_generator() -> AsyncGenerator[str, None]:
        assistant_id = new_message_id()
        reasoning_buf: list[str] = []
        content_buf: list[str] = []
        metrics: dict[str, Any] = {}
        cancelled = False

        model_manager.begin_generation()
        try:
            yield _sse(
                "meta",
                {
                    "session_id": session_id,
                    "title": session["title"],
                    "assistant_id": assistant_id,
                },
            )

            async for ev in stream_chat_completion(
                messages=llm_messages,
                params=merged_params,
                cancel_event=cancel_event,
            ):
                ev_type = ev["type"]
                if ev_type == "reasoning":
                    reasoning_buf.append(ev["text"])
                    yield _sse("reasoning", {"text": ev["text"]})
                elif ev_type == "content":
                    content_buf.append(ev["text"])
                    yield _sse("content", {"text": ev["text"]})
                elif ev_type == "keepalive":
                    yield _sse("keepalive", {"detail": ev["detail"]})
                elif ev_type == "done":
                    metrics = ev.get("metrics") or {}
                    cancelled = bool(ev.get("cancelled", False))

            final_content = "".join(content_buf).strip()
            final_reasoning = "".join(reasoning_buf).strip()

            # Edge case: if model stopped mid-reasoning before emitting final content
            if not final_content and final_reasoning and cancelled:
                final_content = "*(생성이 중단되었습니다)*"

            if final_content or final_reasoning:
                assistant_msg = {
                    "id": assistant_id,
                    "role": "assistant",
                    "content": final_content,
                    "reasoning": final_reasoning,
                    "ts": now_iso(),
                    "metrics": metrics,
                }
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
                    "metrics": metrics,
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
