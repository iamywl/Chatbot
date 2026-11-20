from __future__ import annotations

import json
import os
import re
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from backend.config import DEFAULT_GEN_PARAMS, DEFAULT_MODEL_ID, HISTORY_DIR, PROMPTS_DIR
from backend.crypto_vault import CryptoVault, VaultDecryptError

TURN_MARKER_RE = re.compile(r"^<!-- CHATBOT_TURN:(\{.*?\}) -->\s*$", re.MULTILINE)
REASONING_BLOCK_RE = re.compile(
    r"<!-- CHATBOT_REASONING_START -->\s*<details>\s*<summary>.*?</summary>\s*\n?(.*?)\n?\s*</details>\s*<!-- CHATBOT_REASONING_END -->\s*",
    re.DOTALL,
)
HEADING_RE = re.compile(r"^##\s+(User|Assistant)\s*\n?", re.IGNORECASE)
PERSONA_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,48}$")
SESSION_ID_RE = re.compile(r"^[0-9]{8}_[0-9]{6}_[a-f0-9]{6}$")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def new_session_id() -> str:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    suffix = secrets.token_hex(3)
    return f"{stamp}_{suffix}"


def new_message_id() -> str:
    return f"msg_{secrets.token_hex(4)}"


def serialize_session_markdown(session: dict[str, Any], clean_export: bool = False) -> str:
    frontmatter = {
        "session_id": session["session_id"],
        "title": session.get("title", "새 대화"),
        "model": session.get("model", DEFAULT_MODEL_ID),
        "persona": session.get("persona", "default"),
        "created_at": session.get("created_at", now_iso()),
        "updated_at": session.get("updated_at", now_iso()),
        "params": session.get("params", dict(DEFAULT_GEN_PARAMS)),
    }
    fm_yaml = yaml.safe_dump(frontmatter, allow_unicode=True, sort_keys=False).strip()
    parts = [f"---\n{fm_yaml}\n---\n"]

    for msg in session.get("messages", []):
        role = msg.get("role", "user")
        heading = "## User" if role == "user" else "## Assistant"
        content = (msg.get("content") or "").strip()
        reasoning = (msg.get("reasoning") or "").strip()

        block_lines: list[str] = []
        if not clean_export:
            turn_meta = {
                "role": role,
                "id": msg.get("id") or new_message_id(),
                "ts": msg.get("ts") or now_iso(),
            }
            if msg.get("metrics"):
                turn_meta["metrics"] = msg["metrics"]
            if msg.get("tool_events"):
                turn_meta["tool_events"] = msg["tool_events"]
            if msg.get("attached_files"):
                turn_meta["attached_files"] = msg["attached_files"]
            block_lines.append(
                f"<!-- CHATBOT_TURN:{json.dumps(turn_meta, ensure_ascii=False)} -->"
            )

        block_lines.append(heading)

        if clean_export and role == "assistant" and msg.get("tool_events"):
            block_lines.append("<details>")
            block_lines.append(f"<summary>Agent Tool Calls ({len(msg['tool_events'])})</summary>")
            block_lines.append("")
            for te in msg["tool_events"]:
                status_icon = "✅" if te.get("ok") else ("🚫" if te.get("approved") is False else "⚠️")
                block_lines.append(f"- {status_icon} `{te.get('name')}` — {te.get('summary', '')}")
            block_lines.append("")
            block_lines.append("</details>")
            block_lines.append("")

        if role == "assistant" and reasoning:
            if not clean_export:
                block_lines.append("<!-- CHATBOT_REASONING_START -->")
            block_lines.append("<details>")
            block_lines.append("<summary>Reasoning Process</summary>")
            block_lines.append("")
            block_lines.append(reasoning)
            block_lines.append("")
            block_lines.append("</details>")
            if not clean_export:
                block_lines.append("<!-- CHATBOT_REASONING_END -->")
            block_lines.append("")

        block_lines.append(content)
        parts.append("\n".join(block_lines).rstrip() + "\n")

    return "\n".join(parts).rstrip() + "\n"


def parse_session_markdown(markdown_text: str) -> dict[str, Any]:
    text = markdown_text.lstrip("\ufeff")
    meta: dict[str, Any] = {}
    body = text

    if text.startswith("---\n"):
        end_idx = text.find("\n---\n", 4)
        if end_idx != -1:
            fm_raw = text[4:end_idx]
            meta = yaml.safe_load(fm_raw) or {}
            body = text[end_idx + 5 :]

    messages: list[dict[str, Any]] = []
    matches = list(TURN_MARKER_RE.finditer(body))

    for idx, match in enumerate(matches):
        try:
            turn_meta = json.loads(match.group(1))
        except json.JSONDecodeError:
            turn_meta = {"role": "user", "id": new_message_id(), "ts": ""}

        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        chunk = body[start:end].strip()
        chunk = HEADING_RE.sub("", chunk, count=1).strip()

        reasoning = ""
        r_match = REASONING_BLOCK_RE.search(chunk)
        if r_match:
            reasoning = r_match.group(1).strip()
            chunk = (chunk[: r_match.start()] + chunk[r_match.end() :]).strip()

        msg: dict[str, Any] = {
            "id": turn_meta.get("id") or new_message_id(),
            "role": turn_meta.get("role", "user"),
            "content": chunk,
            "reasoning": reasoning,
            "ts": turn_meta.get("ts", ""),
        }
        if turn_meta.get("metrics"):
            msg["metrics"] = turn_meta["metrics"]
        if turn_meta.get("tool_events"):
            msg["tool_events"] = turn_meta["tool_events"]
        if turn_meta.get("attached_files"):
            msg["attached_files"] = turn_meta["attached_files"]
        messages.append(msg)

    return {
        "session_id": str(meta.get("session_id", "")),
        "title": str(meta.get("title", "제목 없는 대화")),
        "model": str(meta.get("model", DEFAULT_MODEL_ID)),
        "persona": str(meta.get("persona", "default")),
        "created_at": str(meta.get("created_at", "")),
        "updated_at": str(meta.get("updated_at", "")),
        "params": {**DEFAULT_GEN_PARAMS, **(meta.get("params") or {})},
        "messages": messages,
    }


class EncryptedSessionStorage:
    """Stores every chat session as AES-256-GCM encrypted Markdown under history/YYYY-MM/."""

    def __init__(
        self,
        vault: CryptoVault,
        history_dir: Path | None = None,
        prompts_dir: Path | None = None,
    ) -> None:
        self.vault = vault
        self.history_dir = history_dir or HISTORY_DIR
        self.prompts_dir = prompts_dir or PROMPTS_DIR
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)

    def _validate_session_id(self, session_id: str) -> str:
        if not SESSION_ID_RE.match(session_id):
            raise ValueError(f"Invalid session_id: {session_id}")
        return session_id

    def _find_session_file(self, session_id: str) -> Path | None:
        self._validate_session_id(session_id)
        # Fast path: derive YYYY-MM from session_id prefix
        month_folder = f"{session_id[0:4]}-{session_id[4:6]}"
        expected = self.history_dir / month_folder / f"{session_id}.md.enc"
        if expected.is_file():
            return expected
        # Fallback scan
        for p in self.history_dir.rglob(f"{session_id}.md.enc"):
            if p.is_file():
                return p
        return None

    def _session_path_for_id(self, session_id: str) -> Path:
        self._validate_session_id(session_id)
        month_folder = f"{session_id[0:4]}-{session_id[4:6]}"
        target_dir = self.history_dir / month_folder
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir / f"{session_id}.md.enc"

    def count_encrypted_files(self) -> int:
        return sum(1 for _ in self.history_dir.rglob("*.md.enc"))

    def create_session(
        self,
        title: str = "새 대화",
        persona: str = "default",
        model: str = DEFAULT_MODEL_ID,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ts = now_iso()
        session = {
            "session_id": new_session_id(),
            "title": title.strip() or "새 대화",
            "model": model or DEFAULT_MODEL_ID,
            "persona": persona or "default",
            "created_at": ts,
            "updated_at": ts,
            "params": {**DEFAULT_GEN_PARAMS, **(params or {})},
            "messages": [],
        }
        self.save_session(session)
        return session

    def save_session(self, session: dict[str, Any]) -> dict[str, Any]:
        session_id = self._validate_session_id(session["session_id"])
        session["updated_at"] = now_iso()
        markdown = serialize_session_markdown(session, clean_export=False)
        envelope = self.vault.encrypt_session(
            session_id=session_id,
            plaintext_markdown=markdown,
            updated_at=session["updated_at"],
        )
        path = self._session_path_for_id(session_id)
        tmp_path = path.with_suffix(".enc.tmp")
        tmp_path.write_text(
            json.dumps(envelope, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_path, path)
        return session

    def get_session(self, session_id: str) -> dict[str, Any]:
        path = self._find_session_file(session_id)
        if path is None:
            raise FileNotFoundError(f"세션을 찾을 수 없습니다: {session_id}")
        envelope = json.loads(path.read_text(encoding="utf-8"))
        markdown = self.vault.decrypt_session(envelope)
        session = parse_session_markdown(markdown)
        session["session_id"] = session_id
        session["file_relpath"] = str(path.relative_to(self.history_dir.parent))
        return session

    def list_sessions(
        self, query: str = "", persona: str = ""
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        q = query.strip().lower()
        persona_filter = persona.strip()

        for path in self.history_dir.rglob("*.md.enc"):
            try:
                envelope = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            sid = envelope.get("session_id") or path.name.replace(".md.enc", "")
            try:
                markdown = self.vault.decrypt_session(envelope)
                session = parse_session_markdown(markdown)
            except VaultDecryptError:
                if not q and not persona_filter:
                    results.append(
                        {
                            "session_id": sid,
                            "title": "🔒 다른 키로 암호화된 세션",
                            "persona": "",
                            "model": "",
                            "created_at": "",
                            "updated_at": envelope.get("updated_at", ""),
                            "message_count": 0,
                            "preview": "현재 키로 복호화할 수 없습니다.",
                            "decrypt_error": True,
                        }
                    )
                continue

            if persona_filter and session.get("persona") != persona_filter:
                continue

            if q:
                haystack = (
                    session.get("title", "")
                    + "\n"
                    + "\n".join(
                        (m.get("content", "") + "\n" + m.get("reasoning", ""))
                        for m in session.get("messages", [])
                    )
                ).lower()
                if q not in haystack:
                    continue

            messages = session.get("messages", [])
            preview = ""
            if messages:
                last_msg = messages[-1]
                preview = (last_msg.get("content") or "").replace("\n", " ")[:90]

            results.append(
                {
                    "session_id": sid,
                    "title": session.get("title", "새 대화"),
                    "persona": session.get("persona", "default"),
                    "model": session.get("model", DEFAULT_MODEL_ID),
                    "created_at": session.get("created_at", ""),
                    "updated_at": session.get("updated_at")
                    or envelope.get("updated_at", ""),
                    "message_count": len(messages),
                    "preview": preview,
                    "decrypt_error": False,
                }
            )

        results.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
        return results

    def update_session_meta(
        self,
        session_id: str,
        title: str | None = None,
        persona: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        session = self.get_session(session_id)
        if title is not None and title.strip():
            session["title"] = title.strip()[:120]
        if persona is not None and persona.strip():
            session["persona"] = persona.strip()
        if params is not None:
            session["params"] = {**session.get("params", DEFAULT_GEN_PARAMS), **params}
        return self.save_session(session)

    def delete_session(self, session_id: str) -> bool:
        path = self._find_session_file(session_id)
        if path is None:
            return False
        path.unlink()
        return True

    def fork_session(
        self, session_id: str, up_to_message_id: str | None = None
    ) -> dict[str, Any]:
        original = self.get_session(session_id)
        messages = original.get("messages", [])
        if up_to_message_id:
            cut_idx = None
            for i, m in enumerate(messages):
                if m.get("id") == up_to_message_id:
                    cut_idx = i
                    break
            if cut_idx is not None:
                messages = messages[: cut_idx + 1]

        ts = now_iso()
        cloned_messages = [
            {**m, "id": new_message_id()} for m in messages
        ]
        forked = {
            "session_id": new_session_id(),
            "title": f"{original.get('title', '대화')} (분기)",
            "model": original.get("model", DEFAULT_MODEL_ID),
            "persona": original.get("persona", "default"),
            "created_at": ts,
            "updated_at": ts,
            "params": dict(original.get("params", DEFAULT_GEN_PARAMS)),
            "messages": cloned_messages,
        }
        self.save_session(forked)
        return forked

    def export_session_markdown(self, session_id: str) -> tuple[str, str]:
        session = self.get_session(session_id)
        md = serialize_session_markdown(session, clean_export=True)
        safe_title = re.sub(r"[^\w가-힣-]+", "_", session.get("title", "chat")).strip("_")[:40]
        filename = f"{session_id}_{safe_title or 'chat'}.md"
        return filename, md

    # ----- Personas (prompts/*.md) -----

    def _validate_persona_name(self, name: str) -> str:
        clean = name.strip()
        if not PERSONA_NAME_RE.match(clean):
            raise ValueError(
                "페르소나 이름은 영문, 숫자, 하이픈(-), 언더스코어(_)만 사용할 수 있습니다."
            )
        return clean

    def list_personas(self) -> list[dict[str, str]]:
        personas: list[dict[str, str]] = []
        for p in sorted(self.prompts_dir.glob("*.md")):
            content = p.read_text(encoding="utf-8")
            first_line = next(
                (ln.strip() for ln in content.splitlines() if ln.strip()), ""
            )
            personas.append(
                {
                    "name": p.stem,
                    "content": content,
                    "summary": first_line[:80],
                }
            )
        # Put 'default' first
        personas.sort(key=lambda x: (x["name"] != "default", x["name"]))
        return personas

    def get_persona_prompt(self, name: str) -> str:
        if not name or not PERSONA_NAME_RE.match(name):
            name = "default"
        path = self.prompts_dir / f"{name}.md"
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
        default_path = self.prompts_dir / "default.md"
        if default_path.is_file():
            return default_path.read_text(encoding="utf-8").strip()
        return ""

    def save_persona(self, name: str, content: str) -> dict[str, str]:
        clean = self._validate_persona_name(name)
        path = self.prompts_dir / f"{clean}.md"
        path.write_text(content.rstrip() + "\n", encoding="utf-8")
        first_line = next(
            (ln.strip() for ln in content.splitlines() if ln.strip()), ""
        )
        return {"name": clean, "content": content, "summary": first_line[:80]}

    def delete_persona(self, name: str) -> bool:
        clean = self._validate_persona_name(name)
        if clean == "default":
            raise ValueError("default 페르소나는 삭제할 수 없습니다.")
        path = self.prompts_dir / f"{clean}.md"
        if path.is_file():
            path.unlink()
            return True
        return False
