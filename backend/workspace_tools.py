from __future__ import annotations

import asyncio
import difflib
import fnmatch
import os
import re
from pathlib import Path
from typing import Any

from backend.config import BASE_DIR

IGNORED_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".pytest_cache",
    ".mypy_cache",
    ".next",
    "dist",
    "build",
    ".idea",
    ".DS_Store",
}

BLOCKED_FILENAMES = {
    ".env",
    "vault.json",
    "id_rsa",
    "id_ed25519",
    "credentials",
}

BLOCKED_PATTERNS = (
    ".env.*",
    "*.key",
    "*.pem",
    "*.p12",
    "*.pfx",
    "*.md.enc",
)

DANGEROUS_CMD_PATTERNS = (
    re.compile(r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*f|\brm\s+-rf\b"),
    re.compile(r"\bsudo\b"),
    re.compile(r"\bmkfs\b|\bdd\s+if="),
    re.compile(r"\bshutdown\b|\breboot\b|\bhalt\b"),
    re.compile(r"\bgit\s+push\s+.*--force|\bgit\s+reset\s+--hard\b"),
    re.compile(r"\bcurl\b.*\|\s*(sh|bash|zsh)\b"),
    re.compile(r"\bchmod\s+-R\s+777\b"),
)

MAX_READ_BYTES = 256 * 1024
MAX_OUTPUT_CHARS = 16000
MAX_READ_LINES = 400
COMMAND_TIMEOUT_SEC = 30.0


class WorkspaceSecurityError(PermissionError):
    pass


def is_sensitive_path(path: Path) -> bool:
    name = path.name
    if name in BLOCKED_FILENAMES:
        return True
    for pat in BLOCKED_PATTERNS:
        if fnmatch.fnmatch(name, pat):
            return True
    parts = set(path.parts)
    if parts & {".ssh", ".aws", ".gnupg"}:
        return True
    # Protect .git/config (may contain credentials/tokens)
    if ".git" in parts and name in ("config", "credentials"):
        return True
    return False


def is_dangerous_command(command: str) -> bool:
    cmd = command.strip()
    return any(p.search(cmd) for p in DANGEROUS_CMD_PATTERNS)


def is_binary_bytes(sample: bytes) -> bool:
    return b"\x00" in sample[:4096]


def make_unified_diff(old_text: str, new_text: str, rel_path: str) -> str:
    diff_lines = list(
        difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=f"a/{rel_path}",
            tofile=f"b/{rel_path}",
            n=3,
        )
    )
    text = "".join(diff_lines)
    if len(text) > 12000:
        return text[:12000] + "\n... (diff truncated)\n"
    return text


def get_tool_schemas(agent_mode: str) -> list[dict[str, Any]]:
    """Return OpenAI-format tool schemas for the given agent_mode."""
    if agent_mode == "off":
        return []

    read_tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "list_dir",
                "description": "List files and subdirectories in a directory inside the active workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path inside the workspace (use '.' for workspace root).",
                        }
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a text file from the active workspace with 1-based line numbers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative file path inside the workspace.",
                        },
                        "start_line": {
                            "type": "integer",
                            "description": "1-based start line (optional, default 1).",
                        },
                        "end_line": {
                            "type": "integer",
                            "description": "1-based inclusive end line (optional, default 250).",
                        },
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_files",
                "description": "Search for a text or regex pattern across text files in the active workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Substring or regular expression to search for.",
                        },
                        "path": {
                            "type": "string",
                            "description": "Optional relative subdirectory to restrict search (default '.').",
                        },
                        "file_glob": {
                            "type": "string",
                            "description": "Optional filename glob filter, e.g. '*.py' or '*.js'.",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
    ]

    if agent_mode == "read_only":
        return read_tools

    write_and_exec_tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Create a new file or completely overwrite an existing file in the active workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative file path inside the workspace.",
                        },
                        "content": {
                            "type": "string",
                            "description": "Full text content to write to the file.",
                        },
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Replace an exact contiguous text block in an existing workspace file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative file path inside the workspace.",
                        },
                        "target": {
                            "type": "string",
                            "description": "Exact existing text to replace.",
                        },
                        "replacement": {
                            "type": "string",
                            "description": "New replacement text.",
                        },
                    },
                    "required": ["path", "target", "replacement"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Run a shell command inside the active workspace (e.g. pytest, git status, ls, python3, npm test).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Shell command to execute.",
                        },
                        "cwd": {
                            "type": "string",
                            "description": "Optional relative working directory inside the workspace (default '.').",
                        },
                    },
                    "required": ["command"],
                },
            },
        },
    ]

    return read_tools + write_and_exec_tools


READ_TOOL_NAMES = {"list_dir", "read_file", "search_files"}
WRITE_TOOL_NAMES = {"write_file", "edit_file", "run_command"}


class WorkspaceManager:
    """Manages the active local workspace directory, file explorer, and sandboxed agent tools."""

    def __init__(self, initial_root: Path | None = None) -> None:
        self.root: Path = (initial_root or BASE_DIR).resolve()
        self.recent_roots: list[str] = [str(self.root)]

    def set_workspace(self, path_str: str) -> dict[str, Any]:
        candidate = Path(path_str).expanduser().resolve()
        if not candidate.exists() or not candidate.is_dir():
            raise FileNotFoundError(f"디렉터리를 찾을 수 없습니다: {path_str}")
        # Prevent mounting filesystem root `/` directly for safety
        if candidate == Path("/"):
            raise WorkspaceSecurityError("루트 디렉터리(/) 전체는 워크스페이스로 마운트할 수 없습니다.")
        self.root = candidate
        s = str(candidate)
        if s in self.recent_roots:
            self.recent_roots.remove(s)
        self.recent_roots.insert(0, s)
        self.recent_roots = self.recent_roots[:8]
        return self.get_status()

    def get_status(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "name": self.root.name or str(self.root),
            "recent_roots": self.recent_roots,
        }

    async def pick_native_folder(self) -> dict[str, Any]:
        """Open macOS Finder folder picker dialog using osascript."""
        script = (
            f'POSIX path of (choose folder with prompt "워크스페이스 폴더를 선택하세요" '
            f'default location (POSIX file "{self.root}"))'
        )
        proc = await asyncio.create_subprocess_exec(
            "osascript",
            "-e",
            script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out_b, err_b = await proc.communicate()
        if proc.returncode != 0:
            err = err_b.decode("utf-8", errors="replace").strip()
            if "-128" in err or "User canceled" in err:
                return {"cancelled": True, "workspace": self.get_status()}
            raise RuntimeError(f"폴더 선택 창 오류: {err}")
        chosen = out_b.decode("utf-8", errors="replace").strip()
        if not chosen:
            return {"cancelled": True, "workspace": self.get_status()}
        ws = self.set_workspace(chosen)
        return {"cancelled": False, "workspace": ws}

    def resolve_safe_path(self, rel_or_abs: str, *, allow_nonexistent: bool = False) -> Path:
        raw = (rel_or_abs or ".").strip() or "."
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = self.root / p
        resolved = p.resolve()

        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceSecurityError(
                f"워크스페이스({self.root}) 외부 경로에는 접근할 수 없습니다: {rel_or_abs}"
            ) from exc

        if is_sensitive_path(resolved):
            raise WorkspaceSecurityError(
                f"보안 정책에 의해 민감 파일({resolved.name}) 접근이 차단되었습니다."
            )

        if not allow_nonexistent and not resolved.exists():
            raise FileNotFoundError(f"경로를 찾을 수 없습니다: {rel_or_abs}")
        return resolved

    def rel_display(self, path: Path) -> str:
        try:
            rel = path.relative_to(self.root)
            return "." if str(rel) == "" else str(rel)
        except ValueError:
            return str(path)

    # ----- File Explorer API -----

    def list_tree(self, rel_path: str = ".") -> list[dict[str, Any]]:
        target = self.resolve_safe_path(rel_path)
        if not target.is_dir():
            raise ValueError(f"디렉터리가 아닙니다: {rel_path}")

        entries: list[dict[str, Any]] = []
        for child in sorted(
            target.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())
        ):
            if child.name in IGNORED_DIRS:
                continue
            if is_sensitive_path(child):
                continue
            try:
                is_dir = child.is_dir()
                size = 0 if is_dir else child.stat().st_size
            except OSError:
                continue
            entries.append(
                {
                    "name": child.name,
                    "path": self.rel_display(child),
                    "is_dir": is_dir,
                    "size": size,
                }
            )
        return entries

    def read_file_for_context(
        self, rel_path: str, start_line: int = 1, end_line: int | None = None
    ) -> dict[str, Any]:
        target = self.resolve_safe_path(rel_path)
        if not target.is_file():
            raise ValueError(f"파일이 아닙니다: {rel_path}")
        size = target.stat().st_size
        if size > MAX_READ_BYTES * 4:
            raise ValueError(
                f"파일이 너무 큽니다 ({round(size / 1024)} KB). 최대 {MAX_READ_BYTES * 4 // 1024} KB까지 읽을 수 있습니다."
            )
        raw = target.read_bytes()
        if is_binary_bytes(raw):
            raise ValueError(f"바이너리 파일은 읽을 수 없습니다: {rel_path}")

        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()
        total_lines = len(lines)
        s_idx = max(1, int(start_line or 1))
        e_idx = min(total_lines, int(end_line or min(total_lines, s_idx + MAX_READ_LINES - 1)))
        if e_idx - s_idx + 1 > MAX_READ_LINES:
            e_idx = s_idx + MAX_READ_LINES - 1

        sliced = lines[s_idx - 1 : e_idx]
        numbered = "\n".join(f"{s_idx + i}: {ln}" for i, ln in enumerate(sliced))
        if len(numbered) > MAX_OUTPUT_CHARS:
            numbered = numbered[:MAX_OUTPUT_CHARS] + "\n... (output truncated)"

        return {
            "path": self.rel_display(target),
            "total_lines": total_lines,
            "start_line": s_idx,
            "end_line": e_idx,
            "content": numbered,
            "raw_content": "\n".join(sliced),
        }

    # ----- Agent Tool Preview (for Approval Cards) & Execution -----

    def prepare_tool_preview(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Compute approval metadata, risk level, and unified diff preview before executing."""
        info: dict[str, Any] = {
            "name": name,
            "is_write": name in WRITE_TOOL_NAMES,
            "high_risk": False,
            "summary": "",
            "diff": "",
        }
        try:
            if name == "list_dir":
                info["summary"] = f"디렉터리 조회: {args.get('path', '.')}"
            elif name == "read_file":
                info["summary"] = f"파일 읽기: {args.get('path', '')}"
            elif name == "search_files":
                info["summary"] = f"코드 검색: '{args.get('query', '')}' in {args.get('path', '.')}"
            elif name == "write_file":
                rel = str(args.get("path", ""))
                new_content = str(args.get("content", ""))
                target = self.resolve_safe_path(rel, allow_nonexistent=True)
                old_content = (
                    target.read_text(encoding="utf-8", errors="replace")
                    if target.is_file()
                    else ""
                )
                action = "덮어쓰기" if target.is_file() else "새 파일 생성"
                info["summary"] = f"{action}: {self.rel_display(target)}"
                info["diff"] = make_unified_diff(old_content, new_content, self.rel_display(target))
            elif name == "edit_file":
                rel = str(args.get("path", ""))
                old_str = str(args.get("target", ""))
                new_str = str(args.get("replacement", ""))
                target = self.resolve_safe_path(rel)
                old_content = target.read_text(encoding="utf-8", errors="replace")
                info["summary"] = f"파일 수정: {self.rel_display(target)}"
                if old_str and old_str in old_content:
                    updated = old_content.replace(old_str, new_str, 1)
                    info["diff"] = make_unified_diff(old_content, updated, self.rel_display(target))
                else:
                    info["diff"] = "⚠️ 대상 텍스트(target)를 파일에서 찾을 수 없습니다."
            elif name == "run_command":
                cmd = str(args.get("command", ""))
                cwd = str(args.get("cwd", "."))
                info["summary"] = f"터미널 실행 ({cwd}): {cmd}"
                info["high_risk"] = is_dangerous_command(cmd)
        except Exception as exc:
            info["diff"] = f"⚠️ 사전 검증 오류: {exc}"
        return info

    def requires_approval(self, agent_mode: str, name: str, args: dict[str, Any]) -> bool:
        if name in READ_TOOL_NAMES:
            return False
        if agent_mode == "auto":
            if name == "run_command" and is_dangerous_command(str(args.get("command", ""))):
                return True
            return False
        # In "approve" mode, all write/exec tools require explicit user approval
        return True

    async def execute_tool(
        self, agent_mode: str, name: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a sandboxed tool and return {ok, output, diff}."""
        if agent_mode == "off":
            return {"ok": False, "output": "에이전트 모드가 꺼져 있습니다.", "diff": ""}
        if agent_mode == "read_only" and name in WRITE_TOOL_NAMES:
            return {
                "ok": False,
                "output": f"읽기 전용 모드에서는 '{name}' 도구를 실행할 수 없습니다.",
                "diff": "",
            }

        try:
            if name == "list_dir":
                entries = self.list_tree(str(args.get("path", ".")))
                lines = [
                    f"{'[DIR]  ' if e['is_dir'] else '[FILE] '}{e['path']}"
                    + ("" if e["is_dir"] else f" ({e['size']} B)")
                    for e in entries[:200]
                ]
                out = "\n".join(lines) if lines else "(빈 디렉터리)"
                if len(entries) > 200:
                    out += f"\n... 외 {len(entries) - 200}개 항목"
                return {"ok": True, "output": out, "diff": ""}

            if name == "read_file":
                res = self.read_file_for_context(
                    str(args.get("path", "")),
                    start_line=int(args.get("start_line") or 1),
                    end_line=int(args["end_line"]) if args.get("end_line") else None,
                )
                header = f"File: {res['path']} (lines {res['start_line']}-{res['end_line']} of {res['total_lines']})\n"
                return {"ok": True, "output": header + res["content"], "diff": ""}

            if name == "search_files":
                query = str(args.get("query", "")).strip()
                if not query:
                    return {"ok": False, "output": "검색어(query)가 비어 있습니다.", "diff": ""}
                base = self.resolve_safe_path(str(args.get("path", ".")))
                file_glob = str(args.get("file_glob", "")).strip()
                out = await asyncio.to_thread(self._search_sync, base, query, file_glob)
                return {"ok": True, "output": out, "diff": ""}

            if name == "write_file":
                rel = str(args.get("path", ""))
                content = str(args.get("content", ""))
                target = self.resolve_safe_path(rel, allow_nonexistent=True)
                old_content = (
                    target.read_text(encoding="utf-8", errors="replace")
                    if target.is_file()
                    else ""
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                diff = make_unified_diff(old_content, content, self.rel_display(target))
                return {
                    "ok": True,
                    "output": f"파일 저장 완료: {self.rel_display(target)} ({len(content)} chars)",
                    "diff": diff,
                }

            if name == "edit_file":
                rel = str(args.get("path", ""))
                target_text = str(args.get("target", ""))
                replacement = str(args.get("replacement", ""))
                target = self.resolve_safe_path(rel)
                old_content = target.read_text(encoding="utf-8", errors="replace")
                if not target_text or target_text not in old_content:
                    return {
                        "ok": False,
                        "output": f"'{self.rel_display(target)}'에서 target 텍스트를 찾을 수 없습니다. read_file로 정확한 내용을 먼저 확인하세요.",
                        "diff": "",
                    }
                count = old_content.count(target_text)
                new_content = old_content.replace(target_text, replacement, 1)
                target.write_text(new_content, encoding="utf-8")
                diff = make_unified_diff(old_content, new_content, self.rel_display(target))
                note = f" (참고: 동일 문자열 {count}건 중 첫 번째만 치환됨)" if count > 1 else ""
                return {
                    "ok": True,
                    "output": f"파일 수정 완료: {self.rel_display(target)}{note}",
                    "diff": diff,
                }

            if name == "run_command":
                cmd = str(args.get("command", "")).strip()
                if not cmd:
                    return {"ok": False, "output": "실행할 명령어가 비어 있습니다.", "diff": ""}
                cwd_path = self.resolve_safe_path(str(args.get("cwd", ".")))
                if not cwd_path.is_dir():
                    raise ValueError(f"작업 디렉터리가 아닙니다: {args.get('cwd')}")
                proc = await asyncio.create_subprocess_exec(
                    "/bin/zsh",
                    "-lc",
                    cmd,
                    cwd=str(cwd_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                try:
                    out_b, _ = await asyncio.wait_for(
                        proc.communicate(), timeout=COMMAND_TIMEOUT_SEC
                    )
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    return {
                        "ok": False,
                        "output": f"명령 실행 시간이 {int(COMMAND_TIMEOUT_SEC)}초를 초과하여 중단되었습니다.",
                        "diff": "",
                    }
                text = out_b.decode("utf-8", errors="replace")
                if len(text) > MAX_OUTPUT_CHARS:
                    text = text[:MAX_OUTPUT_CHARS] + "\n... (output truncated)"
                code = proc.returncode or 0
                summary = f"$ {cmd}\n[exit code: {code}]\n{text}".strip()
                return {"ok": code == 0, "output": summary, "diff": ""}

            return {"ok": False, "output": f"알 수 없는 도구입니다: {name}", "diff": ""}

        except Exception as exc:
            return {"ok": False, "output": f"도구 실행 오류: {exc}", "diff": ""}

    def _search_sync(self, base: Path, query: str, file_glob: str) -> str:
        try:
            pattern = re.compile(query, re.IGNORECASE)
        except re.error:
            pattern = re.compile(re.escape(query), re.IGNORECASE)

        matches: list[str] = []
        files_to_scan = [base] if base.is_file() else []
        if base.is_dir():
            for root_str, dirnames, filenames in os.walk(base):
                dirnames[:] = [d for d in sorted(dirnames) if d not in IGNORED_DIRS]
                root_p = Path(root_str)
                for fn in sorted(filenames):
                    if file_glob and not fnmatch.fnmatch(fn, file_glob):
                        continue
                    fp = root_p / fn
                    if is_sensitive_path(fp):
                        continue
                    files_to_scan.append(fp)

        for fp in files_to_scan:
            if len(matches) >= 60:
                break
            try:
                if fp.stat().st_size > MAX_READ_BYTES:
                    continue
                raw = fp.read_bytes()
                if is_binary_bytes(raw):
                    continue
                rel = self.rel_display(fp)
                for lineno, line in enumerate(
                    raw.decode("utf-8", errors="replace").splitlines(), 1
                ):
                    if pattern.search(line):
                        matches.append(f"{rel}:{lineno}: {line.strip()[:160]}")
                        if len(matches) >= 60:
                            break
            except OSError:
                continue

        if not matches:
            return f"'{query}'에 대한 검색 결과가 없습니다."
        result = "\n".join(matches)
        if len(matches) >= 60:
            result += "\n... (최대 60건까지 표시됨)"
        return result
