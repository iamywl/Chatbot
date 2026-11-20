from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.config import BASE_DIR, DEFAULT_GIT_DEBOUNCE_SECONDS


class GitSyncWorker:
    """Async Git commit/push/pull manager restricted strictly to encrypted history/ and prompts/."""

    SYNC_PATHS = ("history", "prompts")

    def __init__(
        self,
        repo_dir: Path | None = None,
        debounce_seconds: int = DEFAULT_GIT_DEBOUNCE_SECONDS,
    ) -> None:
        self.repo_dir = repo_dir or BASE_DIR
        self.debounce_seconds = debounce_seconds
        self.auto_sync: bool = True

        self._lock = asyncio.Lock()
        self._debounce_task: asyncio.Task[None] | None = None

        self.syncing: bool = False
        self.last_sync_at: str | None = None
        self.last_result: str = "준비됨"
        self.last_error: str | None = None

    async def _run_git(self, *args: str, timeout: float = 45.0) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=str(self.repo_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return 124, "", "git 명령 실행 시간이 초과되었습니다."
        return (
            proc.returncode or 0,
            out_b.decode("utf-8", errors="replace").strip(),
            err_b.decode("utf-8", errors="replace").strip(),
        )

    async def is_git_repo(self) -> bool:
        code, out, _ = await self._run_git("rev-parse", "--is-inside-work-tree")
        return code == 0 and out == "true"

    async def get_status(self) -> dict[str, Any]:
        if not await self.is_git_repo():
            return {
                "is_repo": False,
                "branch": None,
                "remote_url": None,
                "pending_files": [],
                "pending_count": 0,
                "ahead": 0,
                "behind": 0,
                "auto_sync": self.auto_sync,
                "syncing": self.syncing,
                "last_sync_at": self.last_sync_at,
                "last_result": self.last_result,
                "last_error": self.last_error,
            }

        _, branch, _ = await self._run_git("rev-parse", "--abbrev-ref", "HEAD")
        _, remote_url, _ = await self._run_git("remote", "get-url", "origin")
        _, porcelain, _ = await self._run_git(
            "status", "--porcelain", "--", *self.SYNC_PATHS
        )
        pending_files = [ln.strip() for ln in porcelain.splitlines() if ln.strip()]

        ahead = 0
        behind = 0
        code_ab, ab_out, _ = await self._run_git(
            "rev-list", "--left-right", "--count", "@{upstream}...HEAD"
        )
        if code_ab == 0 and ab_out:
            parts = ab_out.split()
            if len(parts) == 2:
                behind, ahead = int(parts[0]), int(parts[1])

        return {
            "is_repo": True,
            "branch": branch or "main",
            "remote_url": remote_url or None,
            "pending_files": pending_files,
            "pending_count": len(pending_files),
            "ahead": ahead,
            "behind": behind,
            "auto_sync": self.auto_sync,
            "syncing": self.syncing,
            "last_sync_at": self.last_sync_at,
            "last_result": self.last_result,
            "last_error": self.last_error,
        }

    def schedule_sync(self) -> None:
        if not self.auto_sync:
            return
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        self._debounce_task = asyncio.create_task(self._debounced_sync())

    async def _debounced_sync(self) -> None:
        try:
            await asyncio.sleep(self.debounce_seconds)
            await self.sync_now(reason="auto")
        except asyncio.CancelledError:
            pass

    async def sync_now(self, reason: str = "manual") -> dict[str, Any]:
        """Stage ONLY history/ and prompts/, commit with neutral privacy-safe message, and push."""
        async with self._lock:
            if not await self.is_git_repo():
                self.last_error = "Git 저장소가 초기화되어 있지 않습니다."
                return await self.get_status()

            self.syncing = True
            self.last_error = None
            try:
                # Stage only encrypted history/ and prompts/
                code, _, err = await self._run_git("add", "-A", "--", *self.SYNC_PATHS)
                if code != 0:
                    raise RuntimeError(f"git add 실패: {err}")

                # Check if anything is staged
                diff_code, _, _ = await self._run_git(
                    "diff", "--cached", "--quiet", "--", *self.SYNC_PATHS
                )
                committed = False
                if diff_code != 0:
                    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
                    # Privacy-safe commit message (never leaks chat title/content)
                    msg = f"vault: sync encrypted chats ({stamp})"
                    c_code, _, c_err = await self._run_git("commit", "-m", msg)
                    if c_code != 0:
                        raise RuntimeError(f"git commit 실패: {c_err}")
                    committed = True

                # Push if remote origin exists
                r_code, remote_url, _ = await self._run_git("remote", "get-url", "origin")
                if r_code == 0 and remote_url:
                    _, branch, _ = await self._run_git("rev-parse", "--abbrev-ref", "HEAD")
                    branch = branch or "main"
                    p_code, _, p_err = await self._run_git(
                        "push", "-u", "origin", branch, timeout=60.0
                    )
                    if p_code != 0:
                        raise RuntimeError(f"git push 실패: {p_err}")
                    self.last_sync_at = datetime.now().astimezone().isoformat(
                        timespec="seconds"
                    )
                    self.last_result = (
                        f"암호화 커밋 및 Push 완료 ({reason})"
                        if committed
                        else "이미 원격과 최신 상태입니다"
                    )
                else:
                    self.last_sync_at = datetime.now().astimezone().isoformat(
                        timespec="seconds"
                    )
                    self.last_result = (
                        "로컬 커밋 완료 (원격 origin 미설정)"
                        if committed
                        else "변경 사항 없음"
                    )

            except Exception as exc:
                self.last_error = str(exc)
                self.last_result = "동기화 실패"
            finally:
                self.syncing = False

            return await self.get_status()

    async def pull_now(self) -> dict[str, Any]:
        """Pull encrypted history from remote origin (fast-forward / rebase safe)."""
        async with self._lock:
            if not await self.is_git_repo():
                self.last_error = "Git 저장소가 초기화되어 있지 않습니다."
                return await self.get_status()

            self.syncing = True
            self.last_error = None
            try:
                _, branch, _ = await self._run_git("rev-parse", "--abbrev-ref", "HEAD")
                branch = branch or "main"
                code, out, err = await self._run_git(
                    "pull", "--rebase", "--autostash", "origin", branch, timeout=60.0
                )
                if code != 0:
                    raise RuntimeError(f"git pull 실패: {err or out}")
                self.last_sync_at = datetime.now().astimezone().isoformat(
                    timespec="seconds"
                )
                self.last_result = "원격 암호화 기록 Pull 완료"
            except Exception as exc:
                self.last_error = str(exc)
                self.last_result = "Pull 실패"
            finally:
                self.syncing = False

            return await self.get_status()

    async def shutdown(self) -> None:
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
            try:
                await self._debounce_task
            except asyncio.CancelledError:
                pass
