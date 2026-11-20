from __future__ import annotations

import asyncio
import collections
import os
import signal
import time
from typing import Any

import httpx
import psutil

from backend.config import (
    DEFAULT_GEN_PARAMS,
    DEFAULT_IDLE_UNLOAD_MINUTES,
    DEFAULT_MODEL_ID,
    MLX_BASE_URL,
    MLX_HOST,
    MLX_PORT,
    PYTHON_BIN,
    discover_local_mlx_models,
    resolve_model_path,
)


class ModelManager:
    """Manages mlx_lm.server subprocess lifecycle (Load/Unload), RAM telemetry, and Idle Auto-Unload."""

    def __init__(self) -> None:
        self._proc: asyncio.subprocess.Process | None = None
        self._log_task: asyncio.Task[None] | None = None
        self._load_task: asyncio.Task[None] | None = None
        self._idle_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

        self.status: str = "unloaded"  # unloaded | loading | warming | ready | stopping | error
        self.status_detail: str = "모델이 메모리에 올라가 있지 않습니다."
        self.model_id: str | None = None
        self.model_path: str | None = None
        self.external_server: bool = False
        self.started_at: float | None = None
        self.last_activity_at: float | None = None
        self.last_error: str | None = None
        self.idle_unload_minutes: int = DEFAULT_IDLE_UNLOAD_MINUTES
        self._active_generations: int = 0

        self._logs: collections.deque[str] = collections.deque(maxlen=250)

    def _append_log(self, line: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self._logs.append(f"[{ts}] {line.rstrip()}")

    def get_logs(self, limit: int = 80) -> list[str]:
        items = list(self._logs)
        return items[-limit:]

    def start_background_tasks(self) -> None:
        if self._idle_task is None or self._idle_task.done():
            self._idle_task = asyncio.create_task(self._idle_monitor_loop())

    async def shutdown(self) -> None:
        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()
            try:
                await self._idle_task
            except asyncio.CancelledError:
                pass
        if self._proc is not None and self._proc.returncode is None:
            await self.unload()

    def begin_generation(self) -> None:
        self._active_generations += 1
        self.last_activity_at = time.monotonic()

    def end_generation(self) -> None:
        self._active_generations = max(0, self._active_generations - 1)
        self.last_activity_at = time.monotonic()

    def set_idle_unload_minutes(self, minutes: int) -> None:
        self.idle_unload_minutes = max(0, int(minutes))
        if self.status == "ready":
            self.last_activity_at = time.monotonic()

    async def _idle_monitor_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(15)
                if (
                    self.status == "ready"
                    and not self.external_server
                    and self.idle_unload_minutes > 0
                    and self._active_generations == 0
                    and self.last_activity_at is not None
                ):
                    idle_sec = time.monotonic() - self.last_activity_at
                    if idle_sec >= self.idle_unload_minutes * 60:
                        self._append_log(
                            f"Idle timeout ({self.idle_unload_minutes}m) reached. Auto-unloading model..."
                        )
                        await self.unload()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._append_log(f"Idle monitor error: {exc}")

    async def _probe_server_alive(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{MLX_BASE_URL}/v1/models")
                return resp.status_code == 200
        except Exception:
            return False

    async def _read_proc_output(self, proc: asyncio.subprocess.Process) -> None:
        if proc.stdout is None:
            return
        try:
            while True:
                raw = await proc.stdout.readline()
                if not raw:
                    break
                text = raw.decode("utf-8", errors="replace").rstrip()
                if text:
                    self._append_log(text)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self._append_log(f"Log reader error: {exc}")

    async def load(self, model_id: str | None = None) -> dict[str, Any]:
        """Start mlx_lm.server and warm up weights into unified memory."""
        async with self._lock:
            if self.status in ("loading", "warming"):
                return self.get_state()
            if self.status == "ready" and self._is_proc_alive():
                requested_id, _ = resolve_model_path(model_id or DEFAULT_MODEL_ID)
                if self.model_id == requested_id:
                    return self.get_state()
                # Switch model: unload current first
                await self._unload_locked()

            repo_id, snapshot_path = resolve_model_path(model_id or DEFAULT_MODEL_ID)
            self.model_id = repo_id
            self.model_path = snapshot_path
            self.last_error = None
            self.external_server = False

            # Check if an external mlx_lm.server is already running on this port
            if await self._probe_server_alive():
                self.external_server = True
                self.status = "ready"
                self.status_detail = "기존 실행 중인 mlx_lm.server에 연결되었습니다."
                self.started_at = time.monotonic()
                self.last_activity_at = time.monotonic()
                self._append_log(
                    f"Connected to existing mlx_lm.server at {MLX_BASE_URL}"
                )
                return self.get_state()

            self.status = "loading"
            self.status_detail = "mlx_lm.server 프로세스를 시작하는 중..."
            self._append_log(f"Starting mlx_lm.server for {repo_id} ...")

            cmd = [
                PYTHON_BIN,
                "-m",
                "mlx_lm",
                "server",
                "--model",
                snapshot_path,
                "--host",
                MLX_HOST,
                "--port",
                str(MLX_PORT),
                "--temp",
                str(DEFAULT_GEN_PARAMS["temperature"]),
                "--top-p",
                str(DEFAULT_GEN_PARAMS["top_p"]),
                "--max-tokens",
                str(DEFAULT_GEN_PARAMS["max_tokens"]),
                "--log-level",
                "INFO",
            ]

            env = os.environ.copy()
            env["HF_HUB_OFFLINE"] = "1"
            env["TOKENIZERS_PARALLELISM"] = "false"
            env["PYTHONUNBUFFERED"] = "1"

            self._proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            self._log_task = asyncio.create_task(self._read_proc_output(self._proc))
            self._load_task = asyncio.create_task(self._wait_and_warmup(self._proc))
            return self.get_state()

    async def _wait_and_warmup(self, proc: asyncio.subprocess.Process) -> None:
        """Wait for HTTP server, then trigger 1-token warmup so weights actually load into RAM."""
        try:
            # Phase 1: wait for HTTP server socket (up to 60s)
            server_up = False
            for _ in range(120):
                if proc.returncode is not None:
                    raise RuntimeError(
                        f"mlx_lm.server가 시작 중 종료되었습니다 (exit code {proc.returncode})."
                    )
                if await self._probe_server_alive():
                    server_up = True
                    break
                await asyncio.sleep(0.5)

            if not server_up:
                raise TimeoutError("mlx_lm.server HTTP 응답 대기 시간이 초과되었습니다.")

            # Phase 2: warmup request so 60GB weights load eagerly into Apple Metal unified memory
            self.status = "warming"
            self.status_detail = "모델 가중치(약 60GB)를 통합 메모리에 적재하는 중..."
            self._append_log(
                "HTTP server online. Warming up weights into Apple Metal unified memory..."
            )

            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(
                    f"{MLX_BASE_URL}/v1/chat/completions",
                    json={
                        "model": "default_model",
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 1,
                        "temperature": 0.0,
                        "stream": False,
                    },
                )
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"모델 워밍업 실패 (HTTP {resp.status_code}): {resp.text[:300]}"
                    )

            if proc.returncode is not None:
                raise RuntimeError("워밍업 도중 프로세스가 종료되었습니다.")

            self.status = "ready"
            self.status_detail = "모델이 통합 메모리에 적재되어 즉시 대화 가능합니다."
            self.started_at = time.monotonic()
            self.last_activity_at = time.monotonic()
            self._append_log("Model loaded & ready in unified memory.")

        except asyncio.CancelledError:
            self._append_log("Model loading cancelled.")
            raise
        except Exception as exc:
            self.last_error = str(exc)
            self.status = "error"
            self.status_detail = f"로드 실패: {exc}"
            self._append_log(f"ERROR: {exc}")
            # Clean up failed process
            if proc.returncode is None:
                await self._terminate_proc(proc)
            if self._proc is proc:
                self._proc = None

    async def _terminate_proc(self, proc: asyncio.subprocess.Process) -> None:
        if proc.returncode is not None:
            return
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except Exception:
            try:
                proc.terminate()
            except ProcessLookupError:
                return

        try:
            await asyncio.wait_for(proc.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            self._append_log("Process did not exit on SIGTERM; sending SIGKILL...")
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass

    async def _unload_locked(self) -> None:
        self.status = "stopping"
        self.status_detail = "모델을 메모리에서 해제하는 중..."

        if self._load_task and not self._load_task.done():
            self._load_task.cancel()
            try:
                await self._load_task
            except (asyncio.CancelledError, Exception):
                pass
            self._load_task = None

        if self._proc is not None:
            self._append_log(f"Stopping mlx_lm.server (PID {self._proc.pid})...")
            await self._terminate_proc(self._proc)
            self._proc = None
            self._append_log("Model unloaded. Unified memory released.")

        if self._log_task and not self._log_task.done():
            self._log_task.cancel()
            try:
                await self._log_task
            except asyncio.CancelledError:
                pass
            self._log_task = None

        self.external_server = False
        self.status = "unloaded"
        self.status_detail = "모델이 메모리에 올라가 있지 않습니다."
        self.started_at = None
        self.last_activity_at = None
        self.last_error = None

    async def unload(self) -> dict[str, Any]:
        """Stop mlx_lm.server subprocess and immediately release unified memory."""
        async with self._lock:
            if self.external_server:
                self.external_server = False
                self.status = "unloaded"
                self.status_detail = "외부 mlx_lm.server 연결을 해제했습니다 (외부 프로세스는 직접 종료해야 합니다)."
                return self.get_state()
            await self._unload_locked()
            return self.get_state()

    def _is_proc_alive(self) -> bool:
        if self.external_server:
            return True
        return self._proc is not None and self._proc.returncode is None

    def _get_memory_stats(self) -> dict[str, Any]:
        vm = psutil.virtual_memory()
        total_gb = round(vm.total / (1024**3), 1)
        available_gb = round(vm.available / (1024**3), 1)
        used_gb = round((vm.total - vm.available) / (1024**3), 1)
        percent = round(vm.percent, 1)

        proc_rss_gb = 0.0
        pid = None
        if self._proc is not None and self._proc.returncode is None:
            pid = self._proc.pid
            try:
                p = psutil.Process(pid)
                rss = p.memory_info().rss
                for child in p.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except psutil.Error:
                        pass
                proc_rss_gb = round(rss / (1024**3), 2)
            except psutil.Error:
                proc_rss_gb = 0.0

        return {
            "system_total_gb": total_gb,
            "system_used_gb": used_gb,
            "system_available_gb": available_gb,
            "system_percent": percent,
            "model_rss_gb": proc_rss_gb,
            "pid": pid,
        }

    def get_state(self) -> dict[str, Any]:
        # Detect unexpected crash
        if (
            self.status == "ready"
            and not self.external_server
            and (self._proc is None or self._proc.returncode is not None)
        ):
            code = self._proc.returncode if self._proc else None
            self.status = "error"
            self.last_error = f"mlx_lm.server 프로세스가 예기치 않게 종료되었습니다 (code={code})."
            self.status_detail = self.last_error
            self._proc = None

        now = time.monotonic()
        uptime_seconds = (
            int(now - self.started_at)
            if self.status == "ready" and self.started_at is not None
            else 0
        )
        idle_seconds = (
            int(now - self.last_activity_at)
            if self.status == "ready" and self.last_activity_at is not None
            else 0
        )

        return {
            "status": self.status,
            "status_detail": self.status_detail,
            "model_id": self.model_id or DEFAULT_MODEL_ID,
            "external_server": self.external_server,
            "uptime_seconds": uptime_seconds,
            "idle_seconds": idle_seconds,
            "idle_unload_minutes": self.idle_unload_minutes,
            "active_generations": self._active_generations,
            "last_error": self.last_error,
            "memory": self._get_memory_stats(),
            "available_models": discover_local_mlx_models(),
        }
