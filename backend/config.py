from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
HISTORY_DIR = BASE_DIR / "history"
PROMPTS_DIR = BASE_DIR / "prompts"
FRONTEND_DIR = BASE_DIR / "frontend"

HF_HUB_CACHE = Path(
    os.environ.get("HF_HUB_CACHE", str(Path.home() / ".cache" / "huggingface" / "hub"))
)
DEFAULT_MODEL_ID = os.environ.get(
    "MLX_MODEL_ID", "mlx-community/gpt-oss-120b-MXFP4-Q4"
)

MLX_HOST = os.environ.get("MLX_HOST", "127.0.0.1")
MLX_PORT = int(os.environ.get("MLX_PORT", "8080"))
MLX_BASE_URL = f"http://{MLX_HOST}:{MLX_PORT}"

PYTHON_BIN = os.environ.get("PYTHON_BIN", sys.executable)

DEFAULT_GEN_PARAMS: dict[str, Any] = {
    "temperature": 0.6,
    "top_p": 0.95,
    "presence_penalty": 0.2,
    "max_tokens": 4096,
    "reasoning_effort": "medium",
}

DEFAULT_IDLE_UNLOAD_MINUTES = int(os.environ.get("IDLE_UNLOAD_MINUTES", "30"))
DEFAULT_GIT_DEBOUNCE_SECONDS = int(os.environ.get("GIT_DEBOUNCE_SECONDS", "10"))


def _resolve_snapshot_dir(model_dir: Path) -> Path | None:
    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return None
    # Prefer refs/main if present
    ref_main = model_dir / "refs" / "main"
    if ref_main.is_file():
        rev = ref_main.read_text(encoding="utf-8").strip()
        candidate = snapshots_dir / rev
        if candidate.is_dir() and (candidate / "config.json").exists():
            return candidate
    # Fallback: most recently modified snapshot with config.json
    candidates = [
        p
        for p in snapshots_dir.iterdir()
        if p.is_dir() and (p / "config.json").exists()
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _compute_snapshot_size_bytes(snapshot_dir: Path) -> int:
    seen_inodes: set[tuple[int, int]] = set()
    total = 0
    for item in snapshot_dir.rglob("*"):
        try:
            real_path = item.resolve(strict=True)
            if not real_path.is_file():
                continue
            st = real_path.stat()
            inode_key = (st.st_dev, st.st_ino)
            if inode_key not in seen_inodes:
                seen_inodes.add(inode_key)
                total += st.st_size
        except OSError:
            continue
    return total


def discover_local_mlx_models() -> list[dict[str, Any]]:
    """Scan local Hugging Face hub cache for MLX-compatible models."""
    models: list[dict[str, Any]] = []
    if not HF_HUB_CACHE.is_dir():
        return models

    for entry in sorted(HF_HUB_CACHE.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("models--"):
            continue
        snapshot_dir = _resolve_snapshot_dir(entry)
        if snapshot_dir is None:
            continue
        # Ensure it actually has weight files
        has_weights = any(snapshot_dir.glob("*.safetensors"))
        if not has_weights:
            continue

        raw_name = entry.name[len("models--") :]
        repo_id = raw_name.replace("--", "/", 1)
        size_bytes = _compute_snapshot_size_bytes(snapshot_dir)
        models.append(
            {
                "id": repo_id,
                "snapshot_path": str(snapshot_dir),
                "size_gb": round(size_bytes / (1024**3), 2),
                "is_default": repo_id == DEFAULT_MODEL_ID,
            }
        )

    # Sort default model first
    models.sort(key=lambda m: (not m["is_default"], m["id"]))
    return models


def resolve_model_path(model_id: str | None = None) -> tuple[str, str]:
    """Return (repo_id, local_snapshot_path) for the requested or default model."""
    target_id = model_id or DEFAULT_MODEL_ID
    for m in discover_local_mlx_models():
        if m["id"] == target_id or m["snapshot_path"] == target_id:
            return m["id"], m["snapshot_path"]
    # If direct local path was given
    candidate = Path(target_id).expanduser()
    if candidate.is_dir() and (candidate / "config.json").exists():
        return candidate.name, str(candidate)
    raise FileNotFoundError(
        f"Local MLX model not found in cache: {target_id}"
    )
