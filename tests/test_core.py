from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.config import discover_local_mlx_models
from backend.crypto_vault import (
    CryptoVault,
    VaultAuthError,
    VaultDecryptError,
    VaultLockedError,
)
from backend.llm_client import HarmonyStreamParser, build_llm_messages
from backend.storage import (
    EncryptedSessionStorage,
    parse_session_markdown,
    serialize_session_markdown,
)


def test_crypto_vault_passphrase_and_keyfile_roundtrip(tmp_path: Path) -> None:
    vault = CryptoVault(history_dir=tmp_path)
    assert not vault.is_initialized()
    assert not vault.is_unlocked()

    backup_key = vault.init_with_passphrase("my-super-secret-passphrase")
    assert backup_key.startswith("CHATBOT-KEY-v1:")
    assert vault.is_initialized()
    assert vault.is_unlocked()

    # Encrypt session
    secret_text = "비밀 대화 내용: JWT secret 및 알고리즘 풀이"
    env = vault.encrypt_session("20261120_113000_abcdef", secret_text, "2026-11-20T11:30:00+09:00")
    assert "비밀 대화" not in json.dumps(env, ensure_ascii=False)
    assert vault.decrypt_session(env) == secret_text

    # Lock wipes key from memory
    vault.lock()
    assert not vault.is_unlocked()
    with pytest.raises(VaultLockedError):
        vault.decrypt_session(env)

    # Wrong passphrase fails
    with pytest.raises(VaultAuthError):
        vault.unlock_with_passphrase("wrong-passphrase")

    # Unlock with original passphrase succeeds
    vault.unlock_with_passphrase("my-super-secret-passphrase")
    assert vault.decrypt_session(env) == secret_text

    # Lock & unlock with exported .key text also succeeds!
    vault.lock()
    vault.unlock_with_key_text(backup_key)
    assert vault.decrypt_session(env) == secret_text

    # Tampered session_id (AAD mismatch) must fail decryption
    tampered = dict(env)
    tampered["session_id"] = "20261120_113000_000000"
    with pytest.raises(VaultDecryptError):
        vault.decrypt_session(tampered)


def test_crypto_vault_random_keyfile_mode(tmp_path: Path) -> None:
    vault = CryptoVault(history_dir=tmp_path)
    key_text = vault.init_with_random_key()
    assert vault.is_unlocked()

    env = vault.encrypt_session("20261120_120000_123456", "hello keyfile", "2026-11-20T12:00:00+09:00")
    vault.lock()

    with pytest.raises(VaultAuthError):
        vault.unlock_with_passphrase("anything")

    vault.unlock_with_key_text(key_text)
    assert vault.decrypt_session(env) == "hello keyfile"


def test_storage_encrypted_roundtrip_and_tricky_markdown(tmp_path: Path) -> None:
    history_dir = tmp_path / "history"
    prompts_dir = tmp_path / "prompts"
    vault = CryptoVault(history_dir=history_dir)
    vault.init_with_passphrase("storage-test-pw")
    store = EncryptedSessionStorage(vault, history_dir=history_dir, prompts_dir=prompts_dir)

    sess = store.create_session(title="다익스트라 비밀 노트", persona="algo_java")
    sess["messages"].append(
        {
            "id": "msg_0001",
            "role": "user",
            "content": "본문 안에 가짜 헤더\n## User\n## Assistant\n가 있어도 안 깨져야 함",
            "reasoning": "",
            "ts": "2026-11-20T11:30:01+09:00",
        }
    )
    sess["messages"].append(
        {
            "id": "msg_0002",
            "role": "assistant",
            "content": "최종 답변입니다.\n```java\n// ## User\nSystem.out.println(\"ok\");\n```",
            "reasoning": "단계 1: dist[k][node] 정의\n단계 2: PQ 사용",
            "ts": "2026-11-20T11:30:10+09:00",
            "metrics": {"tok_per_sec": 63.5, "completion_tokens": 120},
        }
    )
    store.save_session(sess)

    # Confirm ciphertext file on disk has ZERO plaintext leakage
    enc_files = list(history_dir.rglob("*.md.enc"))
    assert len(enc_files) == 1
    raw_disk = enc_files[0].read_text(encoding="utf-8")
    assert "다익스트라" not in raw_disk
    assert "최종 답변" not in raw_disk
    assert "dist[k][node]" not in raw_disk
    assert "algo_java" not in raw_disk

    # Decrypt and verify exact round-trip
    loaded = store.get_session(sess["session_id"])
    assert loaded["title"] == "다익스트라 비밀 노트"
    assert len(loaded["messages"]) == 2
    assert "## Assistant" in loaded["messages"][0]["content"]
    assert "dist[k][node]" in loaded["messages"][1]["reasoning"]
    assert loaded["messages"][1]["metrics"]["tok_per_sec"] == 63.5

    # Full-text search across decrypted sessions
    found = store.list_sessions(query="dist[k][node]")
    assert len(found) == 1
    assert found[0]["session_id"] == sess["session_id"]
    assert store.list_sessions(query="없는단어xyz") == []

    # Fork session at message 1
    forked = store.fork_session(sess["session_id"], up_to_message_id="msg_0001")
    assert forked["session_id"] != sess["session_id"]
    assert len(forked["messages"]) == 1
    assert "(분기)" in forked["title"]

    # Clean export markdown
    fname, exported_md = store.export_session_markdown(sess["session_id"])
    assert fname.endswith(".md")
    assert "CHATBOT_TURN" not in exported_md
    assert "<summary>Reasoning Process</summary>" in exported_md

    # Persona path traversal rejected
    with pytest.raises(ValueError):
        store.save_persona("../evil", "nope")


def test_harmony_stream_parser_char_by_char_and_chunks() -> None:
    sample = (
        "<|channel|>analysis<|message|>Step 1: think deeply.\nStep 2: verify.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>Here is the final answer: 42!<|return|>"
    )
    # Feed 1 character at a time (worst-case chunk split!)
    parser = HarmonyStreamParser()
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    for ch in sample:
        for kind, text in parser.feed(ch):
            if kind == "reasoning":
                reasoning_parts.append(text)
            else:
                content_parts.append(text)
    for kind, text in parser.flush():
        if kind == "reasoning":
            reasoning_parts.append(text)
        else:
            content_parts.append(text)

    assert "".join(reasoning_parts) == "Step 1: think deeply.\nStep 2: verify."
    assert "".join(content_parts) == "Here is the final answer: 42!"


def test_harmony_parser_think_tags_and_plain_text() -> None:
    # <think>...</think>
    p = HarmonyStreamParser()
    out = p.feed("<think>internal</think>visible answer") + p.flush()
    r = "".join(t for k, t in out if k == "reasoning")
    c = "".join(t for k, t in out if k == "content")
    assert r == "internal"
    assert c == "visible answer"

    # Plain text without any tags
    p2 = HarmonyStreamParser()
    out2 = p2.feed("Hello <world> 1 < 2") + p2.flush()
    assert "".join(t for k, t in out2 if k == "content") == "Hello <world> 1 < 2"


def test_build_llm_messages_strips_past_reasoning() -> None:
    history = [
        {"role": "user", "content": "Q1", "reasoning": ""},
        {"role": "assistant", "content": "A1", "reasoning": "huge secret CoT"},
        {"role": "user", "content": "Q2", "reasoning": ""},
    ]
    msgs = build_llm_messages("System prompt", history)
    assert msgs == [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
    ]


def test_fastapi_vault_and_sessions_e2e(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import backend.main as main_mod

    test_history = tmp_path / "history"
    test_prompts = tmp_path / "prompts"
    test_prompts.mkdir(parents=True, exist_ok=True)
    (test_prompts / "default.md").write_text("You are helpful.\n", encoding="utf-8")

    test_vault = CryptoVault(history_dir=test_history)
    test_storage = EncryptedSessionStorage(
        test_vault, history_dir=test_history, prompts_dir=test_prompts
    )
    monkeypatch.setattr(main_mod, "vault", test_vault)
    monkeypatch.setattr(main_mod, "storage", test_storage)
    monkeypatch.setattr(main_mod.git_sync, "auto_sync", False)

    with TestClient(main_mod.app) as client:
        # Serve frontend index
        r = client.get("/")
        assert r.status_code == 200
        assert "VaultLLM" in r.text

        # Sessions blocked while locked (423)
        r = client.get("/api/sessions")
        assert r.status_code == 423

        # Init vault
        r = client.post(
            "/api/vault/init",
            json={"mode": "passphrase", "passphrase": "test-password-123"},
        )
        assert r.status_code == 200
        backup_key = r.json()["backup_key"]
        assert backup_key.startswith("CHATBOT-KEY-v1:")

        # Create 2 sessions (multi-session support)
        s1 = client.post("/api/sessions", json={"title": "세션 1", "persona": "default"}).json()
        s2 = client.post("/api/sessions", json={"title": "세션 2", "persona": "default"}).json()
        assert s1["session_id"] != s2["session_id"]

        lst = client.get("/api/sessions").json()["sessions"]
        assert len(lst) == 2

        # Lock vault -> cannot read sessions
        client.post("/api/vault/lock")
        assert client.get("/api/sessions").status_code == 423

        # Unlock with .key text -> past sessions restored!
        r = client.post("/api/vault/unlock", json={"key_text": backup_key})
        assert r.status_code == 200
        assert r.json()["unlocked"] is True
        lst2 = client.get("/api/sessions").json()["sessions"]
        assert len(lst2) == 2

        # Offline CLI decrypt.py works with exported .key file!
        key_file = tmp_path / "test.key"
        key_file.write_text(backup_key, encoding="utf-8")
        proc = subprocess.run(
            [
                sys.executable,
                "scripts/decrypt.py",
                "--history-dir",
                str(test_history),
                "--key-file",
                str(key_file),
                "--list",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "세션 1" in proc.stdout
        assert "세션 2" in proc.stdout


def test_discover_local_mlx_models() -> None:
    models = discover_local_mlx_models()
    ids = [m["id"] for m in models]
    assert "mlx-community/gpt-oss-120b-MXFP4-Q4" in ids


def test_git_sync_worker_encrypted_push_and_pull(tmp_path: Path) -> None:
    import asyncio
    from backend.git_sync import GitSyncWorker

    bare_remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(bare_remote)], check=True, capture_output=True)

    repo = tmp_path / "local"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "remote", "add", "origin", str(bare_remote)], cwd=repo, check=True)

    (repo / "README.md").write_text("# Test\n", encoding="utf-8")
    (repo / "prompts").mkdir()
    (repo / "prompts" / "default.md").write_text("Hello\n", encoding="utf-8")
    (repo / "history").mkdir()
    (repo / "history" / ".gitkeep").write_text("", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "push", "-u", "origin", "main"], cwd=repo, check=True, capture_output=True)

    # Create encrypted session and an untracked secret .key file (must NOT be committed!)
    vault = CryptoVault(history_dir=repo / "history")
    vault.init_with_passphrase("git-sync-pass")
    store = EncryptedSessionStorage(vault, history_dir=repo / "history", prompts_dir=repo / "prompts")
    store.create_session(title="비밀 세션 제목")
    (repo / "secret.key").write_text("DO_NOT_COMMIT", encoding="utf-8")

    worker = GitSyncWorker(repo_dir=repo)
    status = asyncio.run(worker.sync_now(reason="test"))
    assert status["last_error"] is None
    assert status["pending_count"] == 0

    # Verify commit message leaks NO title and secret.key was NOT committed
    log = subprocess.run(
        ["git", "log", "-1", "--pretty=%s"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    assert log.startswith("vault: sync encrypted chats")
    assert "비밀 세션 제목" not in log

    tracked = subprocess.run(
        ["git", "ls-files"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout
    assert "secret.key" not in tracked
    assert "history/vault.json" in tracked
    assert ".md.enc" in tracked

