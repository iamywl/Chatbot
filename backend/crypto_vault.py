from __future__ import annotations

import base64
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from backend.config import HISTORY_DIR

KEY_PREFIX = "CHATBOT-KEY-v1:"
VERIFIER_PLAINTEXT = b"CHATBOT_VAULT_OK_V1"
VERIFIER_AAD = b"chatbot:vault-verifier:v1"

SCRYPT_N = 2**15
SCRYPT_R = 8
SCRYPT_P = 1
KEY_LEN = 32


class VaultError(Exception):
    pass


class VaultLockedError(VaultError):
    pass


class VaultAuthError(VaultError):
    pass


class VaultDecryptError(VaultError):
    pass


def _b64e(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _b64d(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))


def derive_key_scrypt(
    passphrase: str,
    salt: bytes,
    n: int = SCRYPT_N,
    r: int = SCRYPT_R,
    p: int = SCRYPT_P,
) -> bytes:
    kdf = Scrypt(salt=salt, length=KEY_LEN, n=n, r=r, p=p)
    return kdf.derive(passphrase.encode("utf-8"))


def format_key_text(key_bytes: bytes) -> str:
    if len(key_bytes) != KEY_LEN:
        raise VaultError("Invalid key length; expected 32 bytes.")
    return f"{KEY_PREFIX}{_b64e(key_bytes)}"


def parse_key_text(key_text: str) -> bytes:
    cleaned = key_text.strip()
    if cleaned.startswith(KEY_PREFIX):
        cleaned = cleaned[len(KEY_PREFIX) :].strip()
    # Try base64 first, then hex
    try:
        raw = base64.b64decode(cleaned, validate=True)
        if len(raw) == KEY_LEN:
            return raw
    except Exception:
        pass
    try:
        raw = bytes.fromhex(cleaned)
        if len(raw) == KEY_LEN:
            return raw
    except Exception:
        pass
    raise VaultAuthError(
        "유효하지 않은 키 형식입니다. (CHATBOT-KEY-v1:... 또는 32바이트 Base64/Hex 키가 필요합니다)"
    )


def _encrypt_with_key(key: bytes, plaintext: bytes, aad: bytes) -> tuple[bytes, bytes]:
    nonce = os.urandom(12)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, aad)
    return nonce, ciphertext


def _decrypt_with_key(key: bytes, nonce: bytes, ciphertext: bytes, aad: bytes) -> bytes:
    try:
        return AESGCM(key).decrypt(nonce, ciphertext, aad)
    except InvalidTag as exc:
        raise VaultDecryptError("복호화에 실패했습니다. 키가 다르거나 데이터가 손상되었습니다.") from exc


class CryptoVault:
    """Memory-only AES-256-GCM key manager. Never persists keys to disk or Git."""

    def __init__(self, history_dir: Path | None = None) -> None:
        self.history_dir = history_dir or HISTORY_DIR
        self.meta_path = self.history_dir / "vault.json"
        self._key: bytearray | None = None
        self._last_activity: float = 0.0
        self._auto_lock_minutes: int = 0  # 0 = manual lock only

    def is_initialized(self) -> bool:
        return self.meta_path.is_file()

    def _check_auto_lock(self) -> None:
        if (
            self._key is not None
            and self._auto_lock_minutes > 0
            and (time.monotonic() - self._last_activity)
            >= self._auto_lock_minutes * 60
        ):
            self.lock()

    def is_unlocked(self) -> bool:
        self._check_auto_lock()
        return self._key is not None

    def touch(self) -> None:
        if self._key is not None:
            self._last_activity = time.monotonic()

    def set_auto_lock_minutes(self, minutes: int) -> None:
        self._auto_lock_minutes = max(0, int(minutes))
        self.touch()

    def _read_meta(self) -> dict[str, Any]:
        if not self.meta_path.is_file():
            raise VaultError("Vault가 아직 초기화되지 않았습니다.")
        return json.loads(self.meta_path.read_text(encoding="utf-8"))

    def _write_meta(self, meta: dict[str, Any]) -> None:
        self.history_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.meta_path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_path, self.meta_path)

    def _store_key_in_memory(self, key_bytes: bytes) -> None:
        self.lock()
        self._key = bytearray(key_bytes)
        self._last_activity = time.monotonic()

    def _require_key(self) -> bytes:
        self._check_auto_lock()
        if self._key is None:
            raise VaultLockedError("Vault가 잠겨 있습니다. 먼저 키 또는 패스프레이즈로 잠금을 해제하세요.")
        self._last_activity = time.monotonic()
        return bytes(self._key)

    def _verify_key_against_meta(self, key_bytes: bytes, meta: dict[str, Any]) -> None:
        verifier = meta.get("verifier")
        if not verifier:
            raise VaultError("vault.json에 검증 메타데이터가 없습니다.")
        try:
            nonce = _b64d(verifier["nonce"])
            ciphertext = _b64d(verifier["ciphertext"])
            plain = _decrypt_with_key(key_bytes, nonce, ciphertext, VERIFIER_AAD)
        except VaultDecryptError as exc:
            raise VaultAuthError("키 또는 패스프레이즈가 일치하지 않습니다.") from exc
        if plain != VERIFIER_PLAINTEXT:
            raise VaultAuthError("키 검증에 실패했습니다.")

    def init_with_passphrase(self, passphrase: str) -> str:
        if self.is_initialized():
            raise VaultError("이미 초기화된 Vault입니다. 기존 키로 잠금을 해제하세요.")
        if len(passphrase) < 6:
            raise VaultAuthError("패스프레이즈는 최소 6자 이상이어야 합니다.")

        salt = os.urandom(16)
        key_bytes = derive_key_scrypt(passphrase, salt)
        nonce, ciphertext = _encrypt_with_key(
            key_bytes, VERIFIER_PLAINTEXT, VERIFIER_AAD
        )

        meta = {
            "version": 1,
            "alg": "AES-256-GCM",
            "kdf": "scrypt",
            "salt": _b64e(salt),
            "scrypt_params": {"n": SCRYPT_N, "r": SCRYPT_R, "p": SCRYPT_P},
            "verifier": {
                "nonce": _b64e(nonce),
                "ciphertext": _b64e(ciphertext),
            },
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        self._write_meta(meta)
        self._store_key_in_memory(key_bytes)
        return format_key_text(key_bytes)

    def init_with_random_key(self) -> str:
        if self.is_initialized():
            raise VaultError("이미 초기화된 Vault입니다. 기존 키로 잠금을 해제하세요.")

        key_bytes = os.urandom(KEY_LEN)
        nonce, ciphertext = _encrypt_with_key(
            key_bytes, VERIFIER_PLAINTEXT, VERIFIER_AAD
        )

        meta = {
            "version": 1,
            "alg": "AES-256-GCM",
            "kdf": "keyfile",
            "verifier": {
                "nonce": _b64e(nonce),
                "ciphertext": _b64e(ciphertext),
            },
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        self._write_meta(meta)
        self._store_key_in_memory(key_bytes)
        return format_key_text(key_bytes)

    def unlock_with_passphrase(self, passphrase: str) -> None:
        meta = self._read_meta()
        if meta.get("kdf") != "scrypt":
            raise VaultAuthError(
                "이 Vault는 랜덤 키 파일(.key) 방식으로 초기화되었습니다. 키 파일로 잠금을 해제하세요."
            )
        salt = _b64d(meta["salt"])
        params = meta.get("scrypt_params", {})
        key_bytes = derive_key_scrypt(
            passphrase,
            salt,
            n=int(params.get("n", SCRYPT_N)),
            r=int(params.get("r", SCRYPT_R)),
            p=int(params.get("p", SCRYPT_P)),
        )
        self._verify_key_against_meta(key_bytes, meta)
        self._store_key_in_memory(key_bytes)

    def unlock_with_key_text(self, key_text: str) -> None:
        meta = self._read_meta()
        key_bytes = parse_key_text(key_text)
        self._verify_key_against_meta(key_bytes, meta)
        self._store_key_in_memory(key_bytes)

    def lock(self) -> None:
        if self._key is not None:
            for i in range(len(self._key)):
                self._key[i] = 0
            self._key = None
        self._last_activity = 0.0

    def export_key_text(self) -> str:
        key_bytes = self._require_key()
        return format_key_text(key_bytes)

    def encrypt_session(
        self, session_id: str, plaintext_markdown: str, updated_at: str
    ) -> dict[str, Any]:
        key_bytes = self._require_key()
        aad = f"chatbot:session:{session_id}:v1".encode("utf-8")
        nonce, ciphertext = _encrypt_with_key(
            key_bytes, plaintext_markdown.encode("utf-8"), aad
        )
        return {
            "v": 1,
            "alg": "AES-256-GCM",
            "session_id": session_id,
            "updated_at": updated_at,
            "nonce": _b64e(nonce),
            "ciphertext": _b64e(ciphertext),
        }

    def decrypt_session(self, envelope: dict[str, Any]) -> str:
        key_bytes = self._require_key()
        session_id = envelope.get("session_id")
        if not session_id or "nonce" not in envelope or "ciphertext" not in envelope:
            raise VaultDecryptError("암호화 파일 포맷이 올바르지 않습니다.")
        aad = f"chatbot:session:{session_id}:v1".encode("utf-8")
        plain = _decrypt_with_key(
            key_bytes,
            _b64d(envelope["nonce"]),
            _b64d(envelope["ciphertext"]),
            aad,
        )
        return plain.decode("utf-8")

    def status(self) -> dict[str, Any]:
        initialized = self.is_initialized()
        unlocked = self.is_unlocked()
        kdf = None
        if initialized:
            try:
                kdf = self._read_meta().get("kdf")
            except Exception:
                kdf = None
        idle_seconds = (
            int(time.monotonic() - self._last_activity)
            if unlocked and self._last_activity > 0
            else 0
        )
        return {
            "initialized": initialized,
            "unlocked": unlocked,
            "kdf": kdf,
            "auto_lock_minutes": self._auto_lock_minutes,
            "idle_seconds": idle_seconds,
        }
