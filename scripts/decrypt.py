#!/usr/bin/env python3
"""
Standalone offline decryptor for Private MLX Chatbot encrypted history (.md.enc).
Works without running the FastAPI server — only needs Python + `cryptography`.

Examples:
  python3 scripts/decrypt.py --list
  python3 scripts/decrypt.py history/2026-11/20261120_120000_ab12cd.md.enc
  python3 scripts/decrypt.py --key-file ~/vault.key history/2026-11/20261120_120000_ab12cd.md.enc -o out.md
"""
from __future__ import annotations

import argparse
import getpass
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.crypto_vault import CryptoVault, VaultError  # noqa: E402
from backend.storage import parse_session_markdown, serialize_session_markdown  # noqa: E402


def unlock_vault_cli(vault: CryptoVault, key_file: str | None) -> None:
    if not vault.is_initialized():
        print("Error: history/vault.json not found. Vault is not initialized.", file=sys.stderr)
        sys.exit(1)

    if key_file:
        key_text = Path(key_file).expanduser().read_text(encoding="utf-8")
        vault.unlock_with_key_text(key_text)
        return

    meta = json.loads(vault.meta_path.read_text(encoding="utf-8"))
    if meta.get("kdf") == "keyfile":
        key_text = getpass.getpass("Enter vault key (CHATBOT-KEY-v1:...): ")
        vault.unlock_with_key_text(key_text)
    else:
        pw = getpass.getpass("Enter vault passphrase (or paste CHATBOT-KEY-v1:...): ")
        if pw.strip().startswith("CHATBOT-KEY-v1:"):
            vault.unlock_with_key_text(pw)
        else:
            vault.unlock_with_passphrase(pw)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decrypt Private MLX Chatbot history (.md.enc) files offline."
    )
    parser.add_argument("files", nargs="*", help="Path(s) to .md.enc file(s) to decrypt")
    parser.add_argument("--list", action="store_true", help="List all encrypted sessions with decrypted titles")
    parser.add_argument("--key-file", help="Path to exported .key file")
    parser.add_argument("--history-dir", default=str(ROOT_DIR / "history"), help="History directory path")
    parser.add_argument("-o", "--output", help="Write decrypted Markdown to this file (single file mode)")
    parser.add_argument("--raw", action="store_true", help="Keep internal HTML turn comments instead of clean Markdown")
    args = parser.parse_args()

    history_dir = Path(args.history_dir).expanduser().resolve()
    vault = CryptoVault(history_dir=history_dir)

    try:
        unlock_vault_cli(vault, args.key_file)
    except VaultError as exc:
        print(f"Unlock failed: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.list or not args.files:
        enc_files = sorted(history_dir.rglob("*.md.enc"), reverse=True)
        if not enc_files:
            print("No encrypted session files found.")
            return
        for p in enc_files:
            try:
                env = json.loads(p.read_text(encoding="utf-8"))
                md = vault.decrypt_session(env)
                sess = parse_session_markdown(md)
                rel = p.relative_to(history_dir.parent) if history_dir.parent in p.parents else p
                print(
                    f"{sess['updated_at'][:19]}  |  [{sess['persona']}]  {sess['title']}  ({len(sess['messages'])} msgs)  -> {rel}"
                )
            except Exception as exc:
                print(f"[UNREADABLE] {p}: {exc}")
        return

    outputs: list[str] = []
    for f in args.files:
        p = Path(f).expanduser()
        env = json.loads(p.read_text(encoding="utf-8"))
        md = vault.decrypt_session(env)
        if not args.raw:
            sess = parse_session_markdown(md)
            md = serialize_session_markdown(sess, clean_export=True)
        outputs.append(md)

    combined = "\n---\n\n".join(outputs)
    if args.output:
        Path(args.output).expanduser().write_text(combined, encoding="utf-8")
        print(f"Decrypted Markdown written to {args.output}")
    else:
        print(combined)


if __name__ == "__main__":
    main()
