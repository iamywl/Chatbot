# Private MLX Chatbot Vault

Apple Silicon(M4 Max) 로컬 LLM(`mlx-community/gpt-oss-120b-MXFP4-Q4`) 기반 개인용 암호화 AI 챗봇 웹 애플리케이션입니다.

## 주요 기능

- **모델 메모리 제어 (Load / Unload)**: 웹 UI에서 `mlx_lm.server` 프로세스를 직접 기동/종료하여 60GB 통합 메모리를 원할 때만 적재하고 즉시 해제할 수 있습니다. (실시간 RAM 게이지 + 미사용 자동 언로드 타이머 지원)
- **AES-256-GCM 암호화 채팅 저장 & Git 동기화**: 모든 대화 기록은 내부적으로 `Markdown + YAML Frontmatter` 형식으로 구성되지만, 디스크와 Git(`history/YYYY-MM/*.md.enc`)에는 암호문만 저장됩니다. 파일명과 커밋 메시지에도 대화 주제가 노출되지 않습니다.
- **개인 키 기반 Vault 복호화**: 패스프레이즈(`scrypt` 키 유도) 또는 개인 `.key` 파일로 과거 암호화 채팅을 복호화해 열람·검색할 수 있습니다. 키는 서버 메모리에만 임시 유지되며 디스크/Git에 저장되지 않습니다.
- **Reasoning(CoT) 분리 스트리밍**: `gpt-oss-120b`의 Harmony 채널(`<|channel|>analysis` / `<|channel|>final`)을 실시간 분리해 사고 과정은 접기 UI로 보여주고, 멀티턴 컨텍스트에는 최종 답변만 재투입해 컨텍스트 폭발과 반복 루프를 방지합니다.
- **멀티 세션 & 페르소나**: 다중 세션 생성·전환·검색·이름 변경·분기(Fork)·Markdown 내보내기 및 `prompts/*.md` 페르소나 관리.

## 실행 방법

```bash
chmod +x run.sh
./run.sh
```

브라우저에서 `http://127.0.0.1:8000`에 접속합니다.

1. 최초 접속 시 **패스프레이즈** 또는 **개인 `.key` 파일 생성**으로 암호화 Vault를 초기화합니다.
2. 상단 헤더의 **Load Model** 버튼으로 `gpt-oss-120b`를 메모리에 적재합니다.
3. 채팅 후 **Unload** 버튼으로 통합 메모리를 즉시 반환할 수 있습니다.

## 서버 없이 과거 채팅 복호화하기 (오프라인 CLI)

```bash
# 패스프레이즈로 전체 복호화 조회
python3 scripts/decrypt.py --list

# 특정 세션 복호화 출력
python3 scripts/decrypt.py history/2026-11/20261120_120000_abcd12.md.enc

# .key 파일로 복호화하여 파일로 저장
python3 scripts/decrypt.py --key-file ~/vault.key history/2026-11/20261120_120000_abcd12.md.enc -o out.md
```

## 디렉터리 구조

```text
├── backend/
│   ├── config.py          # 경로, 기본값, 로컬 HF 캐시 모델 탐색
│   ├── crypto_vault.py    # AES-256-GCM + scrypt / .key 파일 메모리 전용 Vault
│   ├── model_manager.py   # mlx_lm.server 프로세스 Load/Unload + RAM 모니터링 + Idle 자동 해제
│   ├── llm_client.py      # SSE 스트리밍 + gpt-oss-120b Harmony Reasoning 파서
│   ├── storage.py         # Markdown+Frontmatter 직렬화 및 암호화 세션 CRUD/검색/분기
│   ├── git_sync.py        # 비동기 디바운스 Git Commit/Push/Pull 워커
│   └── main.py            # FastAPI 서버 & API 엔드포인트
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── prompts/               # 페르소나 시스템 프롬프트 (Git 관리)
├── history/               # 암호화된 세션 저장소 (*.md.enc, vault.json)
├── scripts/
│   └── decrypt.py         # 오프라인 독립 복호화 CLI
└── tests/
    └── test_core.py
```
