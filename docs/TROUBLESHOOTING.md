# 트러블슈팅 및 멱등성(Idempotency) 설계 기록

## 1. 모델 로드 실패: `XPC_ERROR_CONNECTION_INVALID` / `Unable to load kernel arangefloat32`

- **발생 일시**: 2026-11-24
- **증상**:
  - `mlx.core` 단독 실행(`mx.arange`)은 정상 동작하나, 웹 서버(`uvicorn`)가 실행한 `mlx_lm.server` 워밍업 단계에서 아래 오류로 실패:
    ```text
    RuntimeError: [metal::Device] Unable to load kernel arangefloat32
    Compiler encountered XPC_ERROR_CONNECTION_INVALID (is the OS shutting down?)
    ```
- **원인**:
  1. `ModelManager.load()`에서 `asyncio.create_subprocess_exec(..., start_new_session=True)`로 자식 프로세스를 새 세션으로 분리하면서, macOS의 `MTLCompilerService` XPC 부트스트랩 네임스페이스 연결이 끊어짐.
  2. 오래된 백그라운드 `uvicorn` 세션이 유지된 상태에서 새 Metal 커널 컴파일 요청 시 XPC 연결이 무효화됨.
- **해결**:
  - `backend/model_manager.py`: `start_new_session=True` 제거하여 부모의 유효한 macOS XPC 세션을 그대로 상속하고, 종료 시 `proc.terminate()` / `proc.kill()`로 안전하게 해당 자식 프로세스만 정리하도록 수정.
  - `run.sh`: 기존 `uvicorn backend.main:app` 프로세스가 있으면 정상 종료(`SIGTERM` → 필요 시 `SIGKILL`) 후 재기동하도록 변경.

---

## 2. 멱등성(Idempotency) 보장 원칙

| 대상 | 멱등성 보장 방식 |
| --- | --- |
| **`./run.sh` (서버 실행)** | 이미 실행 중인 `uvicorn backend.main:app`을 감지해 정리한 뒤 기동하므로 몇 번을 실행해도 포트 충돌 없이 동일한 `Running` 상태로 수렴 |
| **`POST /api/model/load`** | `loading` / `warming` / `ready`(동일 모델) 상태에서 중복 호출해도 새 프로세스를 띄우지 않고 현재 상태를 그대로 반환 |
| **`POST /api/model/unload`** | 이미 `unloaded` 상태여도 오류 없이 정상 종료 상태 반환 |
| **`PUT /api/personas/{name}`** | 동일 이름으로 반복 저장 시 덮어쓰기로 항상 최신 상태 유지 |
| **Git 동기화 (`GitSyncWorker.sync_now`)** | `asyncio.Lock` 직렬화 + `git diff --cached --quiet` 체크로 변경 사항이 없으면 빈 커밋을 만들지 않음 |
