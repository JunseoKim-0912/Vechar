# Vechar용 MemMachine 프로젝트 정책 (v0.3.9)

이 문서는 프로젝트 준비와 검증 절차다. Production memory를 활성화하거나 실제 사용자 데이터를 저장하라는 뜻이 아니다. 기본 `MEMORY_PROVIDER=noop`을 유지한다.

## 책임 분리

- Vechar DB: CharacterProfile·WorldProfile의 canonical state와 최근 Message history.
- MemMachine: 같은 사용자·캐릭터의 여러 Conversation에 걸친 episodic long-term experience.
- MemMachine의 short-term memory는 사용하지 않는다. Semantic/profile 결과도 Vechar prompt나 canonical profile에 병합하지 않는다.

MemMachine v0.3.9은 같은 episode가 short-term과 long-term에 있으면 short-term을 우선하고 long-term 응답에서 중복 제거한다. Vechar adapter는 `long_term_memory.episodes`만 후보로 사용하므로 short-term을 켠 기본 프로젝트에서는 최근 기억이 조용히 사라질 수 있다. 로컬 PoC에서는 short-term을 끄고 서버를 재시작한 뒤 A 대화의 사건을 B 대화에서 검색했다.

## 프로젝트 불변식

`MEMORY_PROVIDER=memmachine`을 선택한 프로세스는 첫 adapter 초기화 시 다음을 확인한다.

1. 설치된 `memmachine-client`와 `memmachine-common`, 서버 health의 version이 모두 `0.3.9`.
2. 조회된 organization/project ID가 환경 설정과 일치하고 프로젝트가 이미 존재함.
3. Episodic memory `enabled=True`, `long_term_memory_enabled=True`, `short_term_memory_enabled=False`.

프로젝트 조회는 공개 SDK `get_project()`와 `get_episodic_memory_config()`를 쓴다. Chat 경로는 프로젝트를 만들거나 설정을 바꾸지 않는다. 불일치·프로젝트 없음은 `MemoryConfigurationError`로 실패하며 빈 memory로 위장하지 않는다. 연결 실패·timeout·서버 불건전 상태는 기존 memory-only fallback으로 처리한다. 서버가 재설정 직후 stale short-term episode를 반환해도 configuration error로 차단한다.

이 검사는 프로세스의 첫 adapter 생성 시 한 번 수행된다. 실행 중 외부 관리자가 설정을 바꾸는 경우 빈 결과만으로 모든 drift를 알아낼 수는 없으므로, 설정 변경 후 MemMachine을 재시작하고 검증 프로세스도 재시작해야 한다.

## 명시적 준비 절차

`setup_memmachine_project.py`는 채팅에서 호출되지 않는다. 필요한 `MEMORY_PROVIDER=memmachine`, `MEMMACHINE_BASE_URL`, `MEMMACHINE_ORG_ID`, `MEMMACHINE_PROJECT_ID`를 **관리용 로컬 프로세스에만** 제공한다. 실제 credential은 환경변수로만 전달하며 스크립트나 Git 파일에 넣지 않는다.

```powershell
# character-chatbot/에서, 먼저 read-only 확인
.\.venv\Scripts\python.exe .\setup_memmachine_project.py

# 선택한 disposable/admin 프로젝트에만 명시적으로 생성·설정 적용
.\.venv\Scripts\python.exe .\setup_memmachine_project.py --apply
```

기본 실행은 read-only다. `--apply`는 프로젝트가 없을 때만 생성하고, 필요할 때 `configure_episodic_memory(enabled=True, long_term_memory_enabled=True, short_term_memory_enabled=False)`를 호출한다. 변경됐다면 MemMachine 서버를 재시작하고 read-only 확인을 다시 수행한다. 기존 사용자 데이터가 있는 프로젝트에는 영향 범위를 검토하기 전 `--apply`를 실행하지 않는다. 서버 재시작까지 끝난 별도의 가상 데이터 smoke test가 필요하다.

## Semantic/profile 및 지연 시간

SDK 0.3.9의 `Memory.search()`는 episodic과 semantic 유형을 함께 요청한다. Adapter는 semantic/profile 응답을 무시하고 long-term episodic만 매핑한다. Semantic 검색이 서버 내부에서 추가 자원·지연을 유발하는지는 운영 전 측정해야 한다. 이번 단계에서 private SDK나 직접 REST 호출로 검색 유형을 우회하지 않는다.

PoC 단일 관측치는 same-session 약 2.54초, cross-session 약 3.95초였다. 기본 `MEMMACHINE_TIMEOUT_SECONDS=3`은 **각 SDK HTTP 요청**의 timeout이지 프로젝트 조회·설정 검증·검색을 모두 합친 deadline이 아니다. PoC는 30초 요청별 timeout으로 수행됐으므로 기본 3초로 성공한다고 볼 수 없다. 값을 이번 단계에서 변경하지 않는다. 운영 전 retrieval timeout, P50/P95, generation을 포함한 end-to-end 지연, 동시 요청, 실패율을 측정한다.

## 업그레이드 및 production 차단 조건

서버·client·common은 0.3.9로 고정한다. 업그레이드 시 project config API, 검색 응답의 long-/short-term 구조와 중복 제거, user/agent 필터, 삭제 API, idempotency 지원을 다시 검증한다. 자동 최신 버전 업그레이드는 하지 않는다.

프로젝트 준비만으로 production memory를 켤 수 없다. Completed turn source ID를 영속 ingestion ledger에 기록한 뒤 provider add와 성공 상태를 조정하는 idempotency 설계가 필요하다. 사용자·캐릭터·대화 범위의 안전한 삭제와 tombstone/retry도 아직 없다. 실제 한국어 검색 품질과 운영 지연은 별도로 평가해야 한다. 현재 adapter의 범위 삭제는 계속 fail-closed다.
