# pong-dqn

자료구조 및 실습 기말프로젝트 프로토타입 — 순수 C로 구현한 MLP + DQN으로 Pong 학습.

- MLP 6→64→64→3 (ReLU), Adam, 역전파 직접 구현
- DQN: 타깃 네트워크, 원형 큐 리플레이 버퍼, ε-greedy, Huber 손실
- 외부 라이브러리 없음 (표준 C + libm)

## 빌드 / 실행
```
make                      # pong_dqn 빌드
make gradcheck            # 역전파 수치 미분 검증 (double 빌드)
./pong_dqn train 300000 1 # 학습 (steps, seed) -> best.bin, train_log.csv
./pong_dqn eval best.bin  # 평가
```
하이퍼파라미터는 `pong_dqn.c` 상단의 `#define`. 환경변수 기반 실험 스크립트(Double DQN 등)는 `experiments` 브랜치.

## 상태
프로토타입. 득점 승률은 1000점 재평가 기준 85~97% (랜덤 3.8%). 상대는 고정 룰 기반이라 쉬운 환경이며,
승률이 AI 실력인지 상대 약함인지는 아직 검증 전(추적 정책 기준선 필요).

## v2 (dqn-v2-shaping 브랜치)
착지 예측 입력 + 포텐셜 shaping으로 추적 상대 대비 승률을 올린 버전. 배포 모델 `models/final_model.bin`은 시험 시드 10000점에서 99.9%(선택된 체크포인트 기준, 학습 수렴 성능 아님).
빌드·재현·한계는 [EXPERIMENTS.md](EXPERIMENTS.md) 참고. `make pong_dqn_v2 && ./select_model models/final_model.bin`.

## 랭킹 (`ranking.c` / `ranking.h`)
TOP 10 점수 기록 모듈. 테스트: `make ranking-test`

## 게임 (터미널, v2 모델 + 랭킹)
```
make game && ./pong_game
```
- 메인 화면: **PONG** 큰 제목 + `1. 시작 / 2. 난이도 / 3. 설정 / 4. 종료` + 랭킹 TOP 10 (창이 작으면 제목을 한 줄로 줄여 자리 확보)
- 왼쪽 = DQN(`models/final_model.bin`), 오른쪽 = 사람(↑↓ 또는 W/S)
- 난이도 = 같은 모델의 착지 예측 입력에 오차 σ 주입 (쉬움 0.30 / 보통 0.15 / 어려움 0) = 랭킹 Lv1~3
- 설정: 목표 점수(3/5/7/11), 게임 속도(20~50fps), AI 판단(Q값) 표시
- 경기가 끝나면 `ranking.c` 로 점수 계산 → TOP 10 이면 이름 입력 → `ranking.txt` 저장. 중간에 나가면 기록 안 됨
- 파일 구성
  - `pong_core.h` — 게임 로직 (`pong_dqn_v2.c` include, 난이도·물리·받아침 집계·랭킹 점수)
  - `pong_game.c` — 터미널 UI (ANSI + termios). 터미널은 키를 뗀 순간을 알 수 없어 '키 반복이 끊기면 정지'
