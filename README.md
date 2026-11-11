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
  - `pong_game.c` — 터미널 UI (ANSI + 입력 처리: 리눅스는 termios, 윈도우는 콘솔 API). 터미널은 키를 뗀 순간을 알 수 없어 '키 반복이 끊기면 정지'

## Windows 에서 실행
`pong_game.c` 의 `#ifdef _WIN32` 부분이 윈도우 콘솔(키 입력·한글 출력·ANSI 색)을 처리한다. 리눅스/맥은 위의 `make game && ./pong_game` 그대로.

필요: Windows 10 이상, gcc (MinGW-w64), 콘솔 창 크기 가로 64칸 · 세로 16줄 이상

1. **exe 실행** — 소스 zip 에 들어 있는 `pong_game.exe` 를 터미널(PowerShell / Windows Terminal)에서 실행
   ```
   .\pong_game.exe
   ```
   `models\final_model.bin` 이 실행 위치 기준으로 있어야 한다. 더블클릭하면 창이 바로 닫힐 수 있다.
2. **VS Code 에서 F5** — 이 폴더를 `폴더 열기`로 열고 (C/C++ 확장 + gcc 가 PATH 에 필요) `pong_game.c` 를 연 채로 F5.
   `.vscode/tasks.json` 이 빌드하고, `.vscode/launch.json` 이 VS Code 아래 터미널에서 `pong_game.exe` 를 실행한다.
   gdb 로 디버깅하는 설정도 있지만 경로에 한글이 있으면 gdb 가 시작하지 못한다(영문 경로에서만 사용).
3. **직접 빌드**
   ```
   gcc -O2 -Wall -DPRED -o pong_game.exe pong_game.c ranking.c -lm
   ```
   ※ Visual Studio(MSVC) 빌드는 확인하지 못했다.

