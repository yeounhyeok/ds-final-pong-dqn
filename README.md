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
승률이 AI 실력인지 상대 약함인지는 아직 검증 전(추적 정책 기준선 필요). 플레이 화면 미구현.

## v2 (dqn-v2-shaping 브랜치)
착지 예측 입력 + 포텐셜 shaping으로 추적 상대 대비 승률을 올린 버전. 배포 모델 `models/final_model.bin`은 시험 시드 10000점에서 99.9%(선택된 체크포인트 기준, 학습 수렴 성능 아님).
빌드·재현·한계는 [EXPERIMENTS.md](EXPERIMENTS.md) 참고. `make pong_dqn_v2 && ./select_model models/final_model.bin`.

## 플레이 (v2 모델과 대전)
- **브라우저(권장):** `web/pong.html` 을 브라우저로 열기 (단일 파일, 모델 내장, 설치 불필요). ↑/↓ 또는 W/S 를 누르는 동안 이동, 떼면 정지. 난이도 1/2/3. 재생성: `python3 web/build.py`
- **터미널:** `make play && ./play [1|2|3]` (키 반복 입력이 끊기면 정지, 비대화형 검증 `./play selftest`)
- 난이도 = 같은 모델의 착지 예측 입력에 오차 σ(0.30/0.15/0) 주입. 규칙 추적 상대 대비 승률 약 29/69/99.9%이며 사람 대상 난이도는 미측정
- **관전 모드:** 상단 '모드'에서 `AI vs AI 관전` (같은 모델이 양쪽 조종, 오른쪽은 좌우 반전 입력, 대칭 물리, 1/2/4배속)
- **시각화:** 신경망 실시간 활성화(입력→은닉64→은닉64→Q) + 가중치 4,867개 히트맵
- `web/core.js` 는 `pong_dqn_v2.c` 의 환경·추론을 JS 로 이식한 것 (C 와 Q값 최대차 1.2e-7 확인)
