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
