all: pong_dqn
pong_dqn: pong_dqn.c
	gcc -O2 -Wall -o $@ $< -lm
gradcheck: pong_dqn.c
	gcc -O2 -DREAL=double -o pong_dqn_dbl $< -lm && ./pong_dqn_dbl gradcheck

gradcheck-full: gradcheck_full.c pong_dqn.c
	gcc -O2 -o gradcheck_full gradcheck_full.c -lm && ./gradcheck_full

# ---- v2: 착지 예측 입력(PRED) + 포텐셜 shaping ----
pong_dqn_v2: pong_dqn_v2.c
	gcc -O2 -Wall -DPRED -o $@ $< -lm
gradcheck-v2: pong_dqn_v2.c
	gcc -O2 -DREAL=double -DPRED -o pong_dqn_v2_dbl $< -lm && ./pong_dqn_v2_dbl gradcheck
select-model: tools/select_model.c pong_dqn_v2.c
	gcc -O2 -DPRED -o select_model tools/select_model.c -lm
difficulty: tools/difficulty.c pong_dqn_v2.c
	gcc -O2 -DPRED -o difficulty tools/difficulty.c -lm && ./difficulty

# ---- 터미널 대전: ./play [1|2|3] ----
play: play.c pong_dqn_v2.c ranking.c ranking.h
	gcc -O2 -Wall -DPRED -o $@ play.c ranking.c -lm

# ---- 랭킹 TOP 10 모듈 테스트 ----
ranking-test: tools/ranking_test.c ranking.c ranking.h
	gcc -O2 -Wall -Wextra -o ranking_test tools/ranking_test.c ranking.c && ./ranking_test
