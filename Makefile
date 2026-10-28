all: pong_dqn
pong_dqn: pong_dqn.c
	gcc -O2 -Wall -o $@ $< -lm
gradcheck: pong_dqn.c
	gcc -O2 -DREAL=double -o pong_dqn_dbl $< -lm && ./pong_dqn_dbl gradcheck

gradcheck-full: gradcheck_full.c pong_dqn.c
	gcc -O2 -o gradcheck_full gradcheck_full.c -lm && ./gradcheck_full
