all: pong_dqn
pong_dqn: pong_dqn.c
	gcc -O2 -Wall -o $@ $< -lm
gradcheck: pong_dqn.c
	gcc -O2 -DREAL=double -o pong_dqn_dbl $< -lm && ./pong_dqn_dbl gradcheck
