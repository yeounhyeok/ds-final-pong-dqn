#!/bin/bash
cd ~/work/ds-final/pong-dqn
TAG=rb$1_dd$2_s$3 RB=$1 DDQN=$2 LR=3e-4 TSYNC=1000 GAMMA=0.95 ./pong_dqn train 300000 $3 > exp/rb$1_dd$2_s$3.log 2>&1
