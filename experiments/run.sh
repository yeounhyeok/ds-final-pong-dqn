#!/bin/bash
# 사용: run.sh <steps>   — 격자 8개 병렬 실행, 결과는 results.txt
cd "$(dirname "$0")"; rm -f results.txt
for g in 0.95 0.99; do for h in 0.3 0.0; do for sd in 1 2; do
  ( TAG=g${g}_h${h}_s${sd} GAMMA=$g HIT_R=$h ./exp train $1 $sd > g${g}_h${h}_s${sd}.out 2>&1; grep FINAL g${g}_h${h}_s${sd}.out >> results.txt ) &
done; done; done
wait; echo ALLDONE >> results.txt
