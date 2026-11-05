#!/bin/bash
cd "$(dirname "$0")"; rm -f results2.txt
one(){ IFS='|' read -r tag bin steps seed envs <<< "$1"
  env TAG=$tag $envs ./$bin train $steps $seed > $tag.out 2>&1; grep FINAL $tag.out >> results2.txt; }
export -f one
cat jobs2.txt | xargs -P 2 -d '\n' -I{} bash -c 'one "{}"'
echo ALLDONE >> results2.txt
