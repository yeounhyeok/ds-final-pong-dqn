#!/bin/bash
cd "$(dirname "$0")"; rm -f results4.txt
one(){ IFS='|' read -r tag bin steps seed envs <<< "$1"
  env TAG=$tag $envs ./$bin train $steps $seed > $tag.out 2>&1; grep FINAL $tag.out >> results4.txt; }
export -f one
cat jobs4.txt | xargs -P 2 -d '\n' -I{} bash -c 'one "{}"'
echo ALLDONE >> results4.txt
