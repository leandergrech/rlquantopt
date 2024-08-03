#!/bin/bash

CNT=20
SIDX=0

for i in {0..8}; do
  sbatch -o gridsearch_$i.out training_runner.sh --start-idx $SIDX --cnt $CNT
  SIDX=$((SIDX + CNT))
  sleep 2
done
