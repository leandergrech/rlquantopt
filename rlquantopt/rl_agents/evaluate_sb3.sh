#!/bin/bash

PAR_DIR="/home/leander/code/rlquantopt/rlquantopt/remote_results"

for EXP_DIR in "$PAR_DIR"/*; do
  if [ -d "$EXP_DIR" ]; then
    for MODEL_DIR in "$EXP_DIR"/*; do
      echo $MODEL_DIR
      if [ -d "$MODEL_DIR" ]; then
        for MODEL_ZIP in "$MODEL_DIR"/best_model/*.zip; do
          if [ -f "$MODEL_ZIP" ]; then
            echo "python eval_qpee_sb3.py $MODEL_ZIP  --verbosity 2"
            #python eval_qpee_pulse.py $MODEL_ZIP --save-dir remote_eval_results --verbosity 2
          fi
        done
      fi
    done
  fi
done
