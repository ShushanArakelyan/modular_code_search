#!/bin/bash

source activate ccg_parser
cd /home/shushan/modular_code_search
python -u -m eval.surface_eval  --layoutnet_checkpoint "$1" --action_version "$2" --eval_version codebert
