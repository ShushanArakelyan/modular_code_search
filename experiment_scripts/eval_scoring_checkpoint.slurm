#!/bin/bash

source activate ccg_parser
cd /home/shushan/modular_code_search
python -u -m scoring.eval_finetuned_codebert  --load_from_checkpoint "$1" --device cuda:0 --version CLS --embed_separately