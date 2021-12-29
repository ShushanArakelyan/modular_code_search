#!/bin/bash

source activate ccg_parser
cd /home/shushan/modular_code_search
python -m scoring.finetune_codebert  --num_epochs 20  --device cuda:0 --data_dir /home/shushan/datasets/CodeSearchNet/resources/ccg_parses_only/python/final/jsonl/train/ --include_mismatched_pairs
