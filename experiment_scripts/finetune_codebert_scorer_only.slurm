#!/bin/bash
source activate ccg_parser
cd /home/shushan/modular_code_search
python -m scoring.finetune_codebert  --device cuda:0 --data_dir /home/shushan/datasets/CodeSearchNet/resources/tagged_data_py2_py3/python/final/jsonl/train --num_epochs 20 --scorer_only
