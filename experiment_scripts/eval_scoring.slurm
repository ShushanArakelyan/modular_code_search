#!/bin/bash

source activate ccg_parser
cd /home/shushan/modular_code_search
python -u -m scoring.eval_finetuned_codebert --valid_file_name /home/shushan/datasets/CodeSearchNet/resources/ccg_parses_only/python/final/jsonl/valid/ccg_python_valid_0.jsonl.gz --checkpoint_dir "$1" --device "cuda:0" --version CLS $2