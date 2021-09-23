#!/bin/bash

source activate ccg_parser
cd /home/shushan/modular_code_search
python -u -m eval.eval  --device cuda:0 --valid_file_name '/home/shushan/CodeSearchNet/resources/ccg_parses_only/python/final/jsonl/valid/ccg_python_valid_0.jsonl.gz' --scoring_checkpoint '/home/shushan/finetuned_scoring_models/06-09-2021 20:23:12/model_3_ep_5.tar' --eval_version "tf-idf" --neg_sample_count 999 --oracle_neg_sample_count 9 --layoutnet_checkpoint "$1" --action_version "$2"
