#!/bin/bash

source activate ccg_parser
cd /home/shushan/modular_code_search
python -u -m layout_assembly.main_codebert_oracle_temp --device cuda:0 --data_dir '/home/shushan/CodeSearchNet/resources/ccg_parses_only/python/final/jsonl/train/ccg_train_0.jsonl.gz' --scoring_checkpoint '/home/shushan/finetuned_scoring_models/06-09-2021 20:23:12/model_3_ep_5.tar' --num_epochs 250 --version "$1" --lr "$2" --print_every 1000 --save_every 50000 --layout_net_version "classic" --valid_file_name '/home/shushan/CodeSearchNet/resources/ccg_parses_only/python/final/jsonl/valid/ccg_python_valid_0.jsonl.gz' --oracle_negatives_dir "/home/shushan/codebert_oracle_scores" --num_negatives 9
