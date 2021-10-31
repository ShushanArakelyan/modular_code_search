#!/bin/bash

source activate ccg_parser
cd /home/shushan/modular_code_search
python -u -m layout_assembly.main_with_batching_adamax  --device cuda:0 --data_dir "$1" --scoring_checkpoint "/home/shushan/finetuned_scoring_models/06-09-2021 20:21:51/model_3_ep_5.tar" --num_epochs 250 --version "$2" --lr "$3" --print_every 1000 --save_every 10000 --layout_net_version "classic" --valid_file_name '/home/shushan/CodeSearchNet/resources/ccg_parses_only/python/final/jsonl/valid/ccg_python_valid_0.jsonl.gz' --num_negatives "$4" --normalized_action
