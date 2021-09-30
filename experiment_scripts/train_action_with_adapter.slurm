#!/bin/bash

source activate ccg_parser
cd /home/shushan/modular_code_search
python -u -m layout_assembly.main  --device cuda:0 --data_dir "$1" --scoring_checkpoint '/home/shushan/finetuned_scoring_models/06-09-2021 20:23:12/model_3_ep_5.tar' --num_epochs 50 --version "$2" --lr "$3" --save_every 50000 --layout_net_version "with_adapters" --valid_file_name '/home/shushan/CodeSearchNet/resources/ccg_parses_only/python/final/jsonl/valid/ccg_python_valid_0.jsonl.gz'
