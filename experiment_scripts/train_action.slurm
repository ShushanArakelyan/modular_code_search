#!/bin/bash

source activate ccg_parser
cd /home/shushan/modular_code_search
python -u -m layout_assembly.main  --device cuda:0 --data_dir "$1" --scoring_checkpoint '/home/shushan/finetuned_scoring_models/06-09-2021 20:23:12/model_3_ep_5.tar' --num_epochs 250 --version "$2" --lr "$3" --print_every 1000 --save_every 50000 --layout_net_version "classic"
