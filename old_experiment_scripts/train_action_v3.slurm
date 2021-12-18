#!/bin/bash

source activate ccg_parser
cd /home/shushan/modular_code_search
python -m layout_assembly.main_v3  --device cuda:0 --data_file /home/shushan/datasets/CodeSearchNet/resources/ccg_parses_only/python/final/jsonl/train/ccg_train_0.jsonl.gz --scoring_checkpoint "/home/shushan/finetuned_scoring_models/06-09-2021 20:21:51/model_3_ep_5.tar" --num_epochs 5
