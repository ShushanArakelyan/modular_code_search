#!/bin/bash

export TRANSFORMERS_OFFLINE=1
source activate ccg_parser
cd /home/shushan/modular_code_search
python -u -m layout_assembly.main2  --device cuda:0 --data_dir "/home/shushan/CodeSearchNet/resources/ccg_parses_only_v2/python/final/jsonl/train/ccg_python_0.jsonl.gz" --checkpoint_dir "/home/shushan/modular_code_search/model_checkpoints/action/" --summary_writer_dir "/home/shushan/modular_code_search/runs" --scoring_checkpoint "$1" --num_epochs 250 --num_epochs_pretraining 1  --lr "$2" --print_every 1000 --clip_grad_value 1 --valid_file_name '/home/shushan/CodeSearchNet/resources/ccg_parses_only/python/final/jsonl/valid/ccg_python_valid_0.jsonl' --num_negatives "$3" --adamw "$4" --example_count "$5" --use_lr_scheduler --dropout "$6" --p_at_k 1 --p_at_k 3 --p_at_k 5 --distractor_set_size 100 --do_pretrain --do_train --patience 250 "$7"
