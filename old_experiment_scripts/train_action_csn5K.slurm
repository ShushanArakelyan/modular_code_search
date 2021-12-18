#!/bin/bash

export TRANSFORMERS_OFFLINE=1
source activate ccg_parser
cd /project/hauserc_374/shushan/modular_code_search
python -u -m layout_assembly.main_with_batching_10K_codebert_classifier  --device cuda:0 --data_dir "$1" --checkpoint_dir "/project/hauserc_374/shushan/modular_code_search/model_checkpoints/action/" --summary_writer_dir "/project/hauserc_374/shushan/modular_code_search/runs" --codebert_valid_results_dir "/project/hauserc_374/shushan/CodeBERT/CodeBERT/codesearch/results/valid/" --scoring_checkpoint "/project/hauserc_374/shushan/finetuned_scoring_models/08-11-2021 09:56:34/model_18_ep_0.tar" --num_epochs 250 --version "$2" --lr "$3" --print_every 1000 --save_every 5000 --layout_net_version "classic" --valid_file_name '/project/hauserc_374/shushan/CodeSearchNet/resources/ccg_parses_only/python/final/jsonl/valid/ccg_python_valid_0.jsonl' --num_negatives "$4" --adamw "$5" --example_count "$6" --use_lr_scheduler --dropout "$7" --normalized_action
