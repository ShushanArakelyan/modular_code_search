#!/bin/bash

export TRANSFORMERS_OFFLINE=1
source activate ccg_parser
cd /project/hauserc_374/shushan/modular_code_search
python -u -m layout_assembly.main_with_batching_10K_codebert_classifier  --device cuda:0 --data_dir "$1" --checkpoint_dir "/project/hauserc_374/shushan/modular_code_search/model_checkpoints/action/" --summary_writer_dir "/project/hauserc_374/shushan/modular_code_search/runs" --codebert_valid_results_dir "" --scoring_checkpoint "/project/hauserc_374/shushan/cosqa_scoring_models/01-10-2021 23:34:20/model_18_ep_0.tar" --num_epochs 250 --version "$2" --lr "$3" --print_every 1000 --save_every 10000 --layout_net_version "classic" --valid_file_name 'None' --num_negatives "$4" --normalized_action --adamw "$5" --example_count "$6" --dropout "$7" --use_lr_scheduler
