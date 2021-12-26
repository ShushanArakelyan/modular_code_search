#!/bin/bash

export TRANSFORMERS_OFFLINE=1
source activate ccg_parser
cd /project/hauserc_374/shushan/modular_code_search
python -u -m layout_assembly.main2  --device cuda:0 --data_dir "$1" --checkpoint_dir "/project/hauserc_374/shushan/modular_code_search/model_checkpoints/action/" --summary_writer_dir "/project/hauserc_374/shushan/modular_code_search/runs" --scoring_checkpoint "/project/hauserc_374/shushan/finetuned_scoring_models/06-09-2021 20:21:51/model_3_ep_5.tar" --num_epochs 2500 --num_epochs_pretraining 1  --lr "$2" --print_every 1000 --clip_grad_value 1 --valid_file_name '/project/hauserc_374/shushan/CodeSearchNet/resources/ccg_parses_only/python/final/jsonl/valid/ccg_python_valid_0.jsonl' --num_negatives "$3" --adamw "$4" --example_count "$5" --use_lr_scheduler --dropout "$6" --p_at_k 1 --p_at_k 3 --p_at_k 5 --distractor_set_size 100 --do_train --patience 50 --batch_size 10 --layout_net_training_ckp "/project/hauserc_374/shushan/modular_code_search/model_checkpoints/action/24-12-2021 12:39:22/pretrain/best_model.tar"
