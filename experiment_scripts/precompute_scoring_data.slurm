#!/bin/bash

source activate ccg_parser
cd /home/shushan/modular_code_search
python -m layout_assembly.precompute_scoring --num_negatives "$1" --neg_sampling_strategy "$2"