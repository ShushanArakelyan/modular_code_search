from itertools import chain

import torch
import torch.nn as nn

import codebert_embedder_v2 as embedder
from layout_assembly.utils import ProcessingException, FC2, FC2_normalized, init_weights
from action.action_v8 import ActionModule_v8_one_input, ActionModule_v8_two_inputs


class ActionModule_v8_one_input_action_ablation(ActionModule_v8_one_input):
    def forward(self, _, args, __, precomputed_embeddings):
        raise Exception("This ablation study requires modification of layoutnet!")


class ActionModule_v8_two_inputs_action_ablation(ActionModule_v8_two_inputs):
    def forward(self, _, args, __, precomputed_embeddings):
        raise Exception("This ablation study requires modification of layoutnet!")
