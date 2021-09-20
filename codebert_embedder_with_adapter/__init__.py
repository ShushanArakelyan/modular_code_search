import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from third_party.CodeBERT.CodeBERT.codesearch.utils import convert_examples_to_features, InputExample

from codebert_embedder_with_adapter.roberta_with_adapter import RobertaWithAdaptersConfig, RobertaModelWithAdapter, ParameterGenerator, RobertaWithHyperAdapters

max_seq_length = 512
dim = 768
tokenizer = None
model = None
config = None
codebert = None
param_generator = None
device = None
initialized = False


def init_embedder(_device):
    global device, initialized
    global tokenizer, model, config, codebert, param_generator
    if 'cuda' in _device:
        torch.cuda.set_device(_device)
        device = _device
    else:
        device = 'cpu'
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    config = RobertaWithAdaptersConfig.from_pretrained('microsoft/codebert-base')
    codebert = RobertaModelWithAdapter.from_pretrained("microsoft/codebert-base", config=config, device=device).to(device)
    param_generator = ParameterGenerator(config, device)
    model = RobertaWithHyperAdapters(codebert, param_generator, config, device=device)
    initialized = True

    
def get_feature_inputs(code):
    examples = [InputExample(0, text_a=' ', text_b=code, label="0")]
    """Converts the input tokens into CodeBERT inputs."""
    features = convert_examples_to_features(examples, ["0", "1"], max_seq_length, tokenizer,
                                            "classification", cls_token_at_end=False,
                                            cls_token=tokenizer.cls_token,
                                            sep_token=tokenizer.sep_token,
                                            cls_token_segment_id=1,
                                            pad_on_left=False,
                                            pad_token_segment_id=0)

    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(device)
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).to(device)
    return {'input_ids': input_ids,
            'attention_mask': input_mask,
            'token_type_ids': None}


def get_embedding(hyperparam, inputs, batch=False):
    """Gets the embeddings of all the tokens of the input sentence."""
    output = model(**inputs, output_hidden_states=True, verb_embedding=hyperparam)
    embeddings = output['hidden_states']
    embeddings = inputs['attention_mask'].unsqueeze(dim=2) * embeddings[-1]
#     if not batch:
#         embeddings = embeddings.squeeze()
    return embeddings
  

def embed(hyperparam, code):
    inputs = get_feature_inputs(code)
    embedding = get_embedding(hyperparam, inputs, True)
    separator = (inputs['input_ids'] == tokenizer.sep_token_id).nonzero(as_tuple=False)[0][1]
    code_embeddings = torch.nn.functional.pad(embedding[:, separator + 1:, :],
                                                         (0, 0, 0, separator + 1), 'constant', 0)
    return code_embeddings