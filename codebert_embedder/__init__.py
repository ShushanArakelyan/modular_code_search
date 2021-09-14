import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from third_party.CodeBERT.CodeBERT.codesearch.utils import convert_examples_to_features, InputExample

max_seq_length = 512
dim = 768
tokenizer = None
model = None
device = None
initialized = False


def init_embedder(_device):
    global device, initialized
    global tokenizer, model
    if 'cuda' in _device:
        torch.cuda.set_device(_device)
        device = _device
    else:
        device = 'cpu'
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base").to(device)
    initialized = True


def get_feature_inputs(query, code):
    examples = [InputExample(0, text_a=query, text_b=code, label="0")]
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


def get_embeddings(inputs):
    """Gets the embeddings of all the tokens of the input sentence."""
    output = model(**inputs, output_hidden_states=True)
    embeddings = output['hidden_states']
    embeddings = inputs['attention_mask'].T * embeddings[-1].squeeze()
    return embeddings


def get_orig_tokens_to_roberta_tokens(orig_tokens, codebert_tokens):
    rt_i = 0
    ct_i = 0
    last_end = 0
    output_tokens = []
    while rt_i < len(codebert_tokens) and ct_i < len(orig_tokens):
        if tokenizer.convert_tokens_to_string(codebert_tokens[:rt_i + 1]) == " ".join(
                orig_tokens[:ct_i + 1]):
            current_token_idxs = np.arange(last_end, rt_i + 1)
            output_tokens.append(current_token_idxs)
            last_end = rt_i + 1
            ct_i += 1
        rt_i += 1
    return output_tokens


def filter_embedding_by_id(query_embedding, token_ids):
    token_embeddings = []
    for ti in token_ids:
        te = torch.index_select(query_embedding, index=torch.LongTensor(ti.astype(int)).to(device), dim=0)
        token_embeddings.append(torch.unsqueeze(torch.mean(te, dim=0),
                                                0))  # i am not sure, that the gradients get propagated through here
    token_embeddings = torch.cat(token_embeddings)
    return token_embeddings


def embed(doc, code, fast=False):
    # embed query and code, and get embeddings of tokens_of_interest from query, and max_len tokens from code.
    # converting the docstring and code tokens into CodeBERT inputs
    # CodeBERT inputs are limited by 512 tokens, so this will truncate the inputs
    inputs = get_feature_inputs(' '.join(doc), ' '.join(code))
    separator = np.where(
        inputs['input_ids'][0].cpu().numpy() == tokenizer.sep_token_id)[0][0]

    # ignore CLS tokens at the beginning and at the end
    query_token_ids = inputs['input_ids'][0][1:separator]
    code_token_ids = inputs['input_ids'][0][separator + 1:-1]

    # get truncated version of code and query
    truncated_code_tokens = tokenizer.convert_ids_to_tokens(code_token_ids)
    truncated_query_tokens = tokenizer.convert_ids_to_tokens(query_token_ids)

    # mapping from CodeBERT tokenization to our dataset tokenization
    token_id_mapping = np.asarray(get_orig_tokens_to_roberta_tokens(doc, truncated_query_tokens), dtype=object)
    if token_id_mapping.size == 0:
        return None
    if not fast:
        code_token_id_mapping = np.asarray(get_orig_tokens_to_roberta_tokens(code,
                                                                                  truncated_code_tokens),
                                           dtype=object)
        if code_token_id_mapping.size == 0:
            return None
    else:
        code_token_id_mapping = None
    # get CodeBERT embedding for the example

    embedding = get_embeddings(inputs)
    # query_embedding, code_embedding = embedding[1:separator], embedding[separator + 1:-1]
    cls_embedding = embedding.index_select(dim=0, index=torch.LongTensor(np.arange(0, 1)).to(device))
    query_embedding = embedding.index_select(dim=0, index=torch.LongTensor(np.arange(1, separator)).to(device))
    code_embedding = embedding.index_select(dim=0, index=torch.LongTensor(
        np.arange(separator + 1, embedding.shape[0] - 1)).to(device))
    token_embeddings = filter_embedding_by_id(query_embedding, token_id_mapping)

    out_tuple = (token_id_mapping, token_embeddings, code_token_id_mapping, code_embedding, truncated_query_tokens,
                 truncated_code_tokens, cls_embedding)
    return out_tuple
