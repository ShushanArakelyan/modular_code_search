import numpy as np
import torch
import warnings
from transformers import AutoTokenizer
from codebert_embedder_v2.robertaforsequenceclassification_weighted import RobertaForSequenceClassification_weighted

from codebert_embedder_v2.utils import convert_examples_to_features, convert_examples_to_features_with_scores, InputExample

max_seq_length = 512
dim = 768
tokenizer = None
model = None
classifier = None
device = None
initialized = False
sep_embedding = None
cls_embedding = None
sep_value = 0.1
cls_value = -0.1


def init_embedder(_device, load_finetuned=False, checkpoint_dir=False):
    global device, initialized
    global tokenizer, model, classifier
    global sep_embedding, cls_embedding
    print(_device)
    if 'cuda' in _device:
        torch.cuda.set_device(_device)
        device = _device
    else:
        device = 'cpu'
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    if load_finetuned:
#         model_dir = '/home/anna/CodeBERT/CodeBERT/codesearch/models/'
        classifier = RobertaForSequenceClassification_weighted.from_pretrained(checkpoint_dir).to(device)
        model = classifier.roberta
    else:
        classifier = RobertaForSequenceClassification_weighted.from_pretrained("microsoft/codebert-base").to(device)
        model = classifier.roberta
    initialized = True
    warnings.warn("The weights of the CodeBERT embedder in codebert_embedder module have been reset")
    sep_embedding = (torch.ones((1, dim)) * sep_value).to(device)
    cls_embedding = (torch.ones((1, dim)) * cls_value).to(device)


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


def get_feature_inputs_classifier(queries, codes, scores):
    examples = [InputExample(0, text_a=q, text_b=c, label="0") for q, c in zip(queries, codes)]
    """Converts the input tokens into CodeBERT inputs."""
    features, scores = convert_examples_to_features_with_scores(examples, scores, ["0", "1"], max_seq_length, tokenizer,
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
            'token_type_ids': None}, scores


def get_feature_inputs_batch(queries, codes):
    examples = [InputExample(0, text_a=q, text_b=c, label="0") for q, c in zip(queries, codes)]
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


def get_embeddings(inputs, batch=False):
    """Gets the embeddings of all the tokens of the input sentence."""
    output = model(**inputs, output_hidden_states=True)
    embeddings = output['hidden_states']
    embeddings = inputs['attention_mask'].unsqueeze(dim=2) * embeddings[-1]
    if not batch:
        embeddings = embeddings.squeeze()
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


def embed_batch(docs, codes, return_separators=False):
    inputs = get_feature_inputs_batch(docs, [' '.join(c) for c in codes])
    embedding = get_embeddings(inputs, True)
    query_embeddings = embedding.index_select(dim=1, index=torch.LongTensor([0]).to(device))
    sep_tokens = (inputs['input_ids'] == tokenizer.sep_token_id).nonzero(as_tuple=False)
    counts = torch.unique(sep_tokens[:, 0], return_counts=True)[1]
    first_sep_token_index = [sum(counts[:i]) for i in range(len(counts))]
    last_sep_token_index =  [sum(counts[:i + 1]) - 1 for i in range(len(counts))]
    separator = sep_tokens.index_select(dim=0, index=torch.LongTensor(first_sep_token_index).to(device))[:, 1]
    if return_separators:
        code_embeddings = [torch.cat([sep_embedding, 
                                      embedding[i, sep_tokens[first_sep_token_index[i]][1] + 1:
                                                   sep_tokens[last_sep_token_index[i]][1], :],
                                      sep_embedding], dim=0).unsqueeze(dim=0)
            for i in range(embedding.shape[0])]
        code_embeddings = torch.cat([torch.nn.functional.pad(
            code_embeddings[i], (0, 0, 0, max_seq_length - code_embeddings[i].shape[1] + 1, 0, 0), 'constant', 0) 
                                     for i in range(len(code_embeddings))], dim=0)
    else:
        code_embeddings = torch.cat([torch.nn.functional.pad(embedding[i, separator[i] + 1:, :],
                                                             (0, 0, 0, separator[i] + 1), 'constant', 0).unsqueeze(dim=0)
                                     for i in range(separator.shape[0])], dim=0)
    return query_embeddings, code_embeddings


def embed_in_list(docs, codes):
    inputs = get_feature_inputs_batch(docs, [' '.join(c) for c in codes])
    embedding = get_embeddings(inputs, True)
    query_embeddings = embedding.index_select(dim=1, index=torch.LongTensor([0]).to(device))
    sep_tokens = (inputs['input_ids'] == tokenizer.sep_token_id).nonzero(as_tuple=False)
    code_embeddings = [embedding[i, sep_tokens[i, 1] + 1 : sep_tokens[i + 1, 1], :] for i in range(embedding.shape[0])]
    return query_embeddings, code_embeddings


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
    cls_emb = embedding.index_select(dim=0, index=torch.LongTensor(np.arange(0, 1)).to(device))
    query_embedding = embedding.index_select(dim=0, index=torch.LongTensor(np.arange(1, separator)).to(device))
    code_embedding = embedding.index_select(dim=0, index=torch.LongTensor(
        np.arange(separator + 1, embedding.shape[0] - 1)).to(device))
    token_embeddings = filter_embedding_by_id(query_embedding, token_id_mapping)

    out_tuple = (token_id_mapping, token_embeddings, code_token_id_mapping, code_embedding, truncated_query_tokens,
                 truncated_code_tokens, cls_emb)
    return out_tuple