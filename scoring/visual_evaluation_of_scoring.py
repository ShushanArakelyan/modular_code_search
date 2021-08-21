import torch
from scoring.embedder import Embedder
from finetune_codebert import *
import pandas as pd
from IPython.core.display import HTML

import bleach

device = 'cuda:0'
dim = 768

embedder = Embedder(device)
scorer = torch.nn.Sequential(torch.nn.Linear(dim*2, dim),
                   torch.nn.ReLU(),
                   torch.nn.Linear(dim, 1)).to(device)


checkpoint = '/home/shushan/finetuned_scoring_models/model_0_ep.tar'
models = torch.load(checkpoint)

scorer.load_state_dict(models['scorer'])
embedder.model.load_state_dict(models['embedder'])

input_file_name = f'/home/shushan/datasets/CodeSearchNet/resources/tagged_data_py2_py3/python/final/jsonl/train/python_train_8.jsonl.gz'
data = pd.read_json(input_file_name, lines=True)

class TokenVal(object):
    def __init__(self, token, val):
        self.token = token
        self.val = val

    def __str__(self):
        return self.token
    
    
def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

def color_tokenvals(s, idx):
    r = 255-int(s.val*255)
    color_tuple = [r, r, r]
    color_tuple[idx] = 255
    color = rgb_to_hex(tuple(color_tuple))
    return 'background-color: %s' % color

def tokensvals_to_html(token_vals, color):
    html = '<pre style="float:left; max-width:350px; margin-right:10px"><code>'
    
    for i, tv in enumerate(token_vals):
        if tv.token != '<s>':
            html += '<span title="{}" style="{}">{}</span>'.format(i, color_tokenvals(tv, color), bleach.clean(embedder.tokenizer.convert_tokens_to_string(tv.token)))
    
    html += "</code></pre>"
    
    return html

it = 7000


pair = 'correct_pair'
# pair = 'incorrect_pair'


true_scores_dfs = {}
predicted_scores_dfs = {}
if pair == 'correct_pair':
    doc = data['docstring_tokens'][it]
    code = data['alt_code_tokens'][it]
    static_tags = data['static_tags'][it]
    regex_tags = data['regex_tags'][it]
else:
    np.random.seed(it)
    random_idx = np.random.choice(np.arange(len(data)), 1)[0]
    doc = data['docstring_tokens'][it]
    code = data['alt_code_tokens'][random_idx]
    static_tags = data['static_tags'][random_idx]
    regex_tags = data['regex_tags'][random_idx]
noun_tokens = extract_noun_tokens(' '.join(doc))
# converting the docstring and code tokens into CodeBERT inputs
# CodeBERT inputs are limited by 512 tokens, so this will truncate the inputs
inputs = embedder.get_feature_inputs(' '.join(doc), ' '.join(code))
separator = np.where(
    inputs['input_ids'][0].cpu().numpy() == embedder.tokenizer.sep_token_id)[0][0]
# ignore CLS tokens at the beginning and at the end
query_token_ids = inputs['input_ids'][0][1:separator]
code_token_ids = inputs['input_ids'][0][separator + 1:-1]

# get truncated version of code and query
truncated_code_tokens = embedder.tokenizer.convert_ids_to_tokens(code_token_ids)
truncated_query_tokens = embedder.tokenizer.convert_ids_to_tokens(query_token_ids)

# mapping from CodeBERT tokenization to our dataset tokenization
noun_token_id_mapping = np.asarray(get_word_to_roberta_tokens(doc, 
                                                              truncated_query_tokens, 
                                                              noun_tokens, embedder),
                                   dtype=object)

code_token_id_mapping = np.asarray(get_code_to_roberta_tokens(code, 
                                                              truncated_code_tokens, 
                                                              embedder), 
                                   dtype=object)
# get CodeBERT embedding for the example
embedding = embedder.get_embeddings(inputs)
query_embedding, code_embedding = embedding[1:separator], embedding[separator + 1:-1]
noun_token_embeddings = filter_embedding_by_id(query_embedding, noun_token_id_mapping)

# extract ground truth pairs and get scores for all pairs
for nti, nte, nt in zip(noun_token_id_mapping, noun_token_embeddings, noun_tokens):
    nte = nte.unsqueeze(0)
    # check for regex and static matches
    pos_sample_idxs = []
    idxs = get_matched_labels_binary_v2(code[:len(code_token_id_mapping)], 
                                        code_token_id_mapping, nt).nonzero()[0]
    if idxs.size:
        pos_sample_idxs.extend(idxs)
    idxs = get_static_labels_binary(code_token_id_mapping, nt, static_tags).nonzero()[0]
    if idxs.size:
        pos_sample_idxs.extend(idxs)
    idxs = get_regex_labels_binary(code_token_id_mapping, nt, regex_tags).nonzero()[0]
    if idxs.size:
        pos_sample_idxs.extend(idxs)
    # add positive example
    token_vals = [TokenVal(c, 0) for c in truncated_code_tokens]
    if len(pos_sample_idxs) > 0:
        for pos_sample_idx in pos_sample_idxs:
            token_vals[pos_sample_idx] = TokenVal(truncated_code_tokens[pos_sample_idx], 1)
    true_scores_dfs[nt] = token_vals

    neg_sample_idxs = np.arange(len(truncated_code_tokens))
    token_vals = []
    for neg_sample_idx in neg_sample_idxs:
        neg_sample_selected = torch.index_select(
            code_embedding, index=torch.LongTensor([neg_sample_idx]).to(device), 
                                                 dim=0)
        forward_input = torch.cat((nte, neg_sample_selected), dim=1)
        scorer_out = -np.log(torch.sigmoid(scorer.forward(forward_input)).cpu().detach().numpy())[0][0]
        token_vals.append(TokenVal(truncated_code_tokens[neg_sample_idx], scorer_out))
    predicted_scores_dfs[nt] = token_vals
    display(HTML(f'<h2>{nt}</h2>' + tokensvals_to_html(predicted_scores_dfs[nt], 0) + tokensvals_to_html(true_scores_dfs[nt], 1)))