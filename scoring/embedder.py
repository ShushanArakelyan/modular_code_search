import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from third_party.CodeBERT.CodeBERT.codesearch.utils import convert_examples_to_features, InputExample


class Embedder(object):
    def __init__(self, device=None, model_eval=True):
        self.max_seq_length = 512
        self.dim = 768  # size of the output embedding
        self.tokenizer = None
        self.model = None

        if 'cuda' in device:
            torch.cuda.set_device(device)
            self.device = device
        else:
            self.device = 'cpu'
        self.init_model(model_eval)

    def get_dim(self):
        return self.dim

    def init_model(self, model_eval):
        """Initializes model and tokenizer from pre-trained model checkpoint."""
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = AutoModel.from_pretrained("microsoft/codebert-base").to(self.device)
        if model_eval:
            self.model.eval()

    def get_feature_inputs(self, query, code):
        examples = [InputExample(0, text_a=query, text_b=code, label="0")]
        """Converts the input tokens into CodeBERT inputs."""
        features = convert_examples_to_features(examples, ["0", "1"], self.max_seq_length, self.tokenizer,
                                                "classification", cls_token_at_end=False,
                                                cls_token=self.tokenizer.cls_token,
                                                sep_token=self.tokenizer.sep_token,
                                                cls_token_segment_id=1,
                                                pad_on_left=False,
                                                pad_token_segment_id=0)

        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(self.device)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).to(self.device)
        return {'input_ids': input_ids,
                'attention_mask': input_mask,
                'token_type_ids': None}

    def get_embeddings(self, inputs):
        """Gets the embeddings of all the tokens of the input sentence."""
        output = self.model(**inputs, output_hidden_states=True)
        embeddings = output['hidden_states']
        embeddings = inputs['attention_mask'].T * embeddings[-1].squeeze()
        return embeddings

    def get_orig_tokens_to_roberta_tokens(self, orig_tokens, codebert_tokens):
        rt_i = 0
        ct_i = 0
        last_end = 0
        output_tokens = []
        while rt_i < len(codebert_tokens) and ct_i < len(orig_tokens):
            if self.tokenizer.convert_tokens_to_string(codebert_tokens[:rt_i + 1]) == " ".join(
                    orig_tokens[:ct_i + 1]):
                current_token_idxs = np.arange(last_end, rt_i + 1)
                output_tokens.append(current_token_idxs)
                last_end = rt_i + 1
                ct_i += 1
            rt_i += 1
        return output_tokens

    
    def filter_embedding_by_id(self, query_embedding, token_ids):
        token_embeddings = []
        for ti in token_ids:
            te = torch.index_select(query_embedding, index=torch.LongTensor(ti.astype(int)).to(self.device), dim=0)
            token_embeddings.append(torch.unsqueeze(torch.mean(te, dim=0), 0))  # i am not sure, that the gradients get propagated through here
        token_embeddings = torch.cat(token_embeddings)
        return token_embeddings

    
    def embed(self, doc, code):
        # embed query and code, and get embeddings of tokens_of_interest from query, and max_len tokens from code.
        # converting the docstring and code tokens into CodeBERT inputs
        # CodeBERT inputs are limited by 512 tokens, so this will truncate the inputs
        inputs = self.get_feature_inputs(' '.join(doc), ' '.join(code))
        separator = np.where(
            inputs['input_ids'][0].cpu().numpy() == self.tokenizer.sep_token_id)[0][0]
        # ignore CLS tokens at the beginning and at the end
        query_token_ids = inputs['input_ids'][0][1:separator]
        code_token_ids = inputs['input_ids'][0][separator + 1:-1]

        # get truncated version of code and query
        truncated_code_tokens = self.tokenizer.convert_ids_to_tokens(code_token_ids)
        truncated_query_tokens = self.tokenizer.convert_ids_to_tokens(query_token_ids)

        # mapping from CodeBERT tokenization to our dataset tokenization
        token_id_mapping = np.asarray(self.get_orig_tokens_to_roberta_tokens(doc, truncated_query_tokens), dtype=object)

        code_token_id_mapping = np.asarray(self.get_orig_tokens_to_roberta_tokens(code,
                                                                                  truncated_code_tokens), dtype=object)
        # get CodeBERT embedding for the example
        if token_id_mapping.size == 0 or code_token_id_mapping.size == 0:
            return None

        embedding = self.get_embeddings(inputs)
        # query_embedding, code_embedding = embedding[1:separator], embedding[separator + 1:-1]
        cls_embedding = embedding.index_select(dim=0, index=torch.LongTensor(np.arange(0, 1)).to(self.device))
        query_embedding = embedding.index_select(dim=0, index=torch.LongTensor(np.arange(1, separator)).to(self.device))
        code_embedding = embedding.index_select(dim=0, index=torch.LongTensor(
            np.arange(separator + 1, embedding.shape[0] - 1)).to(self.device))
        token_embeddings = self.filter_embedding_by_id(query_embedding, token_id_mapping)

        out_tuple = (token_id_mapping, token_embeddings, code_token_id_mapping, code_embedding, truncated_query_tokens,
                     truncated_code_tokens, cls_embedding)
        return out_tuple
