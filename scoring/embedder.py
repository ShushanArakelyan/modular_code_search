import torch
import numpy as np

from transformers import AutoTokenizer, AutoModel

from third_party.CodeBERT.CodeBERT.codesearch.utils import convert_examples_to_features, InputExample

class Embedder(object):
    def __init__(self, device=None):
        self.max_seq_length = 128
        if device:
            torch.cuda.set_device(device)
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.init_model()

    def init_model(self):
        """Initializes model and tokenizer from pre-trained model checkpoint."""
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = AutoModel.from_pretrained("microsoft/codebert-base").to(self.device)
        self.model.eval()

    def get_feature_inputs(self, tokens):
        tokens = [InputExample(0, tokens, label="0")]
        """Converts the input tokens into CodeBERT inputs."""
        features = convert_examples_to_features(tokens, ["0", "1"], self.max_seq_length, self.tokenizer,
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

    def get_token_embedding(self, inputs, embeddings, token):
        """From all the embeddings filters out the embedding of the desired token."""
        subtokenized_text = self.tokenizer.tokenize(token)
        subtoken_ids = self.tokenizer.convert_tokens_to_ids(subtokenized_text)
        embed_id = []
        for idx in subtoken_ids:
            embed_id.extend((inputs['input_ids'][0] == idx).nonzero().cpu().numpy())
        embed_id = np.asarray(embed_id).ravel()
        embed = embeddings[embed_id]

        # for instances where RobertaTokenizer split words into multiple sub-word token, we take the average of them
        embed = torch.mean(embed.T, 1, True)
        return embed.squeeze(1)

    def get_sample_embedding(self, query, code):
        """Gets the embeddings of both code and docstring."""
        qinputs = self.get_feature_inputs(query)
        cinputs = self.get_feature_inputs(code)
        with torch.no_grad():
            query_embeddings = self.embedder.get_embeddings(qinputs)
            code_embeddings = self.embedder.get_embeddings(cinputs)
        return query_embeddings, code_embeddings
