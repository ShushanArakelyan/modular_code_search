import torch
import numpy as np

from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
from utils.codebert_utils import convert_examples_to_features


class Embedder(object):
    def __init__(self, device, model_checkpoint):
        self.max_seq_length = 128
        self.model_checkpoint = model_checkpoint
        self.device = device
        self.init_model()

    def init_model(self):
        """Initializes model and tokenizer from pre-trained model checkpoint."""
        config_class, model_class, tokenizer_class = (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
        tokenizer_name = 'roberta-bself.devicease'
        self.tokenizer = tokenizer_class.from_pretrained(tokenizer_name, do_lower_case=True, add_prefix_space=True)
        self.model = model_class.from_pretrained(self.model_checkpoint).to(self.device)
        self.model.eval()

    def get_feature_inputs(self, tokens):
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
