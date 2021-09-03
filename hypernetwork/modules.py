import torch

from scoring.embedder import Embedder


class ActionModule:
    pass


class ScoringModule:
    def __init__(self, device, checkpoint=None):
        self.embedder = Embedder(device, model_eval=False)
        self.scorer = torch.nn.Sequential(torch.nn.Linear(self.embedder.get_dim() * 2, self.embedder.get_dim()),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(self.embedder.get_dim(), 1)).to(device)

        # TODO: not sure about these
        # self.op = torch.optim.Adam(list(self.scorer.parameters()) + list(self.embedder.model.parameters()), lr=1e-8)
        self.loss = torch.nn.BCEWithLogitsLoss(reduction='sum')

        self.device = device
        if checkpoint:
            models = torch.load(checkpoint, map_location=device)
            self.scorer.load_state_dict(models['scorer'])
            self.scorer = self.scorer.to(device)
            self.embedder.model.load_state_dict(models['embedder'])
            self.embedder.model = self.embedder.model.to(device)
            # self.op.load_state_dict(models['optimizer'])

    def forward(self, sample):
        query, code, static_tags, regex_tags, ccg_parse = sample
        embedder_out = self.embedder.embed(query, code)
        if embedder_out is None:
            return None
        self.word_token_id_mapping, self.word_token_embeddings, self.code_token_id_mapping, \
        self.code_embedding, _, self.truncated_code_tokens, self.cls_token_embedding = embedder_out
        if self.word_token_id_mapping.size == 0 or self.code_token_id_mapping.size == 0:
            return None

        tiled_emb = self.cls_token_embedding.repeat(len(self.truncated_code_tokens), 1)
        forward_input = torch.cat((tiled_emb, self.code_embedding), dim=1)
        token_count = max(self.code_token_id_mapping[-1])
        scorer_out = torch.sigmoid(self.scorer.forward(forward_input)).squeeze().cpu().detach().numpy()[:token_count]
        return scorer_out

    def compute_loss(self):
        pass
