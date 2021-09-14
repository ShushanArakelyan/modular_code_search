from transformers import RobertaTokenizer, RobertaModel
import torch

max_seq_length = 512
dim = 768
device = None
tokenizer = None
model = None
initialized = False


def init_embedder(_device):
    global device, initialized
    global tokenizer, model
    if 'cuda' in _device:
        torch.cuda.set_device(_device)
        device = _device
    else:
        device = 'cpu'
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base').to(device)
    initialized = True


def forward(doc):
    inputs = tokenizer(doc, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state
