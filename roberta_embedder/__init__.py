import torch
from transformers import RobertaTokenizer, RobertaModel

from layout_assembly.utils import ProcessingException

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
    with torch.no_grad():
        inputs = tokenizer(doc, return_tensors="pt")
        if inputs['input_ids'].shape[1] <= 2:
            print(doc)
            raise ProcessingException()
        outputs = model(input_ids=inputs['input_ids'][:, 1:-1].to(device),
                        attention_mask=inputs['attention_mask'][:, 1:-1].to(device))
        return outputs.last_hidden_state
