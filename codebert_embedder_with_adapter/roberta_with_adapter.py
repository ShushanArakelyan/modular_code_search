import torch
import torch.nn as nn

from transformers.models.roberta.modeling_roberta import RobertaEncoder, RobertaLayer, RobertaModel
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.modeling_utils import PreTrainedModel, apply_chunking_to_forward
from transformers.activations import ACT2FN

from itertools import chain



class RobertaWithAdaptersConfig(RobertaConfig):
    def __init__(self, adapter_dim=64, adapt_layer_norm=False, unfreeze_hyper_encoder=False, pad_token_id=1, bos_token_id=0, eos_token_id=2, **kwargs):
        """Constructs RobertaConfig."""
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs) 
        
        # Adapter
        self.adapter_dim = adapter_dim
        self.adapt_layer_norm = adapt_layer_norm

        
class EncoderLayerWithAdapter(RobertaLayer):
    def __init__(self, config, device):
        super(EncoderLayerWithAdapter, self).__init__(config)
        self.device = device
        self.adapter_down_layer = torch.nn.Linear(in_features=config.hidden_size, 
                                                  out_features=config.adapter_dim).to(self.device)
        torch.nn.init.xavier_uniform_(self.adapter_down_layer.weight, gain=0.0000001)
        torch.nn.init.constant_(self.adapter_down_layer.bias, 0.0)
        self.adapter_up_layer = torch.nn.Linear(in_features=config.adapter_dim, 
                                                out_features=config.hidden_size).to(self.device)
        torch.nn.init.xavier_uniform_(self.adapter_up_layer.weight, gain=0.0000001)
        torch.nn.init.constant_(self.adapter_up_layer.bias, 0.0)
        self.activation_fn = ACT2FN[config.hidden_act]
        self.use_adapter = True

    def adapter_down(self, x):
        return self.adapter_down_layer(x)

    def adapter_up(self, x):
        return self.adapter_up_layer(x)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
        ):
            
            # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
            self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
            self_attention_outputs = self.attention(
                hidden_states,
                attention_mask,
                head_mask,
                output_attentions=output_attentions,
                past_key_value=self_attn_past_key_value,
            )
            attention_output = self_attention_outputs[0]
            if self.use_adapter:
                residual_adapter = attention_output
                attention_output = self.adapter_down(attention_output)
                attention_output = self.activation_fn(attention_output)
                attention_output = self.adapter_up(attention_output)
                attention_output = residual_adapter + attention_output

            # if decoder, the last output is tuple of self-attn cache
            if self.is_decoder:
                outputs = self_attention_outputs[1:-1]
                present_key_value = self_attention_outputs[-1]
            else:
                outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
            
            cross_attn_present_key_value = None
            if self.is_decoder and encoder_hidden_states is not None:
                assert hasattr(
                    self, "crossattention"
                ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"

                # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
                cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
                cross_attention_outputs = self.crossattention(
                    attention_output,
                    attention_mask,
                    head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    cross_attn_past_key_value,
                    output_attentions,
                )
                attention_output = cross_attention_outputs[0]
                if self.use_adapter:
                    residual_adapter = attention_output
                    attention_output = self.adapter_down(attention_output)
                    attention_output = self.activation_fn(attention_output)
                    attention_output = self.adapter_up(attention_output)
                    attention_output = residual_adapter + attention_output
                outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

                # add cross-attn cache to positions 3,4 of present_key_value tuple
                cross_attn_present_key_value = cross_attention_outputs[-1]
                present_key_value = present_key_value + cross_attn_present_key_value

            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
            )

            outputs = (layer_output,) + outputs

            # if decoder, return the attn key/values as the last output
            if self.is_decoder:
                outputs = outputs + (present_key_value,)

            return outputs
        
    def skip_adapter(self, use_adapter):
        self.use_adapter = use_adapter
        

class RobertaEncoderWithAdapter(RobertaEncoder):
    def __init__(self, config, device):
        super(RobertaEncoderWithAdapter, self).__init__(config)
        self.layer = nn.ModuleList([EncoderLayerWithAdapter(config, device).to(device) for _ in range(config.num_hidden_layers)])
    
    def use_adapter(self, use_adapter):
        for layer in self.layer:
            layer.skip_adapter(use_adapter)
        

class RobertaModelWithAdapter(RobertaModel):
    def __init__(self, config, device, add_pooling_layer=True):
        super(RobertaModelWithAdapter, self).__init__(config)
        self.encoder = RobertaEncoderWithAdapter(config, device).to(device)
        self.init_weights()
    
    def use_adapter(self, use_adapter):
        self.encoder.use_adapter(use_adapter)

        
class SimpleGenerator(nn.Module):
    # takes in a encoded task description and generates parameters of an adapter
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.input_dim = config.hidden_size # config.d_model
        self.hidden_dim = int (config.hidden_size / 4)
        self.output_dim = config.hidden_size * config.adapter_dim * 2 + config.hidden_size + config.adapter_dim
        if config.adapt_layer_norm:
            self.output_dim += 2 * config.hidden_size
        self.linear1 = torch.nn.Linear(self.input_dim, self.hidden_dim).to(device)
        self.activation_fn = ACT2FN[config.hidden_act]
        self.linear2 = torch.nn.Linear(self.hidden_dim, self.output_dim).to(device)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x).to(self.device)
        x = self.linear2(x)
        return x.view(-1)
    

class ParameterGenerator(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        self.config = config
        self.adapter_generators = nn.ModuleList([
            SimpleGenerator(config, device).to(device) for _ in range(config.num_hidden_layers)
        ])
        self.device = device

    def forward(self, x):
        params = [adapter_generator(x) for adapter_generator in self.adapter_generators]

        return params

#     def parameters(self):
#         return chain([ag.parameters() for ag in self.adapter_generators])
    

class RobertaWithHyperAdapters(nn.Module):
    def __init__(self, model, meta_model, config, device):
        super().__init__()

        self.config = config
        self.model = model
        self.device = device
        self.meta_model = meta_model

    def set_verb_embedding(self, verb_embedding):
        # generate adapter parameters using task descriptions
        generated_params = self.meta_model(verb_embedding)

        # apply the parameters to the adapters
        self.apply_params_to_adapters(generated_params)

    def use_adapter(self, use_adapter):
        self.model.use_adapter(use_adapter)
        
    def forward(self, verb_embedding, 
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None):
        # generate adapter parameters using task descriptions
        generated_params = self.meta_model(verb_embedding)

        # apply the parameters to the adapters
        self.apply_params_to_adapters(generated_params)
        
        # use the adapted model to make zero-shot inference
        ret = self.model(input_ids, attention_mask, token_type_ids,
                         position_ids, head_mask, inputs_embeds,
                         encoder_hidden_states, encoder_attention_mask,
                         past_key_values, use_cache, output_attentions, 
                         output_hidden_states, return_dict)

        return ret       

    def apply_params_to_adapters(self, generated_params):
        params = generated_params
        
        d_model = self.config.hidden_size
        d_adapter = self.config.adapter_dim

        for p, layer in zip(params, self.model.encoder.layer):
            del layer.adapter_down_layer.weight, layer.adapter_down_layer.bias
            del layer.adapter_up_layer.weight, layer.adapter_up_layer.bias
            layer.adapter_down_layer.weight = p.index_select(dim=0, index=torch.LongTensor(range(0, d_model*d_adapter)).to(self.device)).view(d_adapter, d_model).to(self.device)
            layer.adapter_up_layer.weight = p.index_select(dim=0, index=torch.LongTensor(range(d_model*d_adapter, d_model*d_adapter*2)).to(self.device)).view(d_model, d_adapter).to(self.device)
            layer.adapter_down_layer.bias = p.index_select(dim=0, index=torch.LongTensor(range(d_model*d_adapter*2, d_model*d_adapter*2+d_adapter)).to(self.device)).view(d_adapter).to(self.device)
            layer.adapter_up_layer.bias = p.index_select(dim=0, index=torch.LongTensor(range(d_model*d_adapter*2+d_adapter, d_model*d_adapter*2+d_adapter+d_model)).to(self.device)).view(d_model).to(self.device)

            if self.config.adapt_layer_norm:
                layer.self_attn_layer_norm.weight.data = layer.self_attn_layer_norm.weight.data + p[-2*d_model: -1*d_model]
                layer.self_attn_layer_norm.bias.data = layer.self_attn_layer_norm.bias.data + p[-1*d_model:]