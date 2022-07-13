import torch 
import torch.nn as nn
from modules.models.nezha.modeling_nezha import NeZhaEmbeddings, NeZhaForMaskedLM, NeZhaModel

def init_entity_embedding():
    # TODO 
    pass
class Embeddings(NeZhaEmbeddings):
    def __init__(self, config, add_type_emb=False, type_num=0, type_w=0, padding_idx=0):
        super().__init__(config)
        self.add_type_emb = add_type_emb
        self.type_w = type_w
        if self.add_type_emb:
            self.entity_embedding = nn.Embedding(type_num, config.hidden_size,padding_idx=padding_idx)

    def forward(self, input_ids=None, token_type_ids=None, inputs_embeds=None,type_ids=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
            
        if self.add_type_emb and type_ids is not None: # type_ids: [bat, seq_len, 15]  
            entity_embed = self.entity_embedding(type_ids)
            entity_embed = torch.sum(entity_embed,dim=-2) # 或许可以进行加权，强调出现次数少的实体的权重
            entity_embed = entity_embed*self.type_w
            
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        if self.add_type_emb and type_ids is not None:
            embeddings = embeddings + entity_embed
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class Gaiic_NeZhaModel(NeZhaModel):

    def __init__(self, config, add_type_emb=False, type_num=0, type_w = 0):
        super().__init__(config)
        self.config = config
        self.embeddings = Embeddings(config,add_type_emb,type_num, type_w = type_w)
        self.add_type_emb = add_type_emb
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            type_ids = None,
            head_mask=None,
            position_ids=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            return_dict=None,
            output_attentions=False,
            output_hidden_states=False,
    ):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, self.device
        )
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds,type_ids=type_ids
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  
        
        return outputs  


class Gaiic_PretrainModel(NeZhaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.bert = Gaiic_NeZhaModel(config)
    