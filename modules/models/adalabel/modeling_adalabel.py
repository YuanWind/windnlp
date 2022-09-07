from transformers.models.bart.modeling_bart import *
from transformers.models.bart.modeling_bart import _expand_mask
import torch.nn as nn
from modules.models.adalabel.configuration_adalabel import AdaLabelConfig
from transformers.utils.generic import ModelOutput

from scripts.utils import get_tensor_device

class TransformerBiDecoder(BartDecoder):
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None) -> None:
        super().__init__(config, embed_tokens)
        self.layers = nn.ModuleList([BartDecoderLayer(config) for _ in range(1)])
        self.post_init()

class RRGBartDecoderLayer(BartDecoderLayer):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.tri_encoder_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.cat_linear = nn.Linear(self.embed_dim*2,self.embed_dim)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        **kwargs
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states        
        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states_inpt = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states_inpt

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states_inpt,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            # 加入triples
            triples_hidden_states,triples_attention_mask = kwargs.pop('triples_hidden_states',None),kwargs.pop('triples_attention_mask',None)
            if triples_hidden_states is not None:
                #  self.encoder_attn 此处是否需要共享
                tri_hidden_states, tri_cross_attn_weights, tri_cross_attn_present_key_value = self.encoder_attn(
                    hidden_states=hidden_states_inpt,
                    key_value_states=triples_hidden_states,
                    attention_mask=triples_attention_mask,
                    layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=cross_attn_past_key_value,
                    output_attentions=output_attentions,
                )
                
                hidden_states = torch.cat([hidden_states, tri_hidden_states], dim=-1)
                hidden_states = self.cat_linear(hidden_states)
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs



class RRGBartDecoder(BartDecoder):
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config, embed_tokens)
        self.layers = nn.ModuleList([RRGBartDecoderLayer(config) for _ in range(config.decoder_layers)])
        
    def forward(self, input_ids: torch.LongTensor = None, attention_mask: Optional[torch.Tensor] = None, encoder_hidden_states: Optional[torch.FloatTensor] = None, encoder_attention_mask: Optional[torch.LongTensor] = None, head_mask: Optional[torch.Tensor] = None, cross_attn_head_mask: Optional[torch.Tensor] = None, past_key_values: Optional[List[torch.FloatTensor]] = None, inputs_embeds: Optional[torch.FloatTensor] = None, use_cache: Optional[bool] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None, **kwargs) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
        # -------------------取出triples--------------------    
        triples_hidden_states,triples_attention_mask = kwargs.pop('triples_hidden_states',None),kwargs.pop('triples_attention_mask',None)
        if triples_hidden_states is not None and triples_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            triples_attention_mask = _expand_mask(triples_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
        
        
        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        "The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            assert not self.gradient_checkpointing, '需要实现gradient_checkpointing里边的逻辑'
            if self.gradient_checkpointing and self.training:
                pass
                # if use_cache:
                #     logger.warning(
                #         "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                #     )
                #     use_cache = False

                # def create_custom_forward(module):
                #     def custom_forward(*inputs):
                #         # None for past_key_value
                #         return module(*inputs, output_attentions, use_cache)

                #     return custom_forward

                # layer_outputs = torch.utils.checkpoint.checkpoint(
                #     create_custom_forward(decoder_layer),
                #     hidden_states,
                #     attention_mask,
                #     encoder_hidden_states,
                #     encoder_attention_mask,
                #     head_mask[idx] if head_mask is not None else None,
                #     cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                #     None,
                    
                # )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    triples_hidden_states=triples_hidden_states,
                    triples_attention_mask=triples_attention_mask,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )

class RRGBartModel(BartModel):
    def __init__(self, config: BartConfig):
        """主要修改forward，加入property编码信息

        Args:
            config (BartConfig): _description_
        """
        super().__init__(config)
        if self.config.mycfg.add_triples:
            self.decoder = RRGBartDecoder(config, self.shared)
        
        if not config.mycfg.share_bart_params:
            self.pi_encoder = BartEncoder(config, self.shared)
        else:
            self.pi_encoder = self.encoder
        # self.generate()
        
    def forward(self, input_ids: torch.LongTensor = None, attention_mask: Optional[torch.Tensor] = None, decoder_input_ids: Optional[torch.LongTensor] = None, decoder_attention_mask: Optional[torch.LongTensor] = None, head_mask: Optional[torch.Tensor] = None, decoder_head_mask: Optional[torch.Tensor] = None, cross_attn_head_mask: Optional[torch.Tensor] = None, encoder_outputs: Optional[List[torch.FloatTensor]] = None, past_key_values: Optional[List[torch.FloatTensor]] = None, inputs_embeds: Optional[torch.FloatTensor] = None, decoder_inputs_embeds: Optional[torch.FloatTensor] = None, use_cache: Optional[bool] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None, pi_input_ids = None, pi_attention_mask = None, **kwargs) -> Union[Tuple, Seq2SeqModelOutput]:
        
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # 编码产品属性信息，共享encoder
        pi_encoder_outputs = None
        if pi_input_ids is not None:
            pi_encoder_outputs = self.pi_encoder(
                input_ids=pi_input_ids,
                attention_mask=pi_attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        if pi_encoder_outputs is not None:
            hidden_states = torch.cat([encoder_outputs[0], pi_encoder_outputs[0]],dim=1)
            enc_attention_mask = torch.cat([attention_mask,pi_attention_mask],dim=1)
        else:
            hidden_states = encoder_outputs[0]
            enc_attention_mask = attention_mask
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        if not self.config.mycfg.add_triples:
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=enc_attention_mask,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=enc_attention_mask,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                tri_hidden_states = kwargs.pop('tri_hidden_states'),
                tri_attention_mask = kwargs.pop('tri_attention_mask')
            )
        if not return_dict:
            return decoder_outputs + encoder_outputs

        
        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            # past_key_values=decoder_outputs.past_key_values,
            # decoder_hidden_states=decoder_outputs.hidden_states,
            # decoder_attentions=decoder_outputs.attentions,
            # cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=hidden_states,
            # encoder_hidden_states=encoder_outputs.hidden_states,
            # encoder_attentions=encoder_outputs.attentions,
        )

class RRGBartForConditionalGeneration(BartForConditionalGeneration):
    def __init__(self, config: BartConfig) -> None:
        """把model改成RRGBart

        Args:
            config (BartConfig): _description_
        """
        super().__init__(config)
        self.model = RRGBartModel(config)
        if config.mycfg.add_triples:
            if not config.mycfg.share_bart_params:
                self.tri_encoder = BartEncoder(config, self.model.shared)
            else:
                self.tri_encoder = self.model.encoder

        if config.mycfg.add_groups:
            
            if not config.mycfg.share_bart_params:
                self.tgt_encoder = BartEncoder(config, self.model.shared)
            else:
                self.tgt_encoder = self.model.encoder
                

            # self.state_linear = nn.Linear(config.d_model*2, config.d_model)
            # self.gru = nn.GRU(input_size=config.d_model, hidden_size=config.d_model,batch_first = True, num_layers=1)
            self.post_linear1 = nn.Linear(config.d_model*2, config.d_model*2)
            self.post_linear2 = nn.Linear(config.d_model*2, config.d_model*2)
            self.prior_linear1 = nn.Linear(config.d_model*1, config.d_model*2)
            self.prior_linear2 = nn.Linear(config.d_model*2, config.d_model*2)
            self.group_linear = nn.Linear(config.d_model, config.d_model)
            self.begin_ids = config.mycfg.begin_ids
            self.end_ids = config.mycfg.end_ids
            self.group_head = nn.Linear(config.d_model,self.model.shared.num_embeddings, bias=False)
            # self.gru_group_states, self.group_mask = None,None
            self.global_z = None
            # self.q_state, self.k_state = nn.Linear(config.d_model, config.d_model),nn.Linear(config.d_model, config.d_model)
            self.group_attn = BartAttention(self.config.d_model,self.config.decoder_attention_heads,dropout=config.attention_dropout,is_decoder=True,)
        self.post_init()
        
    def forward(self, input_ids: torch.LongTensor = None, attention_mask: Optional[torch.Tensor] = None, decoder_input_ids: Optional[torch.LongTensor] = None, decoder_attention_mask: Optional[torch.LongTensor] = None, head_mask: Optional[torch.Tensor] = None, decoder_head_mask: Optional[torch.Tensor] = None, cross_attn_head_mask: Optional[torch.Tensor] = None, encoder_outputs: Optional[List[torch.FloatTensor]] = None, past_key_values: Optional[List[torch.FloatTensor]] = None, inputs_embeds: Optional[torch.FloatTensor] = None, decoder_inputs_embeds: Optional[torch.FloatTensor] = None, labels: Optional[torch.LongTensor] = None, use_cache: Optional[bool] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None, pi_input_ids = None, pi_attention_mask = None, **kwargs) -> Union[Tuple, Seq2SeqLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        tgt_encoder_outputs = None
        if encoder_outputs is None:
            src_encoder_outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            if self.config.mycfg.add_groups:
                tgt_encoder_outputs = self.tgt_encoder(
                    input_ids=decoder_input_ids,
                    attention_mask=decoder_attention_mask,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            
        else:
            src_encoder_outputs = encoder_outputs
            
        # 编码产品属性信息，共享encoder
        pi_encoder_outputs = None
        if pi_input_ids is not None:
            pi_encoder_outputs = self.model.pi_encoder(
                input_ids=pi_input_ids,
                attention_mask=pi_attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # 将评论和产品表示拼接到一起
        if pi_encoder_outputs is not None:
            all_hidden_states = torch.cat([src_encoder_outputs[0], pi_encoder_outputs[0]],dim=1)
            all_attention_mask = torch.cat([attention_mask,pi_attention_mask],dim=1)
        else:
            all_hidden_states = encoder_outputs[0]
            all_attention_mask = attention_mask
            
        tri_encoder_outputs = None
        if self.config.mycfg.add_triples:
            tri_input_ids = kwargs.pop('tri_input_ids',None)
            tri_attention_mask =  kwargs.pop('tri_attention_mask',None)
            if tri_input_ids is not None:
                tri_encoder_outputs = self.tri_encoder(
                    input_ids=tri_input_ids,
                    attention_mask=tri_attention_mask,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
        # 将评论和产品表示拼接到一起
        tri_start = None
        if tri_encoder_outputs is not None:
            tri_start = all_hidden_states.shape[1]
            all_hidden_states = torch.cat([all_hidden_states, tri_encoder_outputs[0]],dim=1)
            all_attention_mask = torch.cat([all_attention_mask,tri_attention_mask],dim=1)
            tri_end = all_hidden_states.shape[1]
        # 正常decode
        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=all_hidden_states,
            encoder_attention_mask=all_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # [bs, tgt_seq_len, vocab_size]
        lm_logits = self.lm_head(decoder_outputs[0]) + self.final_logits_bias
        
        # 记录评论的长度，以便寻找产品表示的开始位置  
        kl_loss = torch.tensor(0) # 初始化 kl loss为None
        if self.config.mycfg.add_groups: # 利用group进行decode
            bat_triple_idxs = kwargs.pop('bat_triple_idxs', None)
            if tri_start is None:
                tri_start, tri_end = kwargs.pop('start_end')[1]
            if self.global_z is None:
                # 做采样，期望学习出回复中应该包含哪些方面
                kl_loss, global_z = self.do_sample( all_hidden_states, 
                                                    all_attention_mask, 
                                                    tgt_encoder_outputs[0] if tgt_encoder_outputs is not None else None,
                                                    decoder_attention_mask if decoder_attention_mask is not None else None
                                                   )
                self.global_z = global_z
            group_decoder_outputs = self.do_group(bat_triple_idxs, all_hidden_states[:,tri_start:tri_end],all_attention_mask,self.global_z,decoder_outputs[0])
            # 自回归生成阶段的每个batch内的每一步生成时group_states应该相同，因此使用类变量保存结果，每个batch生成完之后再重新置为None
            # all_hidden_states = torch.cat([all_hidden_states, self.gru_group_states],dim=1)
            # all_attention_mask = torch.cat([all_attention_mask,self.group_mask],dim=1)
            # group_decoder_outputs = self.model.decoder(
            #     input_ids=decoder_input_ids,
            #     attention_mask=decoder_attention_mask,
            #     encoder_hidden_states=self.gru_group_states,
            #     encoder_attention_mask=self.group_mask,
            #     head_mask=decoder_head_mask,
            #     cross_attn_head_mask=cross_attn_head_mask,
            #     past_key_values=past_key_values,
            #     inputs_embeds=decoder_inputs_embeds,
            #     use_cache=use_cache,
            #     output_attentions=output_attentions,
            #     output_hidden_states=output_hidden_states,
            #     return_dict=return_dict,
            # )
            lm_logits = lm_logits + self.group_head(group_decoder_outputs)
            # lm_logits = self.group_head(group_decoder_outputs)

        dict_outs = {
            'logits':lm_logits,
            'generate2': None,
            'kl_loss': kl_loss
        }
        
        return ModelOutput(**dict_outs)
    
    def do_sample(self, encoder_hidden, encoder_mask, tgt_hidden = None, tgt_mask = None):
        # 对非pad的字符求均值
        avg_enc = []
        for bs_id in range(encoder_hidden.shape[0]):
            avg_enc.append(torch.mean(encoder_hidden[bs_id, encoder_mask.ne(self.config.pad_token_id)[bs_id]],dim=0))
        avg_enc = torch.stack(avg_enc,dim=0)
        
        prior_inp = avg_enc
        prior_mu , prior_logvar = torch.split(self.prior_linear2(torch.tanh(self.prior_linear1(prior_inp))),self.config.d_model,1)
        
        if self.training:# 训练阶段使用后验概率
            avg_tgt = []
            for bs_id in range(tgt_hidden.shape[0]):
                avg_tgt.append(torch.mean(tgt_hidden[bs_id, tgt_mask.ne(self.config.pad_token_id)[bs_id]],dim=0))
            avg_tgt = torch.stack(avg_tgt,dim=0)
            post_inp = torch.cat((avg_enc, avg_tgt), 1)
            post_mu , post_logvar = torch.split(self.post_linear2(torch.tanh(self.post_linear1(post_inp))),self.config.d_model,1)

            global_z = sample_gaussian(post_mu.shape, post_mu , post_logvar)
            kl_div = KL_divergence(prior_mu, prior_logvar, post_mu, post_logvar)
            kl_div = kl_div/prior_mu.shape[0] # 平均kl散度
        else:# 推理阶段使用先验概率
            global_z = sample_gaussian(prior_mu.shape, prior_mu , prior_logvar)
            kl_div = torch.tensor(0)
        
        return kl_div, global_z
    
    def do_group(self, bat_triple_idxs, hidden_states,encoder_mask, global_z=None, decoder_state=None):
        # global_z用于初始化gru，然后group循环过gru
        triples_states = [] 
        
        device = get_tensor_device(hidden_states)
        real_group_num = []
        for bs_idx, groups in enumerate(bat_triple_idxs):
            tmp = self.model.shared(torch.tensor([self.begin_ids],device=device)) # begin
            num = 1
            for sep_idxs in groups:
                group_res = None
                if sep_idxs!=[0]:
                    # i[0]+1 是为了把triple前面的sep去掉
                    t_idxs = [torch.arange(i[0]+1,i[1],device=device) for i in sep_idxs]
                    t_idxs = torch.cat(t_idxs)
                    group_res = torch.index_select(hidden_states[bs_idx], 0, t_idxs)   # [d_model]
                    num += group_res.shape[0]
                # else:
                #     group_res = self.model.shared([torch.tensor(self.config.pad_token_id,device=device)])
                if group_res is not None:
                    tmp = torch.cat([tmp, group_res],dim=0)
            tmp = torch.cat([tmp, 
                             self.model.shared(torch.tensor([self.end_ids],device=device))
                             ],dim=0)
            
            triples_states.append(tmp)
            real_group_num.append(num+1) # 此处+1是加了end的
        real_group_num = torch.tensor(real_group_num)
        max_num = real_group_num.max()
        for idx, t in enumerate(triples_states):
            if t.shape[0] < max_num:
                pad_num = max_num-t.shape[0]
                pad_tensor = self.model.shared(torch.tensor([self.config.pad_token_id],device=device))
                triples_states[idx] = torch.cat([t,pad_tensor.repeat((pad_num,1))], dim=0)
                
        triples_states = torch.stack(triples_states, dim=0)
        # if global_z is None:
        #     init_state = None
        # else:
        #     avg_states = []
        #     for bs_id in range(hidden_states.shape[0]):
        #         avg_states.append(torch.mean(hidden_states[bs_id, encoder_mask.ne(self.config.pad_token_id)[bs_id]],dim=0))
        #     avg_states = torch.stack(avg_states,dim=0)
        #     init_state = torch.cat([avg_states, global_z], dim=-1)
        #     init_state = torch.tanh(self.state_linear(init_state).unsqueeze(0))  
        
        # group_states = torch.tanh(self.group_linear(triples_states))
        # inps_packed = nn.utils.rnn.pack_padded_sequence(group_states, real_group_num, batch_first=True, enforce_sorted=False)
        # packed_group_states, states = self.gru(inps_packed, init_state)
        # group_states, _ = nn.utils.rnn.pad_packed_sequence(packed_group_states,batch_first=True)
        
        group_mask = torch.zeros((triples_states.shape[0],triples_states.shape[1]),device=device)
        for idx in range(real_group_num.shape[0]):
            l = real_group_num[idx]
            group_mask[idx,:l] = 1
        
        group_decoder = decoder_state + global_z[:,None,:]
        # group_attn_weights = torch.bmm(self.q_state(group_decoder),self.k_state(triples_states).transpose(1, 2))
        group_mask_a = _expand_mask(group_mask, global_z.dtype, tgt_len=group_decoder.shape[1])
        # group_attn_weights = group_attn_weights + group_mask_a
        # group_attn_weights = torch.softmax(group_attn_weights, dim=-1)
        # group_states = group_attn_weights.transpose(1, 2) * triples_states
        
        
        # for idx in range(real_group_num.shape[0]):
        #     l = real_group_num[idx]
        #     pad_tensor = self.model.shared(torch.tensor([self.config.pad_token_id],device=device))
        #     group_states[idx,l:] = pad_tensor
        group_decoder = self.group_attn(group_decoder,triples_states,attention_mask=group_mask_a)[0]
            
        return group_decoder
    
    def get_pi_encoder(self):
        return self.model.pi_encoder
    def get_tri_encoder(self):
        return self.tri_encoder
    
    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ):
        # 1. get encoder
        encoder = self.get_encoder()
        pi_encoder = self.get_pi_encoder()
        if self.config.mycfg.add_triples:
            tri_encoder = self.get_tri_encoder()
        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        
        pi_input_ids = encoder_kwargs.pop('pi_input_ids', None)
        pi_attention_mask = encoder_kwargs.pop('pi_attention_mask', None)
        tri_input_ids = encoder_kwargs.pop('tri_input_ids', None)
        tri_attention_mask = encoder_kwargs.pop('tri_attention_mask', None)
        bat_triple_idxs = encoder_kwargs.pop('bat_triple_idxs', None)
        encoder_outputs = encoder(**encoder_kwargs)
        pi_start,pi_end,tri_start,tri_end = None, None,None,None
        if pi_input_ids is not None:
            pi_encoder_outputs = pi_encoder(
                input_ids=pi_input_ids,
                attention_mask=pi_attention_mask,
                return_dict=encoder_kwargs["return_dict"],
            )   
        if tri_input_ids is not None:
            
            tri_encoder_outputs = tri_encoder(
                input_ids=tri_input_ids,
                attention_mask=tri_attention_mask,
                return_dict=encoder_kwargs["return_dict"],
            )  
            
        if pi_input_ids is not None:
            pi_start = inputs_tensor.shape[1]
            hidden_states = torch.cat([encoder_outputs[0], pi_encoder_outputs[0]],dim=1)
            enc_attention_mask = torch.cat([encoder_kwargs['attention_mask'],pi_attention_mask],dim=1)
            src_ids = torch.cat([inputs_tensor,pi_input_ids],dim=1)
            pi_start = src_ids.shape[1]
        else:
            hidden_states = encoder_outputs[0]
            enc_attention_mask = encoder_kwargs['attention_mask']
            src_ids = inputs_tensor
        if tri_input_ids is not None:
            tri_start = src_ids.shape[1]
            hidden_states = torch.cat([hidden_states, tri_encoder_outputs[0]],dim=1)
            enc_attention_mask = torch.cat([enc_attention_mask,tri_attention_mask],dim=1)
            src_ids = torch.cat([src_ids,tri_input_ids],dim=1)
            tri_end = src_ids.shape[1]
            
        encoder_outputs['last_hidden_state'] = hidden_states
        model_kwargs["encoder_outputs"]: ModelOutput = encoder_outputs
        model_kwargs['enc_input_ids'] = src_ids
        model_kwargs['enc_attention_mask'] = enc_attention_mask
        model_kwargs['start_end'] = [(pi_start,pi_end),(tri_start,tri_end)]
        if self.config.mycfg.add_triples:
            model_kwargs['bat_triple_idxs'] = bat_triple_idxs
        return model_kwargs
    
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        input_arguments = {
            "input_ids": kwargs.pop('enc_input_ids',None),  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": kwargs.pop('enc_attention_mask',None),
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            "start_end":kwargs.pop('start_end',None),
            'bat_triple_idxs':kwargs.pop('bat_triple_idxs',None)
        }

        return input_arguments
     
def sample_gaussian(shape, mu, logvar):
    x = torch.randn(shape,device=get_tensor_device(mu))
    return mu + torch.exp(logvar/2) * x

def KL_divergence(prior_mu, prior_logvar, post_mu, post_logvar, loss_mask=None):
    divergence = 0.5 * torch.sum(torch.exp(post_logvar - prior_logvar)
                                        + torch.pow(post_mu - prior_mu, 2) / torch.exp(prior_logvar)
                                        - 1 - (post_logvar - prior_logvar), dim=1)
    if loss_mask is not None:
        return torch.sum(loss_mask * divergence)
    else:
        return torch.sum(divergence)