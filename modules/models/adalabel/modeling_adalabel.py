from transformers.models.bart.modeling_bart import *
import torch.nn as nn
import torch.nn.functional as F
from modules.models.adalabel.configuration_adalabel import AdaLabelConfig
from transformers.utils.generic import ModelOutput

from scripts.utils import get_tensor_device

class TransformerBiDecoder(BartDecoder):
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None) -> None:
        super().__init__(config, embed_tokens)
        self.layers = nn.ModuleList([BartDecoderLayer(config) for _ in range(1)])
        self.post_init()
        
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`
    Copied from onmt
    
    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, dropout, dim, max_len=5000):
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        """Embed inputs.

        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
            step (int or NoneType): If stepwise (``seq_len = 1``), use
                the encoding for this position.
        """

        # emb = emb * math.sqrt(self.dim)
        if step is None:
            emb = emb + self.pe[:emb.size(0)]
        else:
            emb = emb + self.pe[step]
        # emb = self.dropout(emb)
        return emb

class RRGBartModel(BartModel):
    def __init__(self, config: BartConfig):
        """主要修改forward，加入property编码信息

        Args:
            config (BartConfig): _description_
        """
        super().__init__(config)
        if not config.share_encoder_for_pi:
            self.pi_encoder = BartEncoder(config, self.shared)
        else:
            self.pi_encoder = self.encoder
        # self.generate()
        
    def forward(self, input_ids: torch.LongTensor = None, attention_mask: Optional[torch.Tensor] = None, decoder_input_ids: Optional[torch.LongTensor] = None, decoder_attention_mask: Optional[torch.LongTensor] = None, head_mask: Optional[torch.Tensor] = None, decoder_head_mask: Optional[torch.Tensor] = None, cross_attn_head_mask: Optional[torch.Tensor] = None, encoder_outputs: Optional[List[torch.FloatTensor]] = None, past_key_values: Optional[List[torch.FloatTensor]] = None, inputs_embeds: Optional[torch.FloatTensor] = None, decoder_inputs_embeds: Optional[torch.FloatTensor] = None, use_cache: Optional[bool] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None, pi_input_ids = None, pi_attention_mask = None) -> Union[Tuple, Seq2SeqModelOutput]:
        
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
        self.cp_head = nn.Linear(config.d_model,config.d_model)
        self.post_init()
        
    def forward(self, input_ids: torch.LongTensor = None, attention_mask: Optional[torch.Tensor] = None, decoder_input_ids: Optional[torch.LongTensor] = None, decoder_attention_mask: Optional[torch.LongTensor] = None, head_mask: Optional[torch.Tensor] = None, decoder_head_mask: Optional[torch.Tensor] = None, cross_attn_head_mask: Optional[torch.Tensor] = None, encoder_outputs: Optional[List[torch.FloatTensor]] = None, past_key_values: Optional[List[torch.FloatTensor]] = None, inputs_embeds: Optional[torch.FloatTensor] = None, decoder_inputs_embeds: Optional[torch.FloatTensor] = None, labels: Optional[torch.LongTensor] = None, use_cache: Optional[bool] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None, pi_input_ids = None, pi_attention_mask = None) -> Union[Tuple, Seq2SeqLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pi_input_ids = pi_input_ids, pi_attention_mask = pi_attention_mask
        )
        # [bs, tgt_seq_len, vocab_size]
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        if self.config.use_copy_mechanism:
        # 计算copy 机制logits
            if pi_input_ids is not None:
                # src_attention_mask = torch.cat([attention_mask,pi_attention_mask],dim=1)
                src_ids = torch.cat([input_ids,pi_input_ids],dim=1)
            else:
                # src_attention_mask = attention_mask
                src_ids = input_ids
                            # [bs,src_seq_len,d_model] * [bs,d_model, tgt_seq_len] --> [bs,tgt_seq_len,src_seq_len]
            cp_outs = torch.bmm(torch.tanh(self.cp_head(outputs[1])),(outputs[0].transpose(1,2))).transpose(1,2)
            # outputs:[bs, src_seq_len, hidden_dim] --> [bs, tgt_seq_len, vocab_size]
            expanded_src_ids = src_ids[:, None, :].expand(src_ids.shape[0], lm_logits.shape[1], src_ids.shape[1])
            cp_src_logits = torch.zeros_like(lm_logits,dtype=cp_outs.dtype).scatter_(2, expanded_src_ids , cp_outs)
            
            lm_logits = lm_logits+cp_src_logits
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    
    def get_pi_encoder(self):
        return self.model.pi_encoder
    
    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ):
        # 1. get encoder
        encoder = self.get_encoder()
        pi_encoder = self.get_pi_encoder()
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
        bat_triple_idxs = encoder_kwargs.pop('bat_triple_idxs', None)
        pi_input_ids = encoder_kwargs.pop('pi_input_ids', None)
        pi_attention_mask = encoder_kwargs.pop('pi_attention_mask', None)
        encoder_outputs = encoder(**encoder_kwargs)
        
        if pi_input_ids is not None:
            pi_encoder_outputs = pi_encoder(
                input_ids=pi_input_ids,
                attention_mask=pi_attention_mask,
                return_dict=encoder_kwargs["return_dict"],
            )        
        if pi_input_ids is not None:
            hidden_states = torch.cat([encoder_outputs[0], pi_encoder_outputs[0]],dim=1)
            enc_attention_mask = torch.cat([encoder_kwargs['attention_mask'],pi_attention_mask],dim=1)
            src_ids = torch.cat([inputs_tensor,pi_input_ids],dim=1)
        else:
            hidden_states = encoder_outputs[0]
            enc_attention_mask = encoder_kwargs['attention_mask']
            src_ids = inputs_tensor
            
        encoder_outputs['last_hidden_state'] = hidden_states
        model_kwargs["encoder_outputs"]: ModelOutput = encoder_outputs
        model_kwargs['enc_input_ids'] = src_ids
        model_kwargs['enc_attention_mask'] = enc_attention_mask
        model_kwargs['src_len'] = inputs_tensor.shape[1]
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
            "src_len":kwargs.pop('src_len',None),
            'bat_triple_idxs':kwargs.pop('bat_triple_idxs',None)
        }

        return input_arguments
    
class AdaLabelModel(RRGBartModel):
    def __init__(self, config: BartConfig):
        
        super().__init__(config)
        self.shared = None
        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)
        self.bidecoder = TransformerBiDecoder(config)
        # TODO 是否将encoder、decoder、bidecoder的embed_positions改为正余弦函数式的
        # self.encoder.embed_positions = PositionalEncoding(config.dropout, config.d_model, config.max_position_embeddings)
        self.post_init()
    def forward(self, 
                input_ids: torch.LongTensor = None, 
                attention_mask: Optional[torch.Tensor] = None, 
                decoder_input_ids: Optional[torch.LongTensor] = None, 
                decoder_attention_mask: Optional[torch.LongTensor] = None, 
                head_mask: Optional[torch.Tensor] = None, 
                decoder_head_mask: Optional[torch.Tensor] = None, 
                cross_attn_head_mask: Optional[torch.Tensor] = None, 
                encoder_outputs: Optional[List[torch.FloatTensor]] = None, 
                past_key_values: Optional[List[torch.FloatTensor]] = None, 
                inputs_embeds: Optional[torch.FloatTensor] = None, 
                decoder_inputs_embeds: Optional[torch.FloatTensor] = None, 
                use_cache: Optional[bool] = None, 
                output_attentions: Optional[bool] = None, 
                output_hidden_states: Optional[bool] = None, 
                return_dict: Optional[bool] = None, pi_input_ids = None, pi_attention_mask = None) -> Union[Tuple, Seq2SeqModelOutput]:
        seq2SeqModelOutput = super().forward(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, head_mask, decoder_head_mask, cross_attn_head_mask, encoder_outputs, past_key_values, inputs_embeds, decoder_inputs_embeds, use_cache, output_attentions, output_hidden_states, return_dict)
        memory_bank = seq2SeqModelOutput.encoder_last_hidden_state
        bidecoder_outputs = self.bidecoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=memory_bank,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pi_input_ids = pi_input_ids, pi_attention_mask = pi_attention_mask
        )
        
        
        return (seq2SeqModelOutput.last_hidden_state,
                seq2SeqModelOutput.cross_attentions,
                bidecoder_outputs.last_hidden_state, 
                bidecoder_outputs.cross_attentions)
  
class AdaLabel_BartForConditionalGeneration(RRGBartForConditionalGeneration):
    def __init__(self, config: AdaLabelConfig) -> None:
        super().__init__(config)
        self.bidecoder = TransformerBiDecoder(config)
        self.bidecoder_generator = nn.Linear(self.config.d_model, self.config.vocab_size)
        self.post_init()
    
    def forward(self, input_ids: torch.LongTensor = None, attention_mask: Optional[torch.Tensor] = None, decoder_input_ids: Optional[torch.LongTensor] = None, decoder_attention_mask: Optional[torch.LongTensor] = None, head_mask: Optional[torch.Tensor] = None, decoder_head_mask: Optional[torch.Tensor] = None, cross_attn_head_mask: Optional[torch.Tensor] = None, encoder_outputs: Optional[List[torch.FloatTensor]] = None, past_key_values: Optional[List[torch.FloatTensor]] = None, inputs_embeds: Optional[torch.FloatTensor] = None, decoder_inputs_embeds: Optional[torch.FloatTensor] = None, labels: Optional[torch.LongTensor] = None, use_cache: Optional[bool] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None, pi_input_ids = None, pi_attention_mask = None) -> Union[Tuple, Seq2SeqLMOutput]:
        seq2SeqLMOutput = super().forward(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, head_mask, decoder_head_mask, cross_attn_head_mask, encoder_outputs, past_key_values, inputs_embeds, decoder_inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict, pi_input_ids = pi_input_ids, pi_attention_mask = pi_attention_mask)
        memory_bank = seq2SeqLMOutput.encoder_last_hidden_state
        generate1 = seq2SeqLMOutput.logits
        if pi_input_ids is not None:
            # hidden_states = torch.cat([encoder_outputs[0], pi_encoder_outputs[0]],dim=1)
            enc_attention_mask = torch.cat([attention_mask,pi_attention_mask],dim=1)
        else:
            # hidden_states = encoder_outputs[0]
            enc_attention_mask = attention_mask
            
        bidec_outs = self.bidecoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=memory_bank,
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
        generate2 = self.bidecoder_generator(bidec_outs.last_hidden_state)
        
        
        dict_outs = {
            'logits':generate1,
            'generate2': generate2
        }
        
        return ModelOutput(**dict_outs)
    
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
    
class PHV_BartForConditionalGeneration(AdaLabel_BartForConditionalGeneration):
    def __init__(self, config: AdaLabelConfig):
        super().__init__(config)
        self.state_linear = nn.Linear(self.config.d_model*2, self.config.d_model)
        self.gru = nn.GRU(input_size=config.d_model, hidden_size=config.d_model,batch_first = True, num_layers=1)
        self.post_linear1 = nn.Linear(self.config.d_model*3, self.config.d_model*2)
        self.post_linear2 = nn.Linear(self.config.d_model*2, self.config.d_model*2)
        self.prior_linear1 = nn.Linear(self.config.d_model*2, self.config.d_model*2)
        self.prior_linear2 = nn.Linear(self.config.d_model*2, self.config.d_model*2)
        self.begin_ids = config.begin_ids
        self.end_ids = config.end_ids
        self.post_init()
        
    def forward(self, input_ids: torch.LongTensor = None, attention_mask: Optional[torch.Tensor] = None, decoder_input_ids: Optional[torch.LongTensor] = None, decoder_attention_mask: Optional[torch.LongTensor] = None, head_mask: Optional[torch.Tensor] = None, decoder_head_mask: Optional[torch.Tensor] = None, cross_attn_head_mask: Optional[torch.Tensor] = None, encoder_outputs: Optional[List[torch.FloatTensor]] = None, past_key_values: Optional[List[torch.FloatTensor]] = None, inputs_embeds: Optional[torch.FloatTensor] = None, decoder_inputs_embeds: Optional[torch.FloatTensor] = None, labels: Optional[torch.LongTensor] = None, use_cache: Optional[bool] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None, pi_input_ids = None, pi_attention_mask = None, bat_triple_idxs = None, **kwargs) -> Union[Tuple, Seq2SeqLMOutput]:
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
            tgt_encoder_outputs = self.model.encoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            src_encoder_outputs = encoder_outputs
            tgt_encoder_outputs = None
        pi_encoder_outputs = None
        if pi_input_ids is not None:
            pi_encoder_outputs = self.model.encoder(
                input_ids=pi_input_ids,
                attention_mask=pi_attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        src_len = kwargs.pop('src_len', None)
        
        # 采样计算隐变量和batch平均KL散度
        kl_loss, global_z = self.do_sample(src_encoder_outputs, pi_encoder_outputs, tgt_encoder_outputs, src_len)
        # 分组过GRU
        gru_group_states, group_mask = self.do_group(bat_triple_idxs, src_encoder_outputs, pi_encoder_outputs, tgt_encoder_outputs, global_z)
        
        # 拼接GRU输出、src_encoder_outputs[0]、pi_encoder_outputs[0]
        if pi_encoder_outputs is not None:
            enc_hidden_states = torch.cat([src_encoder_outputs[0], pi_encoder_outputs[0]],dim=1)
            enc_attention_mask = torch.cat([attention_mask,pi_attention_mask],dim=1)
        else:
            enc_hidden_states = encoder_outputs[0]
            enc_attention_mask = attention_mask
        
        new_enc_hidden_states = torch.cat([enc_hidden_states, gru_group_states],dim=1)
        new_enc_attention_mask = torch.cat([enc_attention_mask,group_mask],dim=1)
        
        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=new_enc_hidden_states, # 作为decoder的输入
            encoder_attention_mask=new_enc_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        lm_logits = self.lm_head(decoder_outputs[0]) + self.final_logits_bias
        
        if self.config.use_copy_mechanism:
        # 计算copy 机制logits
            if pi_input_ids is not None:
                src_ids = torch.cat([input_ids,pi_input_ids],dim=1)
            else:
                src_ids = input_ids
                            # [bs,src_seq_len,d_model] * [bs,d_model, tgt_seq_len] --> [bs,tgt_seq_len,src_seq_len]
            cp_outs = torch.bmm(torch.tanh(self.cp_head(enc_hidden_states)),(decoder_outputs[0].transpose(1,2))).transpose(1,2)
            # outputs:[bs, src_seq_len, hidden_dim] --> [bs, tgt_seq_len, vocab_size]
            expanded_src_ids = src_ids[:, None, :].expand(src_ids.shape[0], lm_logits.shape[1], src_ids.shape[1])
            cp_src_logits = torch.zeros_like(lm_logits,dtype=cp_outs.dtype).scatter_(2, expanded_src_ids , cp_outs)
            
            lm_logits = lm_logits+cp_src_logits
        
        generate1 = lm_logits
    
        bidec_outs = self.bidecoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=enc_hidden_states,
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
        generate2 = self.bidecoder_generator(bidec_outs.last_hidden_state)

        dict_outs = {
            'logits':generate1,
            'generate2': generate2,
            'kl_loss':kl_loss
        }
        
        return ModelOutput(**dict_outs)
    def do_sample(self, src_encoder_outputs, pi_encoder_outputs, tgt_encoder_outputs,src_len=None):
        # 进行先验概率、后验概率、KL散度、隐变量的计算以及高斯分布采样
        src_cls_states = src_encoder_outputs[0][:,0,:] # [bs, d_model]
        if pi_encoder_outputs is not None:
            pi_cls_states = pi_encoder_outputs[0][:,0,:]
        else:
            pi_cls_states = src_encoder_outputs[0][:,src_len,:]
        
        
        if self.training:# 训练阶段使用后验概率
            tgt_cls_states = tgt_encoder_outputs[0][:,0,:]
            post_inp = torch.cat((src_cls_states, pi_cls_states, tgt_cls_states), 1)
            post_mu , post_logvar = torch.split(self.post_linear2(F.tanh(self.post_linear1(post_inp))),self.config.d_model,1)
            prior_inp = torch.cat((src_cls_states, pi_cls_states), 1)
            prior_mu , prior_logvar = torch.split(self.prior_linear2(F.tanh(self.prior_linear1(prior_inp))),self.config.d_model,1)
            global_z = sample_gaussian(post_mu.shape, post_mu , post_logvar)
            kl_div = KL_divergence(prior_mu, prior_logvar, post_mu, post_logvar)
            kl_div = kl_div/prior_mu.shape[0] # 平均kl散度
        else:# 推理阶段使用先验概率
            prior_inp = torch.cat((src_cls_states, pi_cls_states), 1)
            prior_mu , prior_logvar = torch.split(self.prior_linear2(F.tanh(self.prior_linear1(prior_inp))),self.config.d_model,1)
            global_z = sample_gaussian(prior_mu.shape, prior_mu , prior_logvar)
            kl_div = 0
        
        return kl_div, global_z

        
    def do_group(self, bat_triple_idxs, src_encoder_outputs, pi_encoder_outputs, tgt_encoder_outputs, global_z):
        # global_z用于初始化gru，然后group循环过gru
        src_last_states = src_encoder_outputs[0]
        src_cls_states = src_encoder_outputs[0][:,0,:]
        # pi_cls_states = pi_encoder_outputs[0][:,0,:]
        # tgt_cls_states = tgt_encoder_outputs[0][:,0,:]
        triples_states = [] # 最终要形成 [bs, group_num+2, hidden_size]
        
        device = get_tensor_device(src_last_states)
        real_group_num = []
        # TODO：是否需要加上mask
        for bs_idx, groups in enumerate(bat_triple_idxs):
            tmp = [self.model.shared(torch.tensor(self.begin_ids,device=device))]
            num = 1
            for sep_idxs in groups:
                if sep_idxs!=[0]:
                    t_idxs = torch.tensor(sep_idxs,device=device)
                    mean_pool_res = torch.mean(
                        torch.index_select(src_last_states[bs_idx], 0, t_idxs), # [d_model]
                        dim = 0
                    ) # [1, hidden_size]
                    num+=1
                else:
                    mean_pool_res = torch.zeros_like(tmp[0])
                tmp.append(mean_pool_res)
            tmp.insert(num,self.model.shared(torch.tensor(self.end_ids,device=device)))
            num+=1
            tmp = torch.stack(tmp, dim=0)
            triples_states.append(tmp)
            real_group_num.append(num)
        real_group_num = torch.tensor(real_group_num)
        triples_states = torch.stack(triples_states, dim=0)
        init_state = torch.cat([src_cls_states, global_z], dim=-1)
        init_state = self.state_linear(init_state).unsqueeze(0)
        inps_packed = nn.utils.rnn.pack_padded_sequence(triples_states, real_group_num, batch_first=True, enforce_sorted=False)
        packed_group_states, states = self.gru(inps_packed, init_state)
        gru_group_states, _ = nn.utils.rnn.pad_packed_sequence(packed_group_states,batch_first=True)
        group_mask = torch.zeros((gru_group_states.shape[0],gru_group_states.shape[1]),device=device)
        for idx in range(real_group_num.shape[0]):
            l = real_group_num[idx]
            group_mask[idx,:l] = 1
        return gru_group_states, group_mask
    def no_group(self, bat_triple_idxs, src_encoder_outputs, tgt_encoder_outputs):
        src_last_states = src_encoder_outputs[0] # [bs, seq_len, hidden_size]
        tgt_last_states = tgt_encoder_outputs[0] # [bs, seq_len, hidden_size]
        src_cls_states = src_last_states[:,0,:]
        tgt_cls_states = tgt_last_states[:,0,:]
        triples_states = [] # 最终要形成 [bs, hidden_size]
        # 不分组
        for idx, groups in enumerate(bat_triple_idxs):
            for sep_idxs in groups:
                t_idxs = torch.tensor(sep_idxs)
                mean_pool_res = torch.mean(
                    torch.index_select(src_last_states[idx], 0, t_idxs),
                    dim = 0
                ) # [1, hidden_size]
                triples_states.append(mean_pool_res)
        triples_states = torch.stack(triples_states, dim=0)
        prior_logits = torch.cat([src_cls_states], )

class AdaLabelForConditionalGeneration(BartPretrainedModel):
    def __init__(self, config: AdaLabelConfig) -> None:
        super().__init__(config)
        self.model = AdaLabelModel(config)
        self.generator = nn.Linear(self.config.d_model, self.config.vocab_size)
        self.bidecoder_generator = nn.Linear(self.config.d_model, self.config.vocab_size)
        self.post_init()
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pi_input_ids = None, pi_attention_mask = None
        ):
        
        dec_outs, dec_atten, bidec_outs, bidec_atten = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pi_input_ids = pi_input_ids, pi_attention_mask = pi_attention_mask
        )
        generate1 = self.generator(dec_outs)
        generate2 = self.bidecoder_generator(bidec_outs)
        
        dict_outs = {
            'logits':generate1,
            'generate2': generate2
        }
        
        return ModelOutput(**dict_outs)
        

    def get_encoder(self):
        return self.model.get_encoder()
    
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
    
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

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

class AdaLabLoss(nn.Module):
    """
    With adaptive label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, tgt_vocab_size, batch_size, ignore_index=-100, device="cuda", reduction='sum',
                 temperature=1, eos_index=3):
        self.ignore_index = ignore_index
        self.eos_index = eos_index
        self.tgt_vocab_size = tgt_vocab_size
        super(AdaLabLoss, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.reduction = reduction

        self.step = 0
        self.temperature = temperature
        self.top_head = 2
        self.top_tail = 500
        self.margin = 0.2
        self.alpha_param = 2
        self.topk = 5

    def forward(self, output, target, tgt_batch=None, label_scores=None):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        v = self._get_v(label_scores, target)
        epsilon = self._get_epsilon(output, target, v)

        confidence = 1 - epsilon
        smoothing_penalty = epsilon.unsqueeze(-1) * v

        model_prob = torch.zeros_like(output, device=output.device, dtype=torch.float)
        model_prob.scatter_(1, target.unsqueeze(1), confidence.unsqueeze(1))
        model_prob = model_prob + smoothing_penalty
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction=self.reduction)

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))

    def _get_epsilon(self, output, target, v):
        probs = output.detach().clone().exp()
        prob_max = probs.max(dim=1)[0]
        prob_gtruth = probs.gather(dim=1, index=target.unsqueeze(1)).squeeze()
        epsilon = 1 - prob_max
        maxv = v.max(dim=-1)[0]
        up_bond = 1 / (1 + maxv) - self.margin
        mask = epsilon.gt(up_bond)
        epsilon[mask] = up_bond[mask]
        alpha = (prob_gtruth / prob_max).pow(self.alpha_param)
        epsilon = alpha * epsilon
        return epsilon

    def _get_v(self, label_scores, target):
        v = label_scores.detach().clone()
        v = v / self.temperature
        v.scatter_(1, target.unsqueeze(1), -float('inf'))
        v[:, self.ignore_index] = -float('inf')

        # truncate tail
        upper_values, upper_indices = torch.topk(v, self.top_tail, dim=1)
        kth_upper = upper_values[:, -1].view([-1, 1])
        kth_upper = kth_upper.repeat([1, v.shape[1]]).float()
        upper_ignore = torch.lt(v, kth_upper)
        v = v.masked_fill(upper_ignore, -10000)

        # truncate head
        lower_values, lower_indices = torch.topk(v, self.top_head, dim=1)
        kth_lower = lower_values[:, -1].view([-1, 1])
        kth_lower = kth_lower.repeat([1, v.shape[1]]).float()
        lower_ignore = torch.gt(v, kth_lower)
        v = v.masked_fill(lower_ignore, -10000)

        v = v.softmax(dim=-1)
        return v

    def _compute_entropy(self, output):
        entropy = -torch.sum(output.exp() * output, -1)
        return entropy
     
