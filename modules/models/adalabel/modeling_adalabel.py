from transformers.models.bart.modeling_bart import *
import torch.nn as nn
import torch.nn.functional as F
from modules.models.adalabel.configuration_adalabel import AdaLabelConfig
from transformers.utils.generic import ModelOutput
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

    

class AdaLabelModel(BartModel):
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
                return_dict: Optional[bool] = None) -> Union[Tuple, Seq2SeqModelOutput]:
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
        )
        
        
        return (seq2SeqModelOutput.last_hidden_state,
                seq2SeqModelOutput.cross_attentions,
                bidecoder_outputs.last_hidden_state, 
                bidecoder_outputs.cross_attentions)

class AdaLabel_BartForConditionalGeneration(BartForConditionalGeneration):
    def __init__(self, config: AdaLabelConfig) -> None:
        super().__init__(config)
        self.bidecoder = TransformerBiDecoder(config)
        self.bidecoder_generator = nn.Linear(self.config.d_model, self.config.vocab_size)
        self.post_init()
    
    def forward(self, input_ids: torch.LongTensor = None, attention_mask: Optional[torch.Tensor] = None, decoder_input_ids: Optional[torch.LongTensor] = None, decoder_attention_mask: Optional[torch.LongTensor] = None, head_mask: Optional[torch.Tensor] = None, decoder_head_mask: Optional[torch.Tensor] = None, cross_attn_head_mask: Optional[torch.Tensor] = None, encoder_outputs: Optional[List[torch.FloatTensor]] = None, past_key_values: Optional[List[torch.FloatTensor]] = None, inputs_embeds: Optional[torch.FloatTensor] = None, decoder_inputs_embeds: Optional[torch.FloatTensor] = None, labels: Optional[torch.LongTensor] = None, use_cache: Optional[bool] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> Union[Tuple, Seq2SeqLMOutput]:
        seq2SeqLMOutput = super().forward(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, head_mask, decoder_head_mask, cross_attn_head_mask, encoder_outputs, past_key_values, inputs_embeds, decoder_inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict)
        memory_bank = seq2SeqLMOutput.encoder_last_hidden_state
        generate1 = seq2SeqLMOutput.logits
    
        bidec_outs = self.bidecoder(
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
        )
        generate2 = self.bidecoder_generator(bidec_outs.last_hidden_state)
        
        
        dict_outs = {
            'logits':generate1,
            'generate2': generate2
        }
        
        return ModelOutput(**dict_outs)
    

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
     
