from transformers.models.bart.configuration_bart import *


class AdaLabelConfig(BartConfig):
    def __init__(self, vocab_size=30522, max_position_embeddings=512, encoder_layers=6, encoder_ffn_dim=512, encoder_attention_heads=8, decoder_layers=6, decoder_ffn_dim=512, decoder_attention_heads=8, encoder_layerdrop=0.1, decoder_layerdrop=0.1, activation_function="relu", d_model=512, dropout=0.1, attention_dropout=0.1, activation_dropout=0.1, init_std=0.02, classifier_dropout=0, scale_embedding=False, use_cache=True, num_labels=3, pad_token_id=1, bos_token_id=0, eos_token_id=2, is_encoder_decoder=True, decoder_start_token_id=2, forced_eos_token_id=2, **kwargs):
        super().__init__(vocab_size, max_position_embeddings, encoder_layers, encoder_ffn_dim, encoder_attention_heads, decoder_layers, decoder_ffn_dim, decoder_attention_heads, encoder_layerdrop, decoder_layerdrop, activation_function, d_model,
                         dropout, attention_dropout, activation_dropout, init_std, classifier_dropout, scale_embedding, use_cache, num_labels, pad_token_id, bos_token_id, eos_token_id, is_encoder_decoder, decoder_start_token_id, forced_eos_token_id, **kwargs)
