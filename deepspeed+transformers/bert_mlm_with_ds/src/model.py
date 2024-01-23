class RobertaLMHeadWithMaskedPredict(RobertaLMHead):
    def __init__(self,
                 config: RobertaConfig,
                 embedding_weight: Optional[torch.Tensor] = None) -> None:
        super(RobertaLMHeadWithMaskedPredict, self).__init__(config)
        if embedding_weight is not None:
            self.decoder.weight = embedding_weight

    def forward(  # pylint: disable=arguments-differ
        self,
        features: torch.Tensor,
        masked_token_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """The current `transformers` library does not provide support
        for masked_token_indices. This function provides the support, by
        running the final forward pass only for the masked indices. This saves
        memory

        Args:
            features (torch.Tensor):
                The features to select from. Shape (batch, seq_len, h_dim)
            masked_token_indices (torch.Tensor, optional):
                The indices of masked tokens for index select. Defaults to None.
                Shape: (num_masked_tokens,)

        Returns:
            torch.Tensor:
                The index selected features. Shape (num_masked_tokens, h_dim)

        """
        if masked_token_indices is not None:
            features = torch.index_select(
                features.view(-1, features.shape[-1]), 0, masked_token_indices)
        return super().forward(features)


class RobertaMLMModel(RobertaPreTrainedModel):
    def __init__(self, config: RobertaConfig, encoder: RobertaModel) -> None:
        super().__init__(config)
        self.encoder = encoder
        self.lm_head = RobertaLMHeadWithMaskedPredict(
            config, self.encoder.embeddings.word_embeddings.weight)
        self.lm_head.apply(self._init_weights)

    def forward(
            self,
            src_tokens: torch.Tensor,
            attention_mask: torch.Tensor,
            tgt_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """The forward pass for the MLM task

        Args:
            src_tokens (torch.Tensor):
                The masked token indices. Shape: (batch, seq_len)
            attention_mask (torch.Tensor):
                The attention mask, since the batches are padded
                to the largest sequence. Shape: (batch, seq_len)
            tgt_tokens (torch.Tensor):
                The output tokens (padded with `config.pad_token_id`)

        Returns:
            torch.Tensor:
                The MLM loss
        """
        # shape: (batch, seq_len, h_dim)
        sequence_output, *_ = self.encoder(input_ids=src_tokens,
                                           attention_mask=attention_mask,
                                           return_dict=False)

        pad_token_id = self.config.pad_token_id
        # (labels have also been padded with pad_token_id)
        # filter out all masked labels
        # shape: (num_masked_tokens,)
        masked_token_indexes = torch.nonzero(
            (tgt_tokens != pad_token_id).view(-1)).view(-1)
        # shape: (num_masked_tokens, vocab_size)
        prediction_scores = self.lm_head(sequence_output, masked_token_indexes)
        # shape: (num_masked_tokens,)
        target = torch.index_select(tgt_tokens.view(-1), 0,
                                    masked_token_indexes)

        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        masked_lm_loss = loss_fct(
            prediction_scores.view(-1, self.config.vocab_size), target)
        return masked_lm_loss


def create_model(num_layers: int, num_heads: int, ff_dim: int, h_dim: int,
                 dropout: float) -> RobertaMLMModel:
    """Create a Bert model with the specified `num_heads`, `ff_dim`,
    `h_dim` and `dropout`

    Args:
        num_layers (int):
            The number of layers
        num_heads (int):
            The number of attention heads
        ff_dim (int):
            The intermediate hidden size of
            the feed forward block of the
            transformer
        h_dim (int):
            The hidden dim of the intermediate
            representations of the transformer
        dropout (float):
            The value of dropout to be used.
            Note that we apply the same dropout
            to both the attention layers and the
            FF layers

    Returns:
        RobertaMLMModel:
            A Roberta model for MLM task

    """
    roberta_config_dict = {
        "attention_probs_dropout_prob": dropout,
        "bos_token_id": 0,
        "eos_token_id": 2,
        "hidden_act": "gelu",
        "hidden_dropout_prob": dropout,
        "hidden_size": h_dim,
        "initializer_range": 0.02,
        "intermediate_size": ff_dim,
        "layer_norm_eps": 1e-05,
        "max_position_embeddings": 514,
        "model_type": "roberta",
        "num_attention_heads": num_heads,
        "num_hidden_layers": num_layers,
        "pad_token_id": 1,
        "type_vocab_size": 1,
        "vocab_size": 50265,
    }
    roberta_config = RobertaConfig.from_dict(roberta_config_dict)
    roberta_encoder = RobertaModel(roberta_config)
    roberta_model = RobertaMLMModel(roberta_config, roberta_encoder)
    return roberta_model
