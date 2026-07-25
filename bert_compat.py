"""BERT forward helpers that work across transformers 3.x / 4.x / 5.x."""
from __future__ import annotations

import torch


def bert_sequence_and_pooler(
    bert_model,
    input_ids: torch.Tensor,
    token_type_ids: torch.Tensor,
    attention_mask: torch.Tensor,
):
    outputs = bert_model(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )
    sequence_output = outputs.last_hidden_state
    pooled_output = outputs.pooler_output
    if pooled_output is None:
        pooled_output = sequence_output[:, 0]
    return sequence_output, pooled_output
