from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict

import torch
from torch import nn

from velacl.config import ModelConfig

PAD_ID = 0
CLS_ID = 1
BYTE_OFFSET = 2
VOCAB_SIZE = 258


def encode_text(text: str, max_length: int) -> list[int]:
    byte_ids = [value + BYTE_OFFSET for value in text.encode("utf-8")[: max_length - 1]]
    return [CLS_ID, *byte_ids] + [PAD_ID] * (max_length - len(byte_ids) - 1)


class ByteTransformer(nn.Module):
    """Compact byte-level Transformer encoder for multilingual intent classification."""

    def __init__(self, config: ModelConfig, num_labels: int):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        self.token_embedding = nn.Embedding(VOCAB_SIZE, config.d_model, padding_idx=PAD_ID)
        self.position_embedding = nn.Embedding(config.max_length, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model, num_labels)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids: torch.Tensor, return_embedding: bool = False):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = self.token_embedding(input_ids) * math.sqrt(self.config.d_model)
        hidden = hidden + self.position_embedding(positions)[None, :, :]
        hidden = self.encoder(hidden, src_key_padding_mask=input_ids.eq(PAD_ID))
        pooled = self.norm(hidden[:, 0])
        logits = self.classifier(pooled)
        return (logits, pooled) if return_embedding else logits

    def metadata(self) -> dict:
        return {
            "architecture": "byte-transformer-encoder",
            "model_config": asdict(self.config),
            "num_labels": self.num_labels,
        }


def batch_encode(texts: Iterable[str], max_length: int, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [encode_text(text, max_length) for text in texts], dtype=torch.long, device=device
    )
