import torch

from velacl.config import ModelConfig
from velacl.model import ByteTransformer, encode_text


def test_model_accepts_unicode_and_returns_logits():
    config = ModelConfig(max_length=32, d_model=16, nhead=4, dim_feedforward=32)
    model = ByteTransformer(config, 3)
    inputs = torch.tensor([encode_text("sawubona 🌍", 32)])
    assert model(inputs).shape == (1, 3)
