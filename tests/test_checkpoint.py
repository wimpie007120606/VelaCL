from dataclasses import replace

import torch

from velacl.config import Config, ModelConfig
from velacl.data import build_events
from velacl.model import ByteTransformer
from velacl.trainer import load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip(tmp_path):
    config = replace(
        Config(), model=ModelConfig(max_length=16, d_model=8, nhead=2, dim_feedforward=16)
    )
    model = ByteTransformer(config.model, 2)
    path = tmp_path / "model.pt"
    replay = build_events()[:2]
    save_checkpoint(path, model, ["a", "b"], config, 3, "naive", replay)
    restored, metadata = load_checkpoint(path, torch.device("cpu"))
    assert metadata["stage"] == 3
    assert [item["id"] for item in metadata["replay_events"]] == [event.id for event in replay]
    assert all(
        torch.equal(a, b)
        for a, b in zip(model.state_dict().values(), restored.state_dict().values(), strict=True)
    )
