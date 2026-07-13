from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from velacl.data import load_events
from velacl.trainer import evaluate, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/streaming/events.jsonl")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    events = [event for event in load_events(Path(args.data)) if event.split == "test"]
    labels = checkpoint["labels"]
    print(
        json.dumps(
            evaluate(model, events, {label: i for i, label in enumerate(labels)}, device), indent=2
        )
    )


if __name__ == "__main__":
    main()
