# VelaCL byte Transformer model card

The V1 model is a randomly initialized byte-level Transformer encoder (48 hidden dimensions, one layer by default) trained for eight-way intent classification. It is optimized for a fast, inspectable continual-learning experiment—not general language understanding.

Inputs are UTF-8 bytes truncated to 95 bytes plus a classification token. The output is an intent softmax. Training uses cross-entropy and AdamW. Saved checkpoints include architecture parameters, ordered labels, method and completed stream stage.

Do not use the model for financial decisions, fraud adjudication, identity verification, safety filtering, translation, or customer-facing automation. Confidence can be poorly calibrated. Language detection and serving warnings are heuristics. Evaluation is limited to the project fixture and is reported by method in `experiments/runs`.

