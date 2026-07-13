# VelaCL fixture data card

## Summary and license

The V1 fixture is project-owned, Apache-2.0-licensed text created for integration and continual-learning tests. It is not copied from MzansiText, INJONGO, MASSIVE, customer messages, or a model API. No personal data is included.

## Composition

Five language codes (`en`, `af`, `zu`, `xh`, `st`), four domains, eight intents and six chronological stages. Each record has a deterministic train/test designation plus operational metadata. `data/validation/schema.json` describes the stored form.

## Appropriate use

Use it to test streaming, replay, checkpoints, evaluation, serving and visualization. Do not use it to estimate real-world accuracy, fluency, dialect coverage, safety, cultural validity, or demographic fairness.

## Known problems

The corpus is very small; phrasing and label distributions are artificial; English suffix augmentation is simplistic; authors are not certified translators; train and test phrases are structurally close. Before external research claims, replace it with properly licensed corpora and native-speaker review, deduplicate train/test at semantic level, document consent/provenance, and run repeated-seed statistics.

