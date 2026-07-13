# VelaCL V1 technical report

## Abstract

VelaCL tests four update strategies for a small multilingual Transformer under a six-stage enterprise-intent stream. This report is completed from generated artifacts after the reference run; no placeholder number should be reused on a résumé.

## Protocol

All methods use the same byte Transformer, seed, optimizer family, stage order and test events. Static updates only at stage 0; naive uses the incoming stage; random replay uses a 64-event memory; active methods use an eight-event annotation budget and balance a 64-event memory by language/intent. Macro-F1 is evaluated for every task after each stage. Average forgetting is the mean positive difference between a task's best post-learning score and final score.

## Measured results

Run `make experiment` to regenerate `experiments/runs/summary.json`. The checked-in table below is updated only from that artifact.

| Method | Average task macro-F1 | Average forgetting | Backward transfer |
|---|---:|---:|---:|
| Static | 7.03% | 0.00% | 0.00% |
| Naive sequential | 3.30% | 12.79% | -12.52% |
| Random replay | **30.66%** | 1.00% | 7.06% |
| Active balanced replay | 28.53% | **0.69%** | **9.30%** |

The combined active selector did not beat random replay on average task macro-F1, but it retained tasks slightly better and produced stronger backward transfer while labeling at most eight new examples per stage. This is encouraging, not evidence of a novel general result.

### Selection ablation

| Selector | Average task macro-F1 | Average forgetting | Backward transfer |
|---|---:|---:|---:|
| Combined six-factor score | **28.53%** | **0.69%** | **9.30%** |
| Uncertainty only | 26.91% | 1.52% | 6.33% |
| Diversity only | 24.08% | 3.31% | 5.38% |

On this seed and fixture, the combined score outperformed both single-factor ablations. The experiment does not isolate all six factors independently and must be repeated on realistic data.

## Validity

One seed and a small curated fixture cannot support statistical or external-language claims. Active selection uses known labels to simulate a forgetting-risk signal. The comparison measures this repository's executable pipeline, not superiority over published continual-learning systems.
