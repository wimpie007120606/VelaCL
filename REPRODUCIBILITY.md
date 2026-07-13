# Reproducibility

- Python dependencies are bounded in `pyproject.toml`; Node versions are locked after `npm install`.
- Seed 42 controls Python, NumPy, PyTorch, shuffling, initialization and replay.
- PyTorch deterministic algorithms are requested with warnings for unsupported operations.
- `configs/base.yaml` is the experiment source of truth.
- Dataset SHA-256, configuration, Git commit, platform, Torch/CUDA state, duration and checkpoint path are recorded.
- A checkpoint resumes at the next stage. The current minimal optimizer is recreated per stage by design, so no unstated optimizer state spans stages.
- CPU and accelerator kernels can yield small numeric differences. Report hardware and repeat seeds before comparative claims.

Clean-room check:

```bash
git clone <repository> velacl && cd velacl
make install data experiment test
```

