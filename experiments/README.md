# Experiments

Each study is a config in `configs/` run by a script in `scripts/`; outputs land in `runs/`.

```
experiments/
├── configs/   e.g. cylinder_ppo_baseline.yaml, pinball_discorl.yaml
├── scripts/   e.g. train.py, evaluate.py, meta_train.py
└── runs/      <YYYY-MM-DD>_<config-name>/  (logs, checkpoints, plots — git-ignored)
```

Keep a one-line note per run in `runs/LOG.md` (committed) so results can be traced back to configs.
