# Reinforcement Learning for Active Flow Control Applied to Road Vehicles

Final Year Project — Suhail Halaby
Supervisor: Georgios Rigas, Department of Aeronautics, Imperial College London

**Research question:** can RL algorithms discovered through meta-learning (e.g. DiscoRL, Oh et al. 2025)
outperform bespoke frameworks like REACT on active flow control problems?

## Repository layout

```
.
├── Presentation/            Beamer slides (Imperial theme). Synced with Overleaf, compile with XeLaTeX.
│
├── environments/
│   └── hydrogym/            Everything HydroGym-specific
│       ├── envs/            Custom flow configurations / env subclasses (e.g. road-vehicle bluff body)
│       ├── meshes/          Mesh files (.msh) and mesh-generation scripts
│       ├── configs/         Solver + environment settings (Re, dt, actuation, observation probes)
│       ├── scripts/         Solver-only runs: steady states, unactuated baselines, sanity checks
│       └── checkpoints/     Restart / base-flow checkpoints (large files git-ignored)
│
├── src/afc/                 Project Python package (`pip install -e .`)
│   ├── agents/              Learning rules: discovered (DiscoRL-style) and baselines (PPO, REACT, ...)
│   ├── meta/                Meta-optimisation: meta-network, meta-gradients, agent populations
│   ├── envs/                Gym-style wrappers around HydroGym and cheaper proxy environments
│   └── utils/               Logging, seeding, plotting helpers
│
├── experiments/
│   ├── configs/             One config file per experiment (YAML)
│   ├── scripts/             Entry points: train / evaluate / meta-train
│   └── runs/                Output of each run, one folder per run (git-ignored)
│
├── external/                Third-party code pinned as git submodules (e.g. hydrogym, disco_rl)
├── notebooks/               Exploration and analysis notebooks
├── data/                    raw/ and processed/ data (git-ignored)
├── results/                 Curated figures/ and tables/ for the report and slides
└── tests/                   Unit tests for src/afc
```

## Conventions

- Code that runs a simulation lives in `environments/hydrogym/`; code that *learns* lives in `src/afc/`;
  code that *ties the two together for a specific study* lives in `experiments/`.
- Run outputs go to `experiments/runs/<date>_<name>/` and are not committed. Copy the figures worth
  keeping into `results/figures/`, then reference them from the slides with `../results/figures/...`.
- The folder is named `environments/hydrogym/` rather than `hydrogym/` so it never shadows the
  installed `hydrogym` Python package.

## Setup

HydroGym is built on Firedrake, which is easiest to run in the official HydroGym container or a
Firedrake virtual environment. Install HydroGym there, then install this package on top:

```bash
pip install -e .
```
