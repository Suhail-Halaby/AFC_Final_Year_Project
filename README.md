# Reinforcement Learning for Active Flow Control Applied to Road Vehicles

Final Year Project — Suhail Halaby (supervisor: Georgios Rigas), Imperial College London

```
Presentation/          Beamer slides, synced with Overleaf (compile with XeLaTeX)
Lit_Review/            Literature review
Implementation/
├── hydrogym/          HydroGym flow environments and simulations
├── jax_toy_problems/  Toy PDEs and ODEs written in JAX
├── agents/            RL agents and learning rules
└── results/           Figures and data worth keeping
```

## Setup

See [SETUP.md](SETUP.md) for installing HydroGym (MAIA, JAX, Firedrake) in Docker on Windows, attaching VS Code, and running the solvers from this repo.

Quick MAIA smoke test, inside the container:

```bash
cd /workspace/Implementation/hydrogym
./run_maia_test.sh
```
