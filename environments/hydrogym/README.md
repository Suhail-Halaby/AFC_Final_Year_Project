# HydroGym workspace

Dedicated space for the CFD side of the project: flow configurations, meshes and solver runs built on
[HydroGym](https://github.com/dynamicslab/hydrogym) (Firedrake-based).

| Folder         | Contents                                                                       |
|----------------|--------------------------------------------------------------------------------|
| `envs/`        | Custom `FlowConfig` / env subclasses, e.g. a 2D road-vehicle (Ahmed-type) body |
| `meshes/`      | `.msh` files and the scripts that generate them                                |
| `configs/`     | Reynolds number, timestep, actuator and probe layouts, episode length          |
| `scripts/`     | Solver-only jobs: base-flow/steady-state solves, unactuated runs, validation   |
| `checkpoints/` | Base flows and restart files used to initialise episodes (git-ignored)         |

Suggested progression: validate on the built-in benchmarks (cylinder, pinball, cavity) first, then
move to the road-vehicle geometry.

The RL wrappers that expose these flows to agents live in `src/afc/envs/`, not here.
