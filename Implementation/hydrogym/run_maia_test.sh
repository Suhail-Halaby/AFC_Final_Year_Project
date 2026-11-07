#!/bin/bash
# Launch the MAIA smoke test: prepare the workspace, then run Python + MAIA together.
# Usage:  ./run_maia_test.sh [ENV_NAME] [STEPS]
#   e.g.  ./run_maia_test.sh Cylinder_2D_Re200 50
set -e

ENV="${1:-Cylinder_2D_Re200}"
STEPS="${2:-50}"

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${HERE}/runs/${ENV}"              # outputs go here (git-ignore runs/)
PREP="${HOME}/hydrogym/examples/maia/getting_started/prepare_workspace.py"

# Make sure MAIA and the GPU venv are loaded (harmless if already done in .bashrc)
module unload nvhpc 2>/dev/null || true
module load MAIA
source "${HOME}/venvs/hydrogym_gpu/bin/activate"

# 1. Download the environment from Hugging Face and create the MAIA workspace
python "${PREP}" --env "${ENV}" --work-dir "${WORK}"

# 2. Run 1 Python process + 1 MAIA process, coupled over MPI
cd "${WORK}"
mpirun -np 1 python "${HERE}/run_maia_test.py" --env "${ENV}" --steps "${STEPS}" \
     : -np 1 maia properties.toml
