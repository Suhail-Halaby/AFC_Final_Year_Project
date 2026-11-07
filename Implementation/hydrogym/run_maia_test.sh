#!/bin/bash
# Launch the MAIA smoke test: prepare the workspace, then run Python + MAIA together.
# Usage:  ./run_maia_test.sh [ENV_NAME] [STEPS]
#   e.g.  ./run_maia_test.sh Cylinder_2D_Re200 50
set -e

ENV="${1:-Cylinder_2D_Re200}"
STEPS="${2:-50}"

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${HERE}/runs/${ENV}"              # outputs go here (git-ignored)
PREP="${HOME}/hydrogym/examples/maia/getting_started/prepare_workspace.py"
PROPS="properties_run.toml"             # the run properties file prepare_workspace creates

# Make sure MAIA and the GPU venv are loaded (harmless if already done in .bashrc)
module unload nvhpc 2>/dev/null || true
module unload MAIA 2>/dev/null || true
module load "${MAIA_MODULE:-MAIA/1.0-gompi-2024a-SystemCUDA-CPU}"
source "${HOME}/venvs/hydrogym_gpu/bin/activate"
export OMP_NUM_THREADS=1

# 1. Download the environment from Hugging Face and create the MAIA workspace
python "${PREP}" --env "${ENV}" --work-dir "${WORK}"

# 2. Run 1 Python process + 1 MAIA process, coupled over MPI.
#    --bind-to none / coll_hcoll_enable 0 silence harmless laptop warnings
#    (no CPU binding in Docker, no InfiniBand network card).
cd "${WORK}"
"${EBROOTOPENMPI:?OpenMPI module not loaded}/bin/mpirun" --bind-to none --mca coll_hcoll_enable 0 \
     -np 1 python "${HERE}/run_maia_test.py" --env "${ENV}" --steps "${STEPS}" \
     : -np 1 maia "${PROPS}" --silent