# Environment setup: HydroGym in Docker (Windows + NVIDIA GPU)

How to get HydroGym (MAIA, JAX, JAX-Fluids, Firedrake, NEK5000) running in Docker on a Windows laptop, and how to run code from this repo inside it.

Tested on Windows with an RTX 4050 (6 GB) and Docker Desktop on WSL2.

---

## 1. How it fits together

```
GitHub  ⇄  git push/pull  ⇄  this repo on Windows  ⇄  (mount)  ⇄  /workspace in the container
```

- **Image:** the read-only template downloaded from Docker Hub. It contains Linux, CUDA, MAIA, HydroGym and the Python environments.
- **Container (`hgym`):** a running instance of the image. It acts like a separate Linux machine that shares the laptop's CPU, RAM and GPU.
- **Mount (`-v`):** the repo folder on Windows appears inside the container at `/workspace`. It is the same folder, not a copy: edits on either side appear on both.

Rule of thumb: **code and results live in the repo; tools live in the container.** Git is run on Windows, code is run in the container.

---

## 2. One-off setup

### 2.1 Install Docker Desktop

1. Install Docker Desktop from docker.com and keep **Use WSL 2** ticked (needs admin rights).
2. Open Docker Desktop and wait for *Engine running*.
3. Check it works in PowerShell:
   ```powershell
   docker run hello-world
   ```
4. Keep the NVIDIA driver up to date on Windows. CUDA does **not** need installing on Windows; the image provides it.

### 2.2 Clone this repo (on Windows)

```powershell
git clone https://github.com/Suhail-Halaby/AFC_Final_Year_Project.git "C:\path\to\FYP Repo"
```

Avoid a folder that OneDrive actively syncs: syncing `.git` and simulation output causes file locks and slow I/O.

### 2.3 Pull the image

```powershell
docker pull clagemann/hydrogym-nvhpc-26.1_cuda-12.9_turing_ampere:latest
```

- Use **`turing_ampere`** for RTX 20/30/40-series (incl. the RTX 4050) and A100. Use `hopper_blackwell` only for H100 or RTX 50-series.
- The image is many GB; the first pull can take a long time. If interrupted, re-run the same command and it resumes.
- Leave ~30–40 GB free on `C:`.

### 2.4 Create the container

```powershell
$repo = "C:\path\to\FYP Repo"
docker run -it --gpus all --name hgym --shm-size=4g `
  -v "${repo}:/workspace" -w /workspace `
  clagemann/hydrogym-nvhpc-26.1_cuda-12.9_turing_ampere:latest
```

- `--gpus all` passes the GPU through. The startup message *"NVIDIA Driver was not detected"* is a false alarm under WSL2; `nvidia-smi` working is what matters.
- `--shm-size=4g` helps MPI (used by MAIA).
- Add `-p 6006:6006 -p 8888:8888` here if TensorBoard/Jupyter in the Windows browser will be needed: **ports and mounts can only be set when the container is created.**
- Run `docker run` **once**. Afterwards use `docker start` (section 3). Running it again gives a "name already in use" error.

### 2.5 Configure the shell inside the container

The prompt now looks like `[easybuild@<id> workspace]$`. Run once:

```bash
cat >> ~/.bashrc << 'EOF'
module unload nvhpc 2>/dev/null
module load MAIA
source ~/venvs/hydrogym_gpu/bin/activate
EOF
source ~/.bashrc
```

Why:
- MAIA is provided as an Lmod **module**, not on the PATH. A pre-loaded `nvhpc` module clashes with it, so it is unloaded first.
- Python lives in a venv. There are two: `hydrogym_gpu` (use this) and `hydrogym_cpu`.

### 2.6 Check everything works

```bash
ls /workspace                     # repo files
nvidia-smi                        # RTX 4050 visible
which maia                        # ~/eb_software/software/MAIA/1.0-NVHPC-26.1/bin/maia
python -c "import hydrogym, jax; print(hydrogym.__file__); print(jax.devices())"
pip list | grep -i -E "hydrogym|jax|torch|stable"
```

Expected: `jax.devices()` shows `[CudaDevice(id=0)]`. Lines like `Could not get kernel mode driver version` are harmless under WSL2.

---

## 3. Day-to-day use

| Task | Command (PowerShell unless noted) |
|---|---|
| Start a stopped container | `docker start hgym` |
| Open a shell in it | `docker exec -it hgym bash` (several at once is fine) |
| Start and attach in one go | `docker start -ai hgym` |
| List containers / check status | `docker ps -a` |
| Run a script without entering | `docker exec -it hgym bash -ic "cd /workspace/Implementation && python my_script.py"` |
| Run in the background | `docker exec -d hgym bash -ic "cd /workspace/Implementation && python train.py > runs/train.log 2>&1"` |
| Leave the container shell | `exit` (bash) |

**Am I inside the container?** A `PS C:\...>` prompt is Windows. `[easybuild@... ]$` is the container. `ls /.dockerenv` succeeds only inside.

**Workflow:** `git pull` on Windows → edit in VS Code → run in the container → `git commit` / `git push` on Windows.

---

## 4. VS Code attached to the container (recommended)

Running a script with VS Code's normal Run button uses **Windows Python**, which has none of these packages (`ModuleNotFoundError: No module named 'numpy'`). Don't create a Windows venv; attach to the container instead.

1. Install the **Dev Containers** extension (`ms-vscode-remote.remote-containers`, by Microsoft).
2. Make sure `hgym` is running (`docker start hgym`).
3. Click the **`><`** button bottom-left → **Attach to Running Container…** → **hgym**. A new window opens.
4. **File → Open Folder** → type `/workspace`. (The dialog starts in the container's home folder, so type the path.)
5. Install the **Python** extension in that window (**Install in Container**).
6. `Ctrl+Shift+P` → **Python: Select Interpreter** → **Enter interpreter path** → `/home/easybuild/venvs/hydrogym_gpu/bin/python`.

The bottom-left now shows *Container hgym*. Terminals, Run and the debugger in this window all run inside the container.

---

## 5. Running solvers from this repo

HydroGym is already installed (editable, from `/home/easybuild/hydrogym`), so any script in the repo can `import hydrogym`.

### JAX, JAX-Fluids, Firedrake: plain Python

```bash
python Implementation/jax_toy_problems/my_script.py
```

Firedrake is not in `hydrogym_gpu`; it is likely in `~/firedrake` or `hydrogym_cpu` (check with `~/venvs/hydrogym_cpu/bin/python -c "import firedrake"`).

### MAIA: needs a launcher

MAIA runs as a separate solver process coupled to Python over MPI (MPMD), so the script cannot be run with plain `python` or the Run button. Two steps:

1. Prepare a workspace (downloads the case from Hugging Face, creates `properties.toml`).
2. Launch Python and MAIA together from inside that workspace.

```bash
python ~/hydrogym/examples/maia/getting_started/prepare_workspace.py --env Cylinder_2D_Re200 --work-dir ./runs/Cylinder_2D_Re200
cd runs/Cylinder_2D_Re200
mpirun -np 1 python /workspace/path/to/my_script.py : -np 1 maia properties.toml
```

`Implementation/hydrogym/run_maia_test.sh` does all of this for the smoke test:

```bash
cd /workspace/Implementation/hydrogym
./run_maia_test.sh                      # Cylinder_2D_Re200, 50 steps, zero actuation
./run_maia_test.sh Cylinder_2D_Re200 200
```

It runs `MAIA_install_test.py`, prints the reward each step and writes `rewards.csv` into `runs/<env>/`.

Reference examples inside the container: `~/hydrogym/examples/maia/getting_started/` (`test_maia_env.py`, `train_sb3_maia.py`). Environment names: `existing_environments.yaml` in the HydroGym repo.

### Code layout

Keep reusable logic (wrappers, reward shaping, learning rules) in importable modules under `Implementation/agents/`, and keep run scripts thin. Only the environment-creation line then changes between backends.

---

## 6. Limitations

- **Anything outside `/workspace` is lost** if the container is removed (`docker rm`). That includes `.bashrc` edits and extra `pip install`s. Re-do section 2.5 after recreating.
- **Ports and mounts are fixed at creation.** To change them: `docker rm -f hgym`, then section 2.4 again.
- **6 GB VRAM:** 2D cases are fine; 3D MAIA LBM or the 3D JAX channel will likely run out of memory. Use the HPC (Apptainer) for those.
- **JAX preallocates ~75 % of GPU memory.** When sharing the GPU with MAIA or PyTorch: `export XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- **WSL2 caps RAM** (about half of system RAM by default). Raise it in `C:\Users\<you>\.wslconfig`:
  ```
  [wsl2]
  memory=12GB
  processors=8
  ```
  then `wsl --shutdown` and restart Docker Desktop.
- **Mounted-folder I/O is slower** than the container's own disk. For I/O-heavy runs, write raw output to `~/scratch` and copy what's needed into `/workspace`.
- **No GUI windows** from the container. Write `.vtu`/`.pvd`/`.png` to the repo and open them in ParaView on Windows.
- **Sleep or shutdown kills runs.** Keep the laptop plugged in and disable sleep for long jobs.
- **Run git on Windows,** not in the container, to avoid credential and ownership issues.

---

## 7. HPC (Apptainer)

Clusters don't run Docker. Convert the same image:

```bash
apptainer pull docker://clagemann/hydrogym-nvhpc-26.1_cuda-12.9_turing_ampere:latest
apptainer run --nv --bind $(pwd):/workspace hydrogym-nvhpc-26.1_cuda-12.9_turing_ampere_latest.sif
```

Pick the image matching the cluster's GPUs (`hopper_blackwell` for H100). Compute nodes often have no internet, so run `prepare_workspace.py` on a login node first.

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `ls : Cannot find path 'C:\.dockerenv'` | You are in PowerShell, not the container. `docker start hgym` then `docker exec -it hgym bash`. |
| `Conflict. The container name "/hgym" is already in use` | It already exists; use `docker start hgym`. |
| `Cannot load module "nvhpc-hpcx-cuda12/26.1" because these module(s) are loaded: nvhpc` | `module unload nvhpc` then `module load MAIA`. |
| `which maia` finds nothing | `module load MAIA` has not been run in this shell. |
| `python: command not found` | Venv not active: `source ~/venvs/hydrogym_gpu/bin/activate`. |
| `ModuleNotFoundError: No module named 'numpy'` from VS Code | VS Code is using Windows Python; attach to the container (section 4). |
| `bad interpreter: /bin/bash^M` | Windows line endings: `sed -i 's/\r$//' script.sh`. `.gitattributes` keeps `.sh` files LF from now on. |
| mpirun: "not enough slots" | Add `--oversubscribe` after `mpirun`. |
| `jax.devices()` shows only `CpuDevice` | Container created without `--gpus all`, or the Windows NVIDIA driver is outdated. |
| Image pull seems stuck | Extraction of big layers can look frozen at 100 %. If nothing changes for 15+ min: Ctrl+C, restart Docker Desktop, pull again. |

---

## References

- HydroGym docs: <https://dynamicslab.github.io/hydrogym/docs/quickstart>
- MAIA backend: <https://dynamicslab.github.io/hydrogym/docs/installation/maia>
- JAX backend: <https://dynamicslab.github.io/hydrogym/docs/installation/jax>
- HydroGym source: <https://github.com/dynamicslab/hydrogym>
