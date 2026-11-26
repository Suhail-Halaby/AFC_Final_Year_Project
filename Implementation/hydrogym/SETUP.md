# HydroGym in Docker: setup guide

How to get Docker ready to run HydroGym examples (MAIA, JAX, JAX-Fluids, Firedrake) on a Windows laptop, run code from this repo inside the container, and view results in ParaView.

Tested on Windows 11 with an i7-13700H, an RTX 4050 laptop GPU (6 GB) and Docker Desktop on WSL2.

**Contents**
1. [How it fits together](#1-how-it-fits-together)
2. [One-off setup](#2-one-off-setup)
3. [Starting and stopping](#3-starting-and-stopping)
4. [VS Code attached to the container](#4-vs-code-attached-to-the-container)
5. [Running HydroGym examples](#5-running-hydrogym-examples)
6. [ParaView](#6-paraview)
7. [Limitations](#7-limitations)
8. [HPC (Apptainer)](#8-hpc-apptainer)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. How it fits together

```
GitHub  ⇄  git push/pull  ⇄  repo folder on Windows  ⇄  (mount)  ⇄  /workspace in the container
```

- **Image:** the read-only template downloaded from Docker Hub. It contains Linux, CUDA, MAIA, NEK5000, ParaView server, HydroGym and the Python environments.
- **Container (`hgym`):** a running instance of the image. It behaves like a separate Linux machine that shares the laptop's CPU, RAM and GPU.
- **Mount (`-v`):** the repo folder on Windows appears inside the container at `/workspace`. It is the same folder, not a copy: an edit on either side shows up on both.

**Rule of thumb:** code and results live in the repo; tools live in the container. Run git on Windows, run code in the container.

What's in the image (user `easybuild`, home `/home/easybuild`):

| Path | Contents |
|---|---|
| `~/hydrogym` | HydroGym source, installed editable (this is the up-to-date repo version, not the PyPI one) |
| `~/hydrogym/examples/` | Official examples for every backend |
| `~/venvs/hydrogym_gpu` | Python env with HydroGym, JAX (CUDA), JAX-Fluids, PyTorch (CUDA), Stable-Baselines3. **Use this one.** |
| `~/venvs/hydrogym_cpu` | CPU-only Python env |
| `~/firedrake` | Firedrake (not in `hydrogym_gpu`) |
| `~/JAXFLUIDS` | JAX-Fluids source |
| `~/paraview-install`, `~/paraview-plugins` | ParaView server and the MAIA reader plugin |
| Lmod modules | `MAIA/...`, `Nek5000/...` (`module avail` lists them) |

---

## 2. One-off setup

### 2.1 Install Docker Desktop

1. Install Docker Desktop from docker.com and keep **Use WSL 2** ticked (needs admin rights).
2. Open Docker Desktop and wait for *Engine running*.
3. Check it works, in PowerShell:
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

- **`turing_ampere`** is right for RTX 20/30/40-series (including the RTX 4050) and A100. Use `hopper_blackwell` only for H100 or RTX 50-series.
- It's many GB; the first pull can take a long time. Extracting big layers can look frozen at 100 % for minutes. If interrupted, re-run the same command and it resumes.
- Leave about 30–40 GB free on `C:`.
- Done when it prints `Status: Downloaded newer image for ...`.

### 2.4 Create the container

```powershell
$repo = "C:\path\to\FYP Repo"
docker run -it --gpus all --name hgym --shm-size=4g `
  -p 11111:11111 -p 6006:6006 `
  -v "${repo}:/workspace" -w /workspace `
  clagemann/hydrogym-nvhpc-26.1_cuda-12.9_turing_ampere:latest
```

| Flag | Why |
|---|---|
| `--gpus all` | Passes the GPU through. |
| `--shm-size=4g` | Shared memory for MPI (MAIA couples to Python over MPI). |
| `-p 11111:11111` | ParaView server port (section 6). |
| `-p 6006:6006` | TensorBoard in the Windows browser. |
| `-v "${repo}:/workspace"` | Mounts the repo at `/workspace`. Keep the quotes: the path has spaces. |

- The startup warning *"The NVIDIA Driver was not detected"* is a false alarm under WSL2. `nvidia-smi` working is what counts.
- **Ports and mounts can only be set when the container is created.** To change them, recreate it (`docker rm -f hgym`, then this command again).
- Run `docker run` **once**. Afterwards use `docker start` (section 3). Running it again gives *"name already in use"*.

### 2.5 Configure the shell inside the container

The prompt now looks like `[easybuild@<id> workspace]$`. Run once:

```bash
cat >> ~/.bashrc << 'EOF'
module unload nvhpc 2>/dev/null
module load MAIA/1.0-gompi-2024a-SystemCUDA-CPU
source ~/venvs/hydrogym_gpu/bin/activate
EOF
source ~/.bashrc
```

Why:
- MAIA is provided as an Lmod **module**, not on the PATH. A pre-loaded `nvhpc` module clashes with it, so it's unloaded first.
- The **CPU build** of MAIA is used on the laptop, because the default GPU build crashes on this CPU (section 5.3).
- Python lives in a venv; `hydrogym_gpu` is the one with everything.

Run this block only once. If the prompt shows `(hydrogym_gpu) (hydrogym_gpu)`, the lines were added twice: remove the duplicates with `nano ~/.bashrc`.

### 2.6 Check everything works

```bash
ls /workspace                     # repo files
nvidia-smi                        # RTX 4050 visible
which maia                        # path to the loaded MAIA build
python -c "import hydrogym, jax; print(hydrogym.__file__); print(jax.devices())"
pip list | grep -i -E "hydrogym|jax|torch|stable"
```

Expected: `hydrogym` at `/home/easybuild/hydrogym/...` and `jax.devices()` showing `[CudaDevice(id=0)]`. Lines like `Could not get kernel mode driver version` are harmless under WSL2.

---

## 3. Starting and stopping

### Starting again later

1. Open Docker Desktop and wait for *Engine running*.
2. In PowerShell:
   ```powershell
   docker start hgym
   docker exec -it hgym bash
   ```
   Or attach VS Code (section 4). The `.bashrc` loads MAIA and the venv automatically.
3. If you changed anything elsewhere, `git pull` on Windows first.

### Stopping

1. Commit and push from Windows.
2. In ParaView: *File → Disconnect*.
3. In PowerShell:
   ```powershell
   docker stop hgym
   ```
   This also stops pvserver and any runs inside. **Don't** use `docker rm`: that deletes the container and its `.bashrc` setup.
4. Optional, to free RAM: quit Docker Desktop from the tray, then `wsl --shutdown`.

### Useful commands (PowerShell)

| Task | Command |
|---|---|
| Open a shell in the container | `docker exec -it hgym bash` (run in PowerShell, not inside the container; several at once is fine) |
| Start and attach in one go | `docker start -ai hgym` |
| List containers / status | `docker ps -a` |
| Show published ports | `docker port hgym` |
| Show where the repo is mounted | `docker inspect hgym --format "{{range .Mounts}}{{.Source}} -> {{.Destination}}{{println}}{{end}}"` |
| Run a script without entering | `docker exec -it hgym bash -ic "cd /workspace/Implementation && python my_script.py"` |
| Run in the background | `docker exec -d hgym bash -ic "cd /workspace/Implementation && python train.py > runs/train.log 2>&1"` |

**Am I inside the container?** `PS C:\...>` is Windows. `[easybuild@... ]$` is the container. `ls /.dockerenv` succeeds only inside. `docker` commands only work on Windows.

**Workflow:** `git pull` on Windows → edit in VS Code → run in the container → `git commit` / `git push` on Windows.

---

## 4. VS Code attached to the container

VS Code's normal Run button uses **Windows Python**, which has none of these packages (`ModuleNotFoundError: No module named 'numpy'`). Don't make a Windows venv; attach to the container instead.

1. Install the **Dev Containers** extension (`ms-vscode-remote.remote-containers`, by Microsoft).
2. Make sure `hgym` is running (`docker start hgym`).
3. Click the **`><`** button bottom-left → **Attach to Running Container…** → **hgym**. A new window opens. (Or `Ctrl+Shift+P` → type `attach to running`.)
4. **File → Open Folder** → type `/workspace`. The dialog starts in the container's home folder, so type the path.
5. Install the **Python** extension in that window (**Install in Container**).
6. `Ctrl+Shift+P` → **Python: Select Interpreter** → **Enter interpreter path** → `/home/easybuild/venvs/hydrogym_gpu/bin/python`.

The bottom-left now shows *Container hgym*. Terminals (``Ctrl+` ``, **+** for more), Run and the debugger in this window all run inside the container.

After recreating the container, close this window and attach again.

---

## 5. Running HydroGym examples

HydroGym is installed editable from `~/hydrogym`, so any script in the repo can `import hydrogym`.

### 5.1 JAX, JAX-Fluids, Firedrake: plain Python

```bash
python /workspace/Implementation/jax_toy_problems/my_script.py
```

Official examples: `~/hydrogym/examples/jax/getting_started/`, `~/hydrogym/examples/jaxfluids/`, `~/hydrogym/examples/firedrake/getting_started/`.

Firedrake isn't in `hydrogym_gpu`. Check where it is with `ls ~/firedrake` or `~/venvs/hydrogym_cpu/bin/python -c "import firedrake"`.

### 5.2 MAIA: needs a launcher

MAIA runs as a separate solver process coupled to Python over MPI (MPMD), so a MAIA script **cannot** be run with plain `python` or the Run button. Two steps:

1. Prepare a workspace: downloads the case from Hugging Face and creates `properties_run.toml`.
2. Launch Python and MAIA together from inside that workspace.

**Smoke test (this folder):**

```bash
cd /workspace/Implementation/hydrogym
./run_maia_test.sh                      # Cylinder_2D_Re200, 50 steps, zero actuation
./run_maia_test.sh Cylinder_2D_Re200 200
```

`run_maia_test.sh` loads the CPU MAIA build and the venv, runs `prepare_workspace.py` into `runs/<env>/`, then launches `run_maia_test.py` and MAIA under the right `mpirun`. The Python script prints the reward each step and writes `rewards.csv` into `runs/<env>/`.

Expected output:
```
Environment:       Cylinder_2D_Re200
Observation space: Box(-inf, inf, (80,), float64)
Action space:      Box(-1.0, 1.0, (1,), float64)
Resetting environment
step    0   reward -1.478433   |obs| 6.1617
...
ENV: Finish signal sent to maia
```
With zero actuation the reward oscillates between about −1.32 and −1.52, with a period of about 7 steps. That's the natural vortex shedding (Kármán street) of the uncontrolled cylinder.

**Doing it by hand (for your own scripts):**

```bash
module unload MAIA; module load MAIA/1.0-gompi-2024a-SystemCUDA-CPU
python ~/hydrogym/examples/maia/getting_started/prepare_workspace.py \
  --env Cylinder_2D_Re200 --work-dir ./runs/Cylinder_2D_Re200
cd runs/Cylinder_2D_Re200
export OMP_NUM_THREADS=1
"$EBROOTOPENMPI/bin/mpirun" --bind-to none --mca coll_hcoll_enable 0 \
  -np 1 python /workspace/path/to/my_script.py \
  : -np 1 maia properties_run.toml --silent
```

Things that matter:
- **Call `mpirun` by its full path** (`$EBROOTOPENMPI/bin/mpirun`). The image keeps NVIDIA's MPI permanently on the PATH, and a plain `mpirun` can pick that one, which fails with *"MPI matching the current driver version ... was not installed"*.
- The run file is **`properties_run.toml`**. Older docs say `properties.toml`, which this version doesn't create.
- `--bind-to none --mca coll_hcoll_enable 0` silences harmless laptop warnings about memory binding and missing InfiniBand.
- In this HydroGym version, `maia.from_hf(...)` **must be given `probe_locations`**, otherwise it fails with `object of type 'NoneType' has no len()`. A standard wake grid:
  ```python
  import numpy as np
  import hydrogym.maia as maia

  probe_locations = []
  for x in np.linspace(1.0, 8.0, 8):
      for y in np.linspace(-1.0, 1.0, 5):
          probe_locations.extend([float(x), float(y)])

  env = maia.from_hf("Cylinder_2D_Re200", use_clean_cache=False,
                     probe_locations=probe_locations,
                     obs_normalization_strategy="U_inf")
  ```
  For the cylinder this gives an 80-value observation (u, v at 40 probes) and a 1-value action in [-1, 1].
- Downloaded case data is cached in `~/.cache/huggingface` **inside the container**. The files in `runs/<env>/` are links into that cache, so they break if the container is recreated. Re-running the launcher downloads it again (about 70 MB).

Official MAIA examples: `~/hydrogym/examples/maia/getting_started/` (`test_maia_env.py`, `train_sb3_maia.py`, `run_example_docker.sh`). Environment names: `existing_environments.yaml` in the HydroGym repo.

### 5.3 MAIA builds: use the CPU build on the laptop

| Module | Runs on | On this laptop |
|---|---|---|
| `MAIA/1.0-NVHPC-26.1` (default) | GPU | **Crashes** with `exited on signal 4 (Illegal instruction)` right after `Warming up Device` |
| `MAIA/1.0-gompi-2024a-SystemCUDA-CPU` | CPU | **Works** |

Cause: the GPU build's CPU-side code was compiled with instructions this laptop's processor lacks. The i7-13700H has AVX2 but no AVX-512: `grep -o -w 'avx512[a-z]*' /proc/cpuinfo` prints nothing. The GPU itself is fine.

- **Laptop:** use the CPU build (the default in `.bashrc` and `run_maia_test.sh`). 2D cases are fast enough for development and short training runs. JAX still uses the GPU.
- **HPC:** server CPUs should run the GPU build: `MAIA_MODULE=MAIA ./run_maia_test.sh`, or `module load MAIA`.
- GPU MAIA on the laptop would need a MAIA built for generic x86 (e.g. NVHPC `-tp=haswell`), or MAIA rebuilt inside the container (`~/eb-build.sh`, `~/easyconfigs/`). Worth reporting to the HydroGym team.

### 5.4 Code layout

Keep reusable logic (wrappers, reward shaping, learning rules) in importable modules under `Implementation/agents/`, and keep run scripts thin. Only the environment-creation line then changes between backends. Run outputs go in `runs/` folders, which are git-ignored.

---

## 6. ParaView

The container has a ParaView **server** with the MAIA reader plugin, but no GUI. The GUI runs on Windows and connects to it:

```
ParaView 5.13 on Windows (GUI)  ⇄  localhost:11111  ⇄  pvserver in the container (reads MAIA files)
```

### 6.1 One-off

1. Install **ParaView 5.13** on Windows from paraview.org: `ParaView-5.13.3-Windows-Python3.10-msvc2017-AMD64.msi` (the non-MPI `.msi`; use the `.zip` if you have no admin rights). It must be 5.13 to match the server.
2. The container must have port 11111 published: `docker port hgym` should list `11111/tcp`. If it doesn't, recreate it (section 2.4).
3. In ParaView: **File → Connect… → Add Server**: name `hgym`, type `Client / Server`, host `localhost`, port `11111` → **Configure** → startup type **Manual** → **Save**.
4. Sharper images: **Edit → Settings**, click the **gear** (advanced), **Render View** tab, *Remote/Parallel Rendering Options*. Set **Image Compression** to None (or LZ4), both **Image Reduction Factor**s to `1`, and optionally **Remote Render Threshold** to `1000`. The settings search box finds these quickly.

### 6.2 Each time

1. In a **dedicated** container terminal (keep it separate from the one you run simulations in):
   ```bash
   bash /home/easybuild/paraview-plugins/pvServerLaunch2024.sh 4
   ```
   The number is the number of cores. The script runs `module purge` and starts `pvserver` in the background, so always run it with `bash`, never `source`.
2. In ParaView on Windows: **File → Connect → hgym → Connect**. The Pipeline Browser shows `cs://localhost:11111`.
3. **File → Open** now browses the container. Go **into** `/workspace/Implementation/hydrogym/runs/Cylinder_2D_Re200/out_lb/` and select a file (not the folder):
   - **`restart_.Netcdf`**: the flow field. Open this one.
   - `grid.Netcdf`: the mesh only. The reader finds it automatically.
4. Choose the reader **AIA Reader Parrot Embedded** (m-AIA is MAIA's full name), then **Apply**.
5. Colour by a field from the toolbar dropdown (cross-stream velocity `v` shows the vortex street best) and press **Reset Camera**.
6. Stop the server with `pkill pvserver`, or just `docker stop hgym`.

### 6.3 Making it look better

The blocky squares are the real lattice cells: the 2D cylinder case is deliberately coarse (about 20–25 cells across the cylinder) so RL steps stay cheap. For figures:
- **Smooth colours:** *Filters → Alphabetical → Cell Data to Point Data*, **Apply**, colour the new object.
- **Clean cylinder outline:** open `runs/Cylinder_2D_Re200/stl/cylinder.stl` and colour it solid grey.
- **Vorticity:** after the point-data step, *Filters → Gradient*, tick **Compute Vorticity**, colour by the z-component with a diverging colour map.

Only `restart_.Netcdf` exists unless solution output is turned on in `properties_run.toml`, so there's no animation yet. That file is a link into the cache: copy it before editing.

### 6.4 Without ParaView

For quick plots, use Python on `rewards.csv` or the probe observations:
```python
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("runs/Cylinder_2D_Re200/rewards.csv")
plt.plot(df.step, df.reward); plt.xlabel("step"); plt.ylabel("reward")
plt.savefig("runs/reward.png")
```

---

## 7. Limitations

- **Anything outside `/workspace` is lost** if the container is removed (`docker rm`): `.bashrc` edits, extra `pip install`s and the Hugging Face cache. Re-do section 2.5 after recreating. To keep the cache, add `-v hgym-cache:/home/easybuild/.cache` to `docker run`.
- **Ports and mounts are fixed at creation.**
- **6 GB VRAM:** 2D cases are fine; 3D MAIA LBM or the 3D JAX channel will likely run out of memory. Use the HPC for those.
- **JAX preallocates about 75 % of GPU memory.** When sharing the GPU with MAIA or PyTorch: `export XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- **WSL2 caps RAM** (about half of system RAM by default). Raise it in `C:\Users\<you>\.wslconfig`:
  ```
  [wsl2]
  memory=12GB
  processors=8
  ```
  then `wsl --shutdown` and restart Docker Desktop.
- **Mounted-folder I/O is slower** than the container's own disk. For I/O-heavy runs, write raw output to `~/scratch` and copy what you need into `/workspace`.
- **No GUI windows** from the container. Use ParaView client–server (section 6) or save `.png` plots.
- **Sleep or shutdown kills runs.** Keep the laptop plugged in and disable sleep for long jobs.
- **Run git on Windows,** not in the container.

---

## 8. HPC (Apptainer)

Clusters don't run Docker. Convert the same image:

```bash
apptainer pull docker://clagemann/hydrogym-nvhpc-26.1_cuda-12.9_turing_ampere:latest
apptainer run --nv --bind $(pwd):/workspace hydrogym-nvhpc-26.1_cuda-12.9_turing_ampere_latest.sif
```

- Pick the image matching the cluster's GPUs (`hopper_blackwell` for H100).
- Use the GPU MAIA build there (`module load MAIA`).
- Compute nodes often have no internet, so run `prepare_workspace.py` on a login node first.
- For ParaView on the cluster, start `pvServerLaunch2024.sh` in a job and tunnel the port: `ssh USER@HOST -L 11111:<node>:11111`.

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `ls : Cannot find path 'C:\.dockerenv'` | You're in PowerShell, not the container: `docker start hgym`, then `docker exec -it hgym bash`. |
| `bash: docker: command not found` | You ran a `docker` command inside the container. Run it in PowerShell. |
| `Conflict. The container name "/hgym" is already in use` | It already exists: `docker start hgym`. |
| `Error: No such container: hgym` | It was deleted: redo sections 2.4 and 2.5. |
| `Cannot load module "nvhpc-hpcx-cuda12/26.1" because these module(s) are loaded: nvhpc` | `module unload nvhpc`, then load MAIA. |
| `which maia` finds nothing | No MAIA module loaded in this shell. |
| `python: command not found` | Venv not active: `source ~/venvs/hydrogym_gpu/bin/activate`. |
| `ModuleNotFoundError: No module named 'numpy'` from VS Code | VS Code is using Windows Python: attach to the container (section 4). |
| `exited on signal 4 (Illegal instruction)` after `Warming up Device` | GPU MAIA build incompatible with this CPU: use the CPU build (section 5.3). |
| `mpirun-Error-MPI matching the current driver version ... was not installed` | The NVHPC `mpirun` was picked up: call `"$EBROOTOPENMPI/bin/mpirun"` with the CPU MAIA module loaded. |
| `ConfigError ... object of type 'NoneType' has no len()` | Pass `probe_locations` to `maia.from_hf` (section 5.2). |
| `property file 'properties.toml' not readable` | Use `properties_run.toml`. |
| `failed to bind memory`, `Unable to get list of available IB devices`, `Hcol library init failed` | Harmless on a laptop. |
| mpirun: "not enough slots" | Add `--oversubscribe` after `mpirun`. |
| `bad interpreter: /bin/bash^M` | Windows line endings: `sed -i 's/\r$//' script.sh`. The repo's `.gitattributes` keeps `.sh` files LF. |
| `jax.devices()` shows only `CpuDevice` | Container made without `--gpus all`, or outdated Windows NVIDIA driver. |
| `fatal: detected dubious ownership in repository at '/workspace'` | Run git on Windows, or `git config --global --add safe.directory /workspace`. |
| `git commit` opens Vim | `i`, type the message, `Esc`, `:wq`, Enter. Avoid it with `git commit -m "message"`. |
| Dev Containers command not found in VS Code | Use the `><` button bottom-left, or `Ctrl+Shift+P` → `attach to running`. Check the extension is enabled and reload the window. |
| ParaView: `out_lb` looks empty | Container was recreated, so the links into the cache are broken. Re-run `./run_maia_test.sh`. |
| ParaView asks for a reader for a folder | Go into `out_lb` and select `restart_.Netcdf`; use AIA Reader Parrot Embedded. |
| ParaView image blurry | Turn off image compression and set reduction factors to 1 (section 6.1). |
| ParaView image blocky | That's the real grid: use Cell Data to Point Data (section 6.3). |
| Image pull seems stuck | Wait out extraction; if nothing changes for 15+ min, Ctrl+C, restart Docker Desktop, pull again. |

---

## References

- HydroGym docs: <https://dynamicslab.github.io/hydrogym/docs/quickstart>
- MAIA backend: <https://dynamicslab.github.io/hydrogym/docs/installation/maia>
- MAIA getting-started examples: <https://dynamicslab.github.io/hydrogym/docs/examples/maia/getting_started>
- JAX backend: <https://dynamicslab.github.io/hydrogym/docs/installation/jax>
- HydroGym source: <https://github.com/dynamicslab/hydrogym>
