"""Minimal MAIA smoke test for HydroGym.

Runs a HydroGym MAIA environment with zero (uncontrolled) actuation for a few
steps, prints the reward each step, and saves them to a CSV.

MAIA runs as a separate solver process, so this script must be launched under
mpirun alongside it. Use run_maia_test.sh, which does this for you:

    ./run_maia_test.sh                      # Cylinder_2D_Re200, 50 steps
    ./run_maia_test.sh Cylinder_2D_Re200 200

Under the hood (from inside the prepared workspace folder):

    mpirun -np 1 python run_maia_test.py --env Cylinder_2D_Re200 : -np 1 maia properties.toml
"""

import argparse
import csv
import time

import numpy as np
import hydrogym.maia as maia


def main():
    parser = argparse.ArgumentParser(description="HydroGym MAIA smoke test")
    parser.add_argument("--env", default="Cylinder_2D_Re200", help="HydroGym MAIA environment name")
    parser.add_argument("--steps", type=int, default=50, help="number of control steps to run")
    parser.add_argument("--out", default="rewards.csv", help="CSV file for step rewards")
    args = parser.parse_args()

    # Create the environment (uses the workspace in the current folder / HF cache)
    env = maia.from_hf(args.env, use_clean_cache=False)
    print(f"Environment:       {args.env}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space:      {env.action_space}")

    # Gymnasium returns (obs, info); older Gym returns obs only
    result = env.reset()
    obs = result[0] if isinstance(result, tuple) else result

    # Zero actuation = uncontrolled baseline flow
    action = np.zeros(env.action_space.shape, dtype=np.float32)

    rows = []
    t0 = time.time()
    try:
        for k in range(args.steps):
            out = env.step(action)
            if len(out) == 5:  # Gymnasium: obs, reward, terminated, truncated, info
                obs, reward, terminated, truncated, info = out
                done = terminated or truncated
            else:              # old Gym: obs, reward, done, info
                obs, reward, done, info = out

            rows.append((k, float(reward)))
            print(f"step {k:4d}   reward {float(reward): .6f}   |obs| {np.linalg.norm(obs):.4f}")

            if done:
                print("Episode ended.")
                break
    finally:
        env.close()  # shuts down the coupled MAIA process cleanly

    elapsed = time.time() - t0
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "reward"])
        writer.writerows(rows)

    print(f"\nRan {len(rows)} steps in {elapsed:.1f} s "
          f"({elapsed / max(len(rows), 1):.2f} s/step). Rewards saved to {args.out}")


if __name__ == "__main__":
    main()