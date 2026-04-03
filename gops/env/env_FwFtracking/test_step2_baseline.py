#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
STEP 2: Baseline sanity check (NO RL)

Purpose:
- Verify allocator + prelook + forward kinematics
- Check steer/speed continuity
- Check chassis trajectory reasonableness

Policy:
- action = [bdx, bdy, 0.0]   (follow ball velocity)
"""

import numpy as np
import matplotlib.pyplot as plt

# ====== import your env ======
from gops.create_pkg.create_env import create_env

# =============================
# Config
# =============================
DT = 0.05
EPISODE_STEPS = 1500
ENV_ID = "env_Forsee_A"   # 确保你已经注册了这个 env

# =============================
# Baseline policy
# =============================
def baseline_policy(env):
    """
    Follow ball velocity directly.
    """
    bx, by, bdx, bdy = env.context.state.reference
    return np.array([bdx, bdy, 0.0], dtype=np.float32)


# =============================
# Main
# =============================
def main():
    env = create_env(
        env_id=ENV_ID,
        dt=DT,
        episode_steps=EPISODE_STEPS,
        lookahead=1,          # 可以先用 8
        enable_cuda=False,
    )

    obs, _ = env.reset()
    base_env = env.unwrapped

    # ====== logs ======
    steer_hist = {k: [] for k in base_env.robot._steer.keys()}
    speed_hist = {k: [] for k in base_env.robot._speed.keys()}
    vx_hist, vy_hist, w_hist = [], [], []
    px_hist, py_hist = [], []
    bx_hist, by_hist = [], []

    for step in range(EPISODE_STEPS):
        action = baseline_policy(base_env)

        step_out = env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, _ = step_out
            done = terminated or truncated
        else:
            obs, reward, done, _ = step_out

        # --- robot state ---
        px, py, yaw = base_env.robot.state
        vx, vy = base_env.robot.vel_world
        w = base_env.robot.w_body

        # --- ball state ---
        bx, by, _, _ = base_env.context.state.reference

        # --- record ---
        for name in steer_hist:
            steer_hist[name].append(base_env.robot._steer[name])
            speed_hist[name].append(base_env.robot._speed[name])

        vx_hist.append(vx)
        vy_hist.append(vy)
        w_hist.append(w)
        px_hist.append(px)
        py_hist.append(py)
        bx_hist.append(bx)
        by_hist.append(by)

        if done:
            print(f"Episode terminated at step {step}")
            break

    # =============================
    # Plot
    # =============================
    t = np.arange(len(px_hist)) * DT

    plt.figure(figsize=(14, 10))

    # ---- Trajectory ----
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(px_hist, py_hist, label="robot")
    ax1.plot(bx_hist, by_hist, "--", label="ball")
    ax1.set_aspect("equal")
    ax1.set_title("Trajectory")
    ax1.grid(True)
    ax1.legend()

    # ---- Chassis velocity ----
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(t, vx_hist, label="vx")
    ax2.plot(t, vy_hist, label="vy")
    ax2.plot(t, w_hist, label="w")
    ax2.set_title("Chassis velocity")
    ax2.grid(True)
    ax2.legend()

    # ---- Steer angles ----
    ax3 = plt.subplot(2, 2, 3)
    for name, data in steer_hist.items():
        ax3.plot(t, data, label=name)
    ax3.set_title("Steer angles")
    ax3.grid(True)
    ax3.legend()

    # ---- Wheel speeds ----
    ax4 = plt.subplot(2, 2, 4)
    for name, data in speed_hist.items():
        ax4.plot(t, data, label=name)
    ax4.set_title("Wheel speeds")
    ax4.grid(True)
    ax4.legend()


    plt.tight_layout()
    plt.savefig("step3_baseline_result.png", dpi=150)
    print("Saved figure to step3_baseline_result.png")
    plt.close()


    print("STEP 3 baseline test finished.")


if __name__ == "__main__":
    main()
