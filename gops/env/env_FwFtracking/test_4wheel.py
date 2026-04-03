from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

from gops.env.env_FwFtracking.env_ToRwheelsim import (
    STEER_LIMIT,
    WHEEL_ORDER,
    WHEEL_POS,
    ToRwheelsimRobot,
    wrap_to_pi,
)


DT = 0.05
CASE_DURATION = 6.0
STEPS = int(CASE_DURATION / DT)

V_WHEEL_MAX = 0.4
V_CASE = 0.3
W_CASE = 1.2

CHASSIS_LEN = 0.48
CHASSIS_WID = 0.35
WHEEL_LEN = 0.10
WHEEL_WID = 0.04


def action_all_same(steer: float, speed: float) -> np.ndarray:
    action = np.zeros(8, dtype=np.float32)
    for i in range(4):
        action[2 * i] = steer
        action[2 * i + 1] = speed
    return action


def action_from_body(vx: float, vy: float, w: float) -> np.ndarray:
    action = np.zeros(8, dtype=np.float32)
    for i, name in enumerate(WHEEL_ORDER):
        x_i, y_i = WHEEL_POS[name]
        vix = vx - w * y_i
        viy = vy + w * x_i
        steer = np.arctan2(viy, vix)
        speed = float(np.hypot(vix, viy))
        speed = np.clip(speed, -V_WHEEL_MAX, V_WHEEL_MAX)
        action[2 * i] = steer
        action[2 * i + 1] = speed
    return action


def action_front_back_opposed(speed: float) -> np.ndarray:
    action = np.zeros(8, dtype=np.float32)
    for i, name in enumerate(WHEEL_ORDER):
        steer = 0.0
        if name in ("FR", "FL"):
            spd = speed
        else:
            spd = -speed
        action[2 * i] = steer
        action[2 * i + 1] = spd
    return action


def simulate_case(
    name: str,
    action_fn: Callable[[float], np.ndarray],
    steps: int,
) -> Dict[str, np.ndarray]:
    robot = ToRwheelsimRobot(
        dt=DT,
        v_wheel_max=V_WHEEL_MAX,
        steer_rate_max=None,
        w_max=2.0,
        v_max=0.8,
        a_max=1.0,
    )
    robot.reset(np.array([0.0, 0.0, 0.0], dtype=np.float32))

    states = np.zeros((steps, 3), dtype=np.float32)
    wheel_steer = np.zeros((steps, 4), dtype=np.float32)
    for k in range(steps):
        action = action_fn(k * DT)
        robot.step(action)
        states[k] = robot.state
        for j, name in enumerate(WHEEL_ORDER):
            wheel_steer[k, j] = robot._prev_steer[name]
    return {"name": name, "states": states, "wheel_steer": wheel_steer}


def chassis_polygon(px: float, py: float, yaw: float) -> np.ndarray:
    half_l = CHASSIS_LEN / 2.0
    half_w = CHASSIS_WID / 2.0
    body = np.array(
        [
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ],
        dtype=np.float32,
    )
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    world = (rot @ body.T).T + np.array([px, py])
    return world


def wheel_polygon(
    px: float,
    py: float,
    yaw: float,
    wheel_name: str,
    steer: float,
) -> np.ndarray:
    half_l = WHEEL_LEN / 2.0
    half_w = WHEEL_WID / 2.0
    wheel = np.array(
        [
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ],
        dtype=np.float32,
    )
    wx, wy = WHEEL_POS[wheel_name]
    c, s = np.cos(yaw), np.sin(yaw)
    rot_body = np.array([[c, -s], [s, c]], dtype=np.float32)
    center = rot_body @ np.array([wx, wy], dtype=np.float32) + np.array([px, py])
    cs, ss = np.cos(yaw + steer), np.sin(yaw + steer)
    rot_wheel = np.array([[cs, -ss], [ss, cs]], dtype=np.float32)
    world = (rot_wheel @ wheel.T).T + center
    return world


def build_cases() -> List[Dict[str, np.ndarray]]:
    cases = [
        ("all_forward", lambda t: action_all_same(0.0, V_CASE)),
        ("all_reverse", lambda t: action_all_same(0.0, -V_CASE)),
        ("crab_left", lambda t: action_all_same(np.pi / 2.0, V_CASE)),
        ("crab_right", lambda t: action_all_same(-np.pi / 2.0, V_CASE)),
        ("spin_ccw", lambda t: action_from_body(0.0, 0.0, W_CASE)),
        ("front_back_opposed", lambda t: action_front_back_opposed(V_CASE)),
    ]
    return [simulate_case(name, fn, STEPS) for name, fn in cases]


def main() -> None:
    cases = build_cases()

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    lines = []
    robots = []
    headings = []
    wheels = []

    lim = 2.5
    for ax, case in zip(axes, cases):
        ax.set_aspect("equal")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_title(case["name"])
        line, = ax.plot([], [], "b-", lw=1.5)
        poly = Polygon(np.zeros((4, 2)), closed=True, fill=False, edgecolor="k", lw=2)
        ax.add_patch(poly)
        head_line, = ax.plot([], [], "r-", lw=2)
        lines.append(line)
        robots.append(poly)
        headings.append(head_line)
        wheel_patches = []
        for _ in WHEEL_ORDER:
            wheel_poly = Polygon(
                np.zeros((4, 2)),
                closed=True,
                fill=True,
                facecolor="lightgray",
                edgecolor="dimgray",
                lw=1.2,
            )
            ax.add_patch(wheel_poly)
            wheel_patches.append(wheel_poly)
        wheels.append(wheel_patches)

    def update(frame: int):
        artists = []
        for idx, case in enumerate(cases):
            states = case["states"]
            steer_hist = case["wheel_steer"]
            frame_idx = min(frame, len(states) - 1)
            path = states[: frame_idx + 1]
            px, py, yaw = states[frame_idx]
            steer_now = steer_hist[frame_idx]

            lines[idx].set_data(path[:, 0], path[:, 1])
            robots[idx].set_xy(chassis_polygon(px, py, yaw))

            head_len = 0.3
            hx = px + head_len * np.cos(yaw)
            hy = py + head_len * np.sin(yaw)
            headings[idx].set_data([px, hx], [py, hy])

            for j, name in enumerate(WHEEL_ORDER):
                wheels[idx][j].set_xy(wheel_polygon(px, py, yaw, name, steer_now[j]))

            artists.extend([lines[idx], robots[idx], headings[idx]] + wheels[idx])
        return artists

    frames = STEPS
    _ani = FuncAnimation(fig, update, frames=frames, interval=DT * 1000, blit=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
