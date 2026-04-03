#!/usr/bin/env python
#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Description: Evaluation script for Forsee_A tracking policy.

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

from gops.create_pkg.create_alg import create_alg
from gops.create_pkg.create_env import create_env
from gops.utils.init_args import init_args


@dataclass
class EvalConfig:
    # Paths and experiment setup
    checkpoint_path: str = (
        "results/env_Forsee_A/DSACT_xxxxxx/apprfunc/apprfunc_xxxxxx_opt.pkl"
    )
    save_dir: str = "results/Forsee_A/eval"
    env_id: str = "env_Forsee_A"
    algorithm: str = "DSACT"
    seed: int = 12345

    # Evaluation options
    num_episodes: int = 5
    render: bool = False
    save_fig: bool = True
    save_anim: bool = True
    anim_fps: int = 30
    anim_format: str = "mp4"  # "mp4" or "gif"

    # Environment parameters (optional overrides)
    dt: float = 0.05
    t_total: float = 30.0
    follow_dist: float = 0.3
    v_wheel_max: float = 0.3
    w_max: float = 1.5
    v_max: float = 0.3
    a_max: float = 0.3
    steer_rate_max: float = None
    wheel_acc_max: float = None
    lookahead: int = 8


def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def extract_env_ref(env) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    # Unwrap GOPS env to access reference data.
    base_env = getattr(env, "unwrapped", env)
    robot_state = base_env.robot.state
    context_state = base_env.context.state
    rx, ry, yaw_ref = base_env._get_ref_target()
    return robot_state, context_state.reference, (rx, ry, yaw_ref)


def get_velocity(env) -> Tuple[float, float]:
    base_env = getattr(env, "unwrapped", env)
    vx, vy = base_env.robot.vel_world
    return float(vx), float(vy)


def get_w_body(env) -> float:
    base_env = getattr(env, "unwrapped", env)
    return float(base_env.robot.w_body)


def get_wheel_state(env) -> Tuple[np.ndarray, np.ndarray]:
    base_env = getattr(env, "unwrapped", env)
    steer = np.array([base_env.robot._steer[name] for name in WHEEL_POS.keys()], dtype=np.float32)
    speed = np.array([base_env.robot._speed[name] for name in WHEEL_POS.keys()], dtype=np.float32)
    return steer, speed


def policy_act_deterministic(networks, obs: np.ndarray) -> np.ndarray:
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = networks.policy(obs_t)
        act_dist = networks.create_action_distributions(logits)
        if hasattr(act_dist, "mode"):
            act = act_dist.mode()
        else:
            act = act_dist.mean
    return act.squeeze(0).cpu().numpy()


# Visualization geometry (match env settings)
CHASSIS_L = 0.48
CHASSIS_W = 0.35
WHEEL_POS = {
    "FR": (+0.24, -0.175),
    "RR": (-0.24, -0.175),
    "RL": (-0.24, +0.175),
    "FL": (+0.24, +0.175),
}
WHEEL_LEN = 0.10
WHEEL_WID = 0.04
WHEEL_RADIUS = 0.06


def evaluate_policy(config: EvalConfig) -> Dict[str, np.ndarray]:
    ensure_dir(config.save_dir)
    print(
        f"Eval start: episodes={config.num_episodes}, render={config.render}, "
        f"save_anim={config.save_anim}, checkpoint={config.checkpoint_path}"
    )

    args = {
        "env_id": config.env_id,
        "algorithm": config.algorithm,
        "seed": config.seed,
        "trainer": "off_serial_trainer",
        "sampler_name": "off_sampler",
        "buffer_name": "replay_buffer",
        "sample_batch_size": 20,
        "replay_batch_size": 256,
        "buffer_warm_size": 10000,
        "buffer_max_size": 200000,
        "sample_interval": 1,
        "save_folder": None,
        "log_save_interval": 10000,
        "apprfunc_save_interval": 50000,
        "dt": config.dt,
        "t_total": config.t_total,
        "follow_dist": config.follow_dist,
        "v_wheel_max": config.v_wheel_max,
        "w_max": config.w_max,
        "v_max": config.v_max,
        "a_max": config.a_max,
        "steer_rate_max": config.steer_rate_max,
        "wheel_acc_max": config.wheel_acc_max,
        "lookahead": config.lookahead,
        "enable_cuda": False,
    }
    # Merge training config if available (for apprfunc settings)
    result_dir = os.path.dirname(os.path.dirname(config.checkpoint_path))
    config_path = os.path.join(result_dir, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            train_args = json.load(f)
        # Keep eval overrides, but fill missing keys from training config
        for k, v in train_args.items():
            if k not in args:
                args[k] = v

    env = create_env(**args)
    args = init_args(env, **args)
    alg = create_alg(**args)

    # Load checkpoint
    if not os.path.isfile(config.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {config.checkpoint_path}")
    state_dict = torch.load(config.checkpoint_path, map_location="cpu")
    alg.load_state_dict(state_dict)

    # Set policy to deterministic / eval mode
    alg.eval()

    results = {
        "episode_return": [],
        "mean_pos_err": [],
        "max_pos_err": [],
        "rms_pos_err": [],
    }

    for ep in range(config.num_episodes):
        print(f"Episode {ep + 1}/{config.num_episodes} reset...")
        obs, _ = env.reset(seed=config.seed + ep)
        done = False
        ep_return = 0.0

        time_hist: List[float] = []
        pose_hist: List[Tuple[float, float, float]] = []
        vel_hist: List[Tuple[float, float]] = []
        acc_hist: List[Tuple[float, float]] = []
        yaw_hist: List[float] = []
        yaw_rate_hist: List[float] = []
        yaw_acc_hist: List[float] = []
        action_hist: List[Tuple[float, float, float]] = []

        pos_err_hist: List[float] = []
        vel_err_hist: List[float] = []
        yaw_err_hist: List[float] = []
        reward_hist: List[float] = []

        traj_robot: List[Tuple[float, float]] = []
        traj_ref: List[Tuple[float, float]] = []
        traj_ball: List[Tuple[float, float]] = []

        steer_angle_hist: List[np.ndarray] = []
        steer_rate_hist: List[np.ndarray] = []
        wheel_speed_hist: List[np.ndarray] = []
        wheel_acc_hist: List[np.ndarray] = []

        step = 0
        prev_vel = None
        prev_w = None
        prev_steer = None
        prev_speed = None
        while not done:
            action = policy_act_deterministic(alg.networks, obs)
            step_out = env.step(action)
            if len(step_out) == 5:
                obs, reward, terminated, truncated, _ = step_out
                done = terminated or truncated
            else:
                obs, reward, done, _ = step_out

            robot_state, ref, ref_target = extract_env_ref(env)
            px, py, yaw = robot_state
            bx, by, bdx, bdy = ref
            rx, ry, yaw_ref = ref_target
            vx, vy = get_velocity(env)
            w_body = get_w_body(env)
            steer, speed = get_wheel_state(env)

            pos_err = np.hypot(px - rx, py - ry)
            vel_err = np.hypot(vx - bdx, vy - bdy)
            yaw_err = wrap_to_pi(yaw - yaw_ref)

            if prev_vel is None:
                ax, ay = 0.0, 0.0
            else:
                ax = (vx - prev_vel[0]) / config.dt
                ay = (vy - prev_vel[1]) / config.dt
            if prev_w is None:
                w_acc = 0.0
            else:
                w_acc = (w_body - prev_w) / config.dt

            if prev_steer is None:
                steer_rate = np.zeros_like(steer)
            else:
                steer_rate = wrap_to_pi(steer - prev_steer) / config.dt
            if prev_speed is None:
                wheel_acc = np.zeros_like(speed)
            else:
                wheel_acc = (speed - prev_speed) / config.dt

            prev_vel = (vx, vy)
            prev_w = w_body
            prev_steer = steer.copy()
            prev_speed = speed.copy()

            traj_robot.append((px, py))
            traj_ref.append((rx, ry))
            traj_ball.append((bx, by))
            steer_angle_hist.append(steer)
            steer_rate_hist.append(steer_rate)
            wheel_speed_hist.append(speed)
            wheel_acc_hist.append(wheel_acc)

            pos_err_hist.append(pos_err)
            vel_err_hist.append(vel_err)
            yaw_err_hist.append(yaw_err)
            reward_hist.append(float(reward))
            time_hist.append(step * config.dt)
            pose_hist.append((px, py, yaw))
            vel_hist.append((vx, vy))
            acc_hist.append((ax, ay))
            yaw_hist.append(yaw)
            yaw_rate_hist.append(w_body)
            yaw_acc_hist.append(w_acc)
            action_hist.append((float(action[0]), float(action[1]), float(action[2])))

            ep_return += float(reward)
            step += 1
            if step % 200 == 0:
                print(f"Episode {ep + 1}: step {step}, return {ep_return:.2f}")

            if config.render:
                env.render()

        pos_err_arr = np.array(pos_err_hist, dtype=np.float32)
        rms_pos = float(np.sqrt(np.mean(pos_err_arr ** 2)))
        results["episode_return"].append(ep_return)
        results["mean_pos_err"].append(float(np.mean(pos_err_arr)))
        results["max_pos_err"].append(float(np.max(pos_err_arr)))
        results["rms_pos_err"].append(rms_pos)
        print(
            f"Episode {ep + 1} done: steps={step}, return={ep_return:.2f}, "
            f"mean_pos_err={np.mean(pos_err_arr):.4f}"
        )

        if config.save_fig:
            plot_episode(
                config.save_dir,
                ep,
                np.array(traj_robot),
                np.array(traj_ref),
                time_hist,
                np.array(vel_hist),
                np.array(acc_hist),
                np.array(yaw_hist),
                np.array(yaw_rate_hist),
                np.array(yaw_acc_hist),
                np.array(action_hist),
                pos_err_hist,
                vel_err_hist,
                yaw_err_hist,
                steer_angle_hist,
                steer_rate_hist,
                wheel_speed_hist,
                wheel_acc_hist,
                reward_hist,
            )
        if config.save_anim:
            save_episode_animation(
                config.save_dir,
                ep,
                time_hist,
                pose_hist,
                vel_hist,
                traj_ref,
                steer_angle_hist,
                wheel_speed_hist,
                config.anim_fps,
                config.anim_format,
                config.dt,
            )

    summarize_results(results)
    return results


def plot_episode(
    save_dir: str,
    ep: int,
    traj_robot: np.ndarray,
    traj_ref: np.ndarray,
    time_hist: List[float],
    vel_hist: np.ndarray,
    acc_hist: np.ndarray,
    yaw_hist: np.ndarray,
    yaw_rate_hist: np.ndarray,
    yaw_acc_hist: np.ndarray,
    action_hist: np.ndarray,
    pos_err_hist: List[float],
    vel_err_hist: List[float],
    yaw_err_hist: List[float],
    steer_angle_hist: List[np.ndarray],
    steer_rate_hist: List[np.ndarray],
    wheel_speed_hist: List[np.ndarray],
    wheel_acc_hist: List[np.ndarray],
    reward_hist: List[float],
) -> None:
    import matplotlib.pyplot as plt

    steer_angle_hist = np.array(steer_angle_hist)
    steer_rate_hist = np.array(steer_rate_hist)
    wheel_speed_hist = np.array(wheel_speed_hist)
    wheel_acc_hist = np.array(wheel_acc_hist)

    fig = plt.figure(figsize=(12, 9))
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(traj_robot[:, 0], traj_robot[:, 1], label="robot")
    ax1.plot(traj_ref[:, 0], traj_ref[:, 1], label="reference")
    ax1.set_title("Trajectory")
    ax1.set_aspect("equal")
    ax1.grid(True)
    ax1.legend()

    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(pos_err_hist, label="pos error")
    ax2.plot(vel_err_hist, label="vel error")
    ax2.plot(yaw_err_hist, label="yaw error")
    ax2.set_title("Tracking Errors")
    ax2.grid(True)
    ax2.legend()

    ax3 = fig.add_subplot(3, 2, 3)
    ax3.plot(time_hist, vel_hist[:, 0], label="vx")
    ax3.plot(time_hist, vel_hist[:, 1], label="vy")
    ax3.set_title("Chassis Velocity (world)")
    ax3.grid(True)
    ax3.legend()

    ax4 = fig.add_subplot(3, 2, 4)
    ax4.plot(time_hist, acc_hist[:, 0], label="ax")
    ax4.plot(time_hist, acc_hist[:, 1], label="ay")
    ax4.set_title("Chassis Acceleration (world)")
    ax4.grid(True)
    ax4.legend()

    ax5 = fig.add_subplot(3, 2, 5)
    ax5.plot(time_hist, yaw_hist, label="yaw")
    ax5.plot(time_hist, yaw_rate_hist, label="yaw rate")
    ax5.set_title("Yaw and Yaw Rate")
    ax5.grid(True)
    ax5.legend()

    ax6 = fig.add_subplot(3, 2, 6)
    ax6.plot(time_hist, reward_hist, label="reward")
    ax6.set_title("Reward")
    ax6.grid(True)
    ax6.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"episode_{ep:03d}_overview.png"))
    plt.close(fig)

    fig2 = plt.figure(figsize=(12, 9))
    ax1 = fig2.add_subplot(3, 2, 1)
    ax1.plot(time_hist, action_hist[:, 0], label="vx_des")
    ax1.plot(time_hist, action_hist[:, 1], label="vy_des")
    ax1.plot(time_hist, action_hist[:, 2], label="w_des")
    ax1.set_title("Action Commands")
    ax1.grid(True)
    ax1.legend()

    ax2 = fig2.add_subplot(3, 2, 2)
    ax2.plot(time_hist, yaw_acc_hist, label="yaw acc")
    ax2.set_title("Yaw Acceleration")
    ax2.grid(True)
    ax2.legend()

    ax3 = fig2.add_subplot(3, 2, 3)
    ax3.plot(steer_angle_hist, label=["FR", "RR", "RL", "FL"])
    ax3.set_title("Steer Angles")
    ax3.grid(True)
    ax3.legend()

    ax4 = fig2.add_subplot(3, 2, 4)
    ax4.plot(steer_rate_hist, label=["FR", "RR", "RL", "FL"])
    ax4.set_title("Steer Rates")
    ax4.grid(True)
    ax4.legend()

    ax5 = fig2.add_subplot(3, 2, 5)
    ax5.plot(wheel_speed_hist, label=["FR", "RR", "RL", "FL"])
    ax5.set_title("Wheel Speeds")
    ax5.grid(True)
    ax5.legend()

    ax6 = fig2.add_subplot(3, 2, 6)
    ax6.plot(wheel_acc_hist, label=["FR", "RR", "RL", "FL"])
    ax6.set_title("Wheel Accelerations")
    ax6.grid(True)
    ax6.legend()

    fig2.tight_layout()
    fig2.savefig(os.path.join(save_dir, f"episode_{ep:03d}_wheels.png"))
    plt.close(fig2)


def save_episode_animation(
    save_dir: str,
    ep: int,
    time_hist: List[float],
    pose_hist: List[Tuple[float, float, float]],
    vel_hist: List[Tuple[float, float]],
    traj_ref: List[Tuple[float, float]],
    steer_hist: List[np.ndarray],
    speed_hist: List[np.ndarray],
    fps: int,
    anim_format: str,
    dt: float,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.patches import Circle, Polygon

    pose = np.array(pose_hist, dtype=np.float32)
    vel = np.array(vel_hist, dtype=np.float32)
    ref = np.array(traj_ref, dtype=np.float32)
    steer_hist = np.array(steer_hist, dtype=np.float32)
    speed_hist = np.array(speed_hist, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.grid(True)
    ax.plot(ref[:, 0], ref[:, 1], color="tab:orange", linewidth=1.0, label="reference")
    robot_line, = ax.plot([], [], color="tab:blue", linewidth=1.5, label="robot path")
    ball_line, = ax.plot([], [], color="tab:red", linewidth=1.0, label="ball path")
    ball_pt = Circle((0.0, 0.0), radius=0.05, color="tab:red")
    ax.add_patch(ball_pt)
    chassis = Polygon(np.zeros((4, 2)), closed=True, fill=False, edgecolor="k", lw=2)
    ax.add_patch(chassis)
    heading_line, = ax.plot([], [], "k-", lw=2, label="heading")
    vel_line, = ax.plot([], [], "g-", lw=2, label="velocity")
    wheel_patches = {}
    wheel_spokes = {}
    for name in WHEEL_POS.keys():
        poly = Polygon(
            np.zeros((4, 2)),
            closed=True,
            fill=True,
            facecolor="lightgray",
            edgecolor="darkgray",
            lw=1.5,
        )
        wheel_patches[name] = poly
        ax.add_patch(poly)
        spk, = ax.plot([], [], color="orange", lw=2)
        wheel_spokes[name] = spk
    ax.legend(loc="upper right")

    xmin = min(np.min(pose[:, 0]), np.min(ref[:, 0])) - 0.8
    xmax = max(np.max(pose[:, 0]), np.max(ref[:, 0])) + 0.8
    ymin = min(np.min(pose[:, 1]), np.min(ref[:, 1])) - 0.8
    ymax = max(np.max(pose[:, 1]), np.max(ref[:, 1])) + 0.8
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    path_x: List[float] = []
    path_y: List[float] = []
    ball_x: List[float] = []
    ball_y: List[float] = []
    wheel_phi = {name: 0.0 for name in WHEEL_POS.keys()}
    time_text = ax.text(
        0.02,
        0.96,
        "",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
    )

    def init():
        robot_line.set_data([], [])
        ball_line.set_data([], [])
        heading_line.set_data([], [])
        vel_line.set_data([], [])
        time_text.set_text("")
        return (
            [robot_line, ball_line, heading_line, vel_line, time_text, ball_pt, chassis]
            + list(wheel_patches.values())
            + list(wheel_spokes.values())
        )

    def update(i):
        px, py, yaw = pose[i]
        vx, vy = vel[i]
        bx, by = ref[i]

        path_x.append(px)
        path_y.append(py)
        ball_x.append(bx)
        ball_y.append(by)
        robot_line.set_data(path_x, path_y)
        ball_line.set_data(ball_x, ball_y)
        ball_pt.center = (bx, by)

        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, -s], [s, c]])
        chassis_local = np.array(
            [
                [CHASSIS_L / 2, CHASSIS_W / 2],
                [CHASSIS_L / 2, -CHASSIS_W / 2],
                [-CHASSIS_L / 2, -CHASSIS_W / 2],
                [-CHASSIS_L / 2, CHASSIS_W / 2],
            ]
        )
        chassis_world = (rot @ chassis_local.T).T + np.array([px, py])
        chassis.set_xy(chassis_world)

        head = np.array([0.25 * np.cos(yaw), 0.25 * np.sin(yaw)])
        heading_line.set_data([px, px + head[0]], [py, py + head[1]])

        vel_line.set_data([px, px + 0.6 * vx], [py, py + 0.6 * vy])

        for idx, name in enumerate(WHEEL_POS.keys()):
            steer = steer_hist[i][idx]
            speed = speed_hist[i][idx]
            wx, wy = WHEEL_POS[name]
            center = rot @ np.array([wx, wy]) + np.array([px, py])

            wheel_shape = np.array(
                [
                    [WHEEL_LEN / 2, WHEEL_WID / 2],
                    [WHEEL_LEN / 2, -WHEEL_WID / 2],
                    [-WHEEL_LEN / 2, -WHEEL_WID / 2],
                    [-WHEEL_LEN / 2, WHEEL_WID / 2],
                ]
            )
            wheel_rot = rot @ np.array(
                [[np.cos(steer), -np.sin(steer)], [np.sin(steer), np.cos(steer)]]
            )
            wheel_world = (wheel_rot @ wheel_shape.T).T + center
            wheel_patches[name].set_xy(wheel_world)

            wheel_phi[name] = wrap_to_pi(
                wheel_phi[name] + (speed / max(WHEEL_RADIUS, 1e-6)) * dt
            )
            phi = wheel_phi[name]
            spoke_len = WHEEL_WID * 0.9
            spoke_local = np.array([0.0, spoke_len / 2])
            spoke_rot = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
            spoke_local = spoke_rot @ spoke_local
            p1 = center - wheel_rot @ spoke_local
            p2 = center + wheel_rot @ spoke_local
            wheel_spokes[name].set_data([p1[0], p2[0]], [p1[1], p2[1]])

        if i < len(time_hist):
            time_text.set_text(f"t={time_hist[i]:.2f} s")
        return (
            [robot_line, ball_line, heading_line, vel_line, time_text, ball_pt, chassis]
            + list(wheel_patches.values())
            + list(wheel_spokes.values())
        )

    ani = animation.FuncAnimation(
        fig, update, frames=len(pose), init_func=init, interval=1000 / fps, blit=False
    )

    use_ffmpeg = animation.writers.is_available("ffmpeg")
    if anim_format == "mp4" and not use_ffmpeg:
        anim_format = "gif"
    anim_path = os.path.join(save_dir, f"episode_{ep:03d}.{anim_format}")
    if anim_format == "gif":
        ani.save(anim_path, writer="pillow", fps=fps)
    else:
        ani.save(anim_path, writer="ffmpeg", fps=fps)
    plt.close(fig)


def summarize_results(results: Dict[str, List[float]]) -> None:
    def _stat(name: str) -> None:
        arr = np.array(results[name], dtype=np.float32)
        print(f"{name}: mean={np.mean(arr):.4f}, std={np.std(arr):.4f}")

    print("Evaluation summary:")
    _stat("episode_return")
    _stat("mean_pos_err")
    _stat("max_pos_err")
    _stat("rms_pos_err")


if __name__ == "__main__":
    cfg = EvalConfig()
    evaluate_policy(cfg)
