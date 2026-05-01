#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Description: Evaluation script for ToRwheelsim tracking policy.

import json
import os
import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from gops.create_pkg.create_alg import create_alg
from gops.create_pkg.create_env import create_env


@dataclass
class EvalConfig:
    # Paths and experiment setup
    checkpoint_path: str = (
        "results/RPL_FWTsim_chunk_histstack001_gamma2_clean_v1/apprfunc/apprfunc_1485000_opt.pkl"
    )
    save_dir: str = "results/RPL_FWTsim_chunk_histstack001_gamma2_clean_v1/eval_apprfunc_1485000_opt"
    env_id: Optional[str] = None
    algorithm: str = "DSACT"
    traj_type: str = "rounded_rect"
    seed: int = 12345
    ref_time: float = None
    episode_steps: int = 1500

    # Baseline (MPC) options
    run_baseline: bool = False
    baseline_script: str = "gops/env/env_FwFtracking/_ToRwheeLssim.py"
    baseline_save_dir: str = "results/RPL_FWTsim_chunk_histstack001_gamma2_clean_v1/eval_apprfunc_1485000_opt_baseline_mpc"
    baseline_side: float = -1.0

    # Evaluation options
    num_episodes: int = 3
    render: bool = True
    save_fig: bool = True
    save_anim: bool = True
    anim_fps: int = 30
    anim_format: str = "gif"  # "mp4" or "gif"
    open_dir: bool = False

    # Environment parameters (optional overrides)
    dt: float = 0.05
    t_total: float = 30.0
    follow_dist: float = 0.3
    v_wheel_max: float = 0.3
    w_max: float = 1.5
    v_max: float = 0.3
    a_max: float = 0.3
    steer_rate_max: float = None


def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_mpc_baseline(config: EvalConfig, ref_time: float) -> None:
    ensure_dir(config.baseline_save_dir)
    cmd = [
        sys.executable,
        config.baseline_script,
        "--output-dir",
        config.baseline_save_dir,
        "--seed",
        str(config.seed),
        "--ref-time",
        str(ref_time),
        "--follow-dist",
        str(config.follow_dist),
        "--side",
        str(config.baseline_side),
        "--steps",
        str(config.episode_steps),
        "--no-render",
    ]
    print(f"Running MPC baseline: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def compute_reference(
    context_state,
    follow_dist: float,
    robot_xy: Optional[Tuple[float, float]] = None,
    side: float = 1.0,
    right_normal: bool = False,
) -> Tuple[float, float, float]:
    ref = getattr(context_state, "reference", context_state)
    bx, by, bdx, bdy = ref
    speed = np.hypot(bdx, bdy)
    if speed > 1e-6:
        ux, uy = bdx / speed, bdy / speed
    else:
        ux, uy = 1.0, 0.0
    if right_normal:
        perp = np.array([uy, -ux], dtype=np.float32)
    else:
        perp = np.array([-uy, ux], dtype=np.float32)
    pnorm = np.hypot(perp[0], perp[1])
    if pnorm < 1e-6:
        perp_unit = np.array([0.0, 1.0])
    else:
        perp_unit = perp / pnorm
    rx = bx + perp_unit[0] * follow_dist * side
    ry = by + perp_unit[1] * follow_dist * side
    if robot_xy is None:
        yaw_ref = np.arctan2(by - ry, bx - rx)
    else:
        px, py = robot_xy
        yaw_ref = np.arctan2(by - py, bx - px)
    return float(rx), float(ry), float(yaw_ref)


def extract_env_ref(env) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    # Unwrap GOPS env to access reference data.
    base_env = getattr(env, "unwrapped", env)
    robot_state = base_env.robot.state
    context_state = base_env.context.state
    ref = np.array(context_state.reference, dtype=np.float32)
    robot_xy = (float(robot_state[0]), float(robot_state[1]))
    if hasattr(base_env, "side"):
        rx, ry, yaw_ref = compute_reference(
            ref,
            base_env.follow_dist,
            robot_xy=robot_xy,
            side=getattr(base_env, "side", 1.0),
        )
    else:
        rx, ry, yaw_ref = compute_reference(
            ref,
            base_env.follow_dist,
            robot_xy=robot_xy,
            right_normal=True,
        )
    target_ref = np.array([rx, ry], dtype=np.float32)
    return robot_state, ref, target_ref, yaw_ref


def get_velocity(env) -> Tuple[float, float]:
    base_env = getattr(env, "unwrapped", env)
    vx, vy = base_env.robot.vel_world
    return float(vx), float(vy)


def action_to_rates(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.size >= 8:
        action = action[:8]
    steer_rate = action[0::2]
    wheel_acc = action[1::2]
    return steer_rate, wheel_acc


def get_wheel_order(env) -> List[str]:
    base_env = getattr(env, "unwrapped", env)
    if hasattr(base_env, "robot") and hasattr(base_env.robot, "_steer"):
        return list(base_env.robot._steer.keys())
    return list(WHEEL_POS.keys())


def get_wheel_state(env) -> Tuple[np.ndarray, np.ndarray]:
    base_env = getattr(env, "unwrapped", env)
    wheel_order = get_wheel_order(env)
    steer = np.array([base_env.robot._steer[name] for name in wheel_order], dtype=np.float32)
    speed = np.array([base_env.robot._speed[name] for name in wheel_order], dtype=np.float32)
    return steer, speed


def compute_wheel_metrics(env) -> Tuple[np.ndarray, np.ndarray, float]:
    base_env = getattr(env, "unwrapped", env)
    yaw = float(base_env.robot.state[2])
    vx, vy = base_env.robot.vel_world
    w_body = float(base_env.robot.w_body)
    v_wheel_max = max(float(base_env.robot.v_wheel_max), 1e-6)

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    v_body_x = float(vx * cos_yaw + vy * sin_yaw)
    v_body_y = float(-vx * sin_yaw + vy * cos_yaw)

    slip_list = []
    util_list = []
    wheel_order = get_wheel_order(env)
    for name in wheel_order:
        x_i, y_i = WHEEL_POS.get(name, (0.0, 0.0))
        steer = float(base_env.robot._steer[name])
        speed = float(base_env.robot._speed[name])
        wheel_vx = v_body_x - w_body * y_i
        wheel_vy = v_body_y + w_body * x_i
        slip = (-np.sin(steer) * wheel_vx + np.cos(steer) * wheel_vy)
        slip_list.append(float(slip / v_wheel_max))
        util_list.append(float(abs(speed) / v_wheel_max))

    if abs(v_body_x) < 1e-6:
        denom = 1e-6 if v_body_x >= 0.0 else -1e-6
    else:
        denom = v_body_x
    sideslip = float(np.arctan2(v_body_y, denom))
    return np.array(slip_list, dtype=np.float32), np.array(util_list, dtype=np.float32), sideslip


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


def make_series_labels(data: np.ndarray, prefix: str, base_names: List[str] = None) -> List[str]:
    if data.ndim == 1:
        return [prefix]
    width = int(data.shape[1])
    if base_names is not None and len(base_names) == width:
        return list(base_names)
    return [f"{prefix}{i + 1}" for i in range(width)]


def prepare_eval_args(env, args: Dict) -> Dict:
    eval_args = dict(args)
    torch.set_num_threads(4)
    eval_args["use_gpu"] = False
    eval_args["enable_cuda"] = False
    eval_args["cnn_shared"] = eval_args.get("cnn_shared", False)

    if "obsv_dim" not in eval_args:
        eval_args["obsv_dim"] = env.observation_space.shape[0]
    if "action_dim" not in eval_args:
        eval_args["action_dim"] = env.action_space.shape[0]
    if "action_type" not in eval_args:
        eval_args["action_type"] = "continu"
    if "action_high_limit" not in eval_args:
        eval_args["action_high_limit"] = env.action_space.high.astype("float32")
    if "action_low_limit" not in eval_args:
        eval_args["action_low_limit"] = env.action_space.low.astype("float32")
    return eval_args


def get_reward_terms(step_info: Dict) -> Dict[str, float]:
    return {
        "reward_task": float(step_info.get("reward_task", 0.0)),
        "reward_mode": float(step_info.get("reward_mode", 0.0)),
        "reward_safe": float(step_info.get("reward_safe", 0.0)),
        "reward_smooth": float(step_info.get("reward_smooth", 0.0)),
    }


def evaluate_policy(config: EvalConfig) -> Dict[str, np.ndarray]:
    ensure_dir(config.save_dir)
    print(
        f"Eval start: episodes={config.num_episodes}, render={config.render}, "
        f"save_anim={config.save_anim}, checkpoint={config.checkpoint_path}"
    )
    print(f"Figures and animations will be saved to: {os.path.abspath(config.save_dir)}")

    ref_time = config.ref_time
    if ref_time is None and config.run_baseline:
        ref_time = 0.0
    if config.run_baseline:
        run_mpc_baseline(config, ref_time if ref_time is not None else 0.0)

    args = {
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
        "traj_type": config.traj_type,
        "episode_steps": config.episode_steps,
        "v_wheel_max": config.v_wheel_max,
        "w_max": config.w_max,
        "v_max": config.v_max,
        "a_max": config.a_max,
        "steer_rate_max": config.steer_rate_max,
        "enable_cuda": False,
    }
    if config.env_id is not None:
        args["env_id"] = config.env_id
    # Merge training config if available (for apprfunc settings)
    result_dir = os.path.dirname(os.path.dirname(config.checkpoint_path))
    config_path = os.path.join(result_dir, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            train_args = json.load(f)
        # Start from training config to preserve obs/action dims, then apply eval overrides.
        merged = dict(train_args)
        for k, v in args.items():
            if v is None:
                continue
            merged[k] = v
        args = merged

    env = create_env(**args)
    args = prepare_eval_args(env, args)
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
        "mean_abs_yaw_err": [],
        "max_abs_yaw_err": [],
        "mean_speed": [],
        "max_speed": [],
        "steps_taken": [],
        "completed_full": [],
        "mean_ctrl_energy": [],
    }

    for ep in range(config.num_episodes):
        print(f"Episode {ep + 1}/{config.num_episodes} reset...")
        if ref_time is None:
            obs, _ = env.reset(seed=config.seed + ep)
        else:
            obs, _ = env.reset(seed=config.seed + ep, ref_time=ref_time)
        done = False
        ep_return = 0.0

        time_hist: List[float] = []
        pose_hist: List[Tuple[float, float, float]] = []
        vel_hist: List[Tuple[float, float]] = []
        pos_err_hist: List[float] = []
        yaw_err_hist: List[float] = []
        ctrl_energy_hist: List[float] = []

        traj_robot: List[Tuple[float, float]] = []
        traj_ref: List[Tuple[float, float]] = []
        traj_ball: List[Tuple[float, float]] = []
        steer_rate_hist: List[np.ndarray] = []
        wheel_acc_hist: List[np.ndarray] = []
        steer_angle_hist: List[np.ndarray] = []
        wheel_speed_hist: List[np.ndarray] = []
        reward_hist: List[float] = []
        reward_task_hist: List[float] = []
        reward_mode_hist: List[float] = []
        reward_safe_hist: List[float] = []
        reward_smooth_hist: List[float] = []
        slip_hist: List[np.ndarray] = []
        util_hist: List[np.ndarray] = []
        sideslip_hist: List[float] = []

        step = 0
        while not done:
            action = policy_act_deterministic(alg.networks, obs)
            step_out = env.step(action)
            if len(step_out) == 5:
                obs, reward, terminated, truncated, step_info = step_out
                done = terminated or truncated
            else:
                obs, reward, done, step_info = step_out

            robot_state, ref, target_ref, yaw_ref = extract_env_ref(env)
            px, py, yaw = robot_state
            bx, by, bdx, bdy = ref
            rx, ry = target_ref
            vx, vy = get_velocity(env)

            pos_err = np.hypot(px - rx, py - ry)
            vel_err = np.hypot(vx - bdx, vy - bdy)
            yaw_err = wrap_to_pi(yaw - yaw_ref)

            steer_rate, wheel_acc = action_to_rates(np.asarray(action))
            steer_angle, wheel_speed = get_wheel_state(env)
            slip_norm, util_norm, sideslip = compute_wheel_metrics(env)
            ctrl_energy = float(np.sum(np.asarray(action) ** 2))

            traj_robot.append((px, py))
            traj_ref.append((rx, ry))
            traj_ball.append((bx, by))
            steer_rate_hist.append(steer_rate)
            wheel_acc_hist.append(wheel_acc)
            steer_angle_hist.append(steer_angle)
            wheel_speed_hist.append(wheel_speed)
            pos_err_hist.append(pos_err)
            yaw_err_hist.append(yaw_err)
            ctrl_energy_hist.append(ctrl_energy)
            reward_hist.append(float(reward))
            reward_terms = get_reward_terms(step_info)
            reward_task_hist.append(reward_terms["reward_task"])
            reward_mode_hist.append(reward_terms["reward_mode"])
            reward_safe_hist.append(reward_terms["reward_safe"])
            reward_smooth_hist.append(reward_terms["reward_smooth"])
            slip_hist.append(slip_norm)
            util_hist.append(util_norm)
            sideslip_hist.append(sideslip)
            elapsed_steps = int(step_info.get("elapsed_steps", step + 1))
            time_hist.append(elapsed_steps * config.dt)
            pose_hist.append((px, py, yaw))
            vel_hist.append((vx, vy))
            ep_return += float(reward)
            step += 1
            if step % 200 == 0:
                print(f"Episode {ep + 1}: step {step}, return {ep_return:.2f}")

            if config.render:
                env.render()

        pos_err_arr = np.array(pos_err_hist, dtype=np.float32)
        yaw_err_arr = np.array(yaw_err_hist, dtype=np.float32)
        vel_arr = np.array(vel_hist, dtype=np.float32)
        speed_arr = np.linalg.norm(vel_arr, axis=1) if vel_arr.size else np.zeros(1, dtype=np.float32)
        rms_pos = float(np.sqrt(np.mean(pos_err_arr ** 2)))
        results["episode_return"].append(ep_return)
        results["mean_pos_err"].append(float(np.mean(pos_err_arr)))
        results["max_pos_err"].append(float(np.max(pos_err_arr)))
        results["rms_pos_err"].append(rms_pos)
        results["mean_abs_yaw_err"].append(float(np.mean(np.abs(yaw_err_arr))))
        results["max_abs_yaw_err"].append(float(np.max(np.abs(yaw_err_arr))))
        results["mean_speed"].append(float(np.mean(speed_arr)))
        results["max_speed"].append(float(np.max(speed_arr)))
        results["steps_taken"].append(float(step_info.get("elapsed_steps", step)))
        results["completed_full"].append(float(step_info.get("elapsed_steps", step) >= config.episode_steps))
        results["mean_ctrl_energy"].append(float(np.mean(ctrl_energy_hist)))
        print(
            f"Episode {ep + 1} done: steps={step}, return={ep_return:.2f}, "
            f"mean_pos_err={np.mean(pos_err_arr):.4f}"
        )

        if config.save_fig:
            plot_episode(
                config.save_dir,
                ep,
                traj_robot,
                traj_ref,
                traj_ball,
                pos_err_hist,
                yaw_err_hist,
                steer_angle_hist,
                steer_rate_hist,
                wheel_speed_hist,
                wheel_acc_hist,
                reward_hist,
                reward_task_hist,
                reward_mode_hist,
                reward_safe_hist,
                reward_smooth_hist,
                slip_hist,
                util_hist,
                sideslip_hist,
                time_hist,
                vel_hist,
            )
            save_episode_timeseries_csv(
                config.save_dir,
                ep,
                time_hist,
                pose_hist,
                vel_hist,
                traj_ref,
                traj_ball,
                pos_err_hist,
                yaw_err_hist,
                steer_angle_hist,
                steer_rate_hist,
                wheel_speed_hist,
                wheel_acc_hist,
            )
        if config.save_anim:
            try:
                save_episode_animation(
                    config.save_dir,
                    ep,
                    time_hist,
                    pose_hist,
                    vel_hist,
                    traj_ref,
                    traj_ball,
                    steer_angle_hist,
                    wheel_speed_hist,
                    slip_hist,
                    util_hist,
                    config.anim_fps,
                    config.anim_format,
                    config.dt,
                )
            except Exception as exc:
                print(f"Warning: failed to save animation for episode {ep + 1}: {exc}")

    summarize_results(results)
    save_eval_summary_csv(config.save_dir, results)
    print(f"Saved evaluation outputs to: {os.path.abspath(config.save_dir)}")
    try:
        if config.open_dir and os.name == "nt":
            os.startfile(os.path.abspath(config.save_dir))
    except OSError:
        pass
    return results


def plot_episode(
    save_dir: str,
    ep: int,
    traj_robot: List[Tuple[float, float]],
    traj_ref: List[Tuple[float, float]],
    traj_ball: List[Tuple[float, float]],
    pos_err_hist: List[float],
    yaw_err_hist: List[float],
    steer_angle_hist: List[np.ndarray],
    steer_rate_hist: List[np.ndarray],
    wheel_speed_hist: List[np.ndarray],
    wheel_acc_hist: List[np.ndarray],
    reward_hist: List[float],
    reward_task_hist: List[float],
    reward_mode_hist: List[float],
    reward_safe_hist: List[float],
    reward_smooth_hist: List[float],
    slip_hist: List[np.ndarray],
    util_hist: List[np.ndarray],
    sideslip_hist: List[float],
    time_hist: List[float],
    vel_hist: List[Tuple[float, float]],
) -> None:
    import matplotlib.pyplot as plt

    traj_robot = np.array(traj_robot)
    traj_ref = np.array(traj_ref)
    traj_ball = np.array(traj_ball)
    steer_angle_hist = np.array(steer_angle_hist)
    steer_angle_deg_hist = np.rad2deg(steer_angle_hist)
    steer_rate_hist = np.array(steer_rate_hist)
    wheel_speed_hist = np.array(wheel_speed_hist)
    wheel_acc_hist = np.array(wheel_acc_hist)
    slip_hist = np.array(slip_hist)
    util_hist = np.array(util_hist)
    time_arr = np.array(time_hist, dtype=np.float32)
    vel_hist = np.array(vel_hist, dtype=np.float32)
    speed_hist = np.linalg.norm(vel_hist, axis=1) if vel_hist.size else np.zeros(1, dtype=np.float32)
    base_wheel_names = list(WHEEL_POS.keys())
    steer_labels = make_series_labels(steer_angle_hist, "wheel_", base_wheel_names)
    rate_labels = make_series_labels(steer_rate_hist, "wheel_", base_wheel_names)
    speed_labels = make_series_labels(wheel_speed_hist, "wheel_", base_wheel_names)
    acc_labels = make_series_labels(wheel_acc_hist, "wheel_", base_wheel_names)
    slip_labels = make_series_labels(slip_hist, "wheel_", base_wheel_names)
    util_labels = make_series_labels(util_hist, "wheel_", base_wheel_names)

    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(4, 2, 1)
    ax1.plot(traj_robot[:, 0], traj_robot[:, 1], label="robot")
    ax1.plot(traj_ref[:, 0], traj_ref[:, 1], label="reference R")
    ax1.plot(traj_ball[:, 0], traj_ball[:, 1], label="target ball B", linestyle="--")
    ax1.set_title("Trajectory")
    ax1.set_aspect("equal")
    ax1.grid(True)
    ax1.legend()

    ax2 = fig.add_subplot(4, 2, 2)
    ax2.plot(pos_err_hist, label="position error")
    ax2.plot(yaw_err_hist, label="yaw error")
    ax2.set_title("Tracking Error")
    ax2.grid(True)
    ax2.legend()

    ax3 = fig.add_subplot(4, 2, 3)
    ax3.plot(steer_angle_deg_hist, label=steer_labels)
    ax3.set_title("Steer Angles (deg)")
    ax3.set_ylabel("deg")
    ax3.grid(True)
    ax3.legend()

    ax4 = fig.add_subplot(4, 2, 4)
    ax4.plot(steer_rate_hist, label=rate_labels)
    ax4.set_title("Steer Rates")
    ax4.grid(True)
    ax4.legend()

    ax5 = fig.add_subplot(4, 2, 5)
    ax5.plot(wheel_speed_hist, label=speed_labels)
    ax5.set_title("Wheel Speeds")
    ax5.grid(True)
    ax5.legend()

    ax6 = fig.add_subplot(4, 2, 6)
    ax6.plot(wheel_acc_hist, label=acc_labels)
    ax6.set_title("Wheel Accelerations")
    ax6.grid(True)
    ax6.legend()

    ax7 = fig.add_subplot(4, 2, 7)
    ax7.plot(slip_hist, label=slip_labels)
    ax7.set_title("Wheel Slip (normalized lateral)")
    ax7.grid(True)
    ax7.legend()

    ax8 = fig.add_subplot(4, 2, 8)
    ax8.plot(util_hist, label=util_labels)
    ax8.set_title("Wheel Utilization |speed|/v_max")
    ax8.grid(True)
    ax8.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"episode_{ep:03d}.png"))
    plt.close(fig)

    fig2 = plt.figure(figsize=(8, 4))
    ax = fig2.add_subplot(2, 1, 1)
    ax.plot(reward_hist)
    ax.set_title("Reward")
    ax.grid(True)
    ax2 = fig2.add_subplot(2, 1, 2)
    ax2.plot(np.array(sideslip_hist))
    ax2.set_title("Body Sideslip Angle (rad)")
    ax2.grid(True)
    fig2.tight_layout()
    fig2.savefig(os.path.join(save_dir, f"episode_{ep:03d}_reward.png"))
    plt.close(fig2)

    fig_breakdown = plt.figure(figsize=(10, 8))
    ax1 = fig_breakdown.add_subplot(3, 1, 1)
    ax1.plot(reward_hist, label="reward_total", color="tab:blue")
    ax1.set_title("Reward Total")
    ax1.grid(True)
    ax1.legend()

    ax2 = fig_breakdown.add_subplot(3, 1, 2)
    ax2.plot(reward_task_hist, label="reward_task")
    ax2.plot(reward_mode_hist, label="reward_mode")
    ax2.plot(reward_safe_hist, label="reward_safe")
    ax2.plot(reward_smooth_hist, label="reward_smooth")
    ax2.set_title("Reward Breakdown")
    ax2.grid(True)
    ax2.legend()

    ax3 = fig_breakdown.add_subplot(3, 1, 3)
    reward_stack = np.vstack(
        [
            np.asarray(reward_task_hist, dtype=np.float32),
            np.asarray(reward_mode_hist, dtype=np.float32),
            np.asarray(reward_safe_hist, dtype=np.float32),
            np.asarray(reward_smooth_hist, dtype=np.float32),
        ]
    ).T
    ax3.plot(np.cumsum(reward_stack[:, 0]), label="cum_task")
    ax3.plot(np.cumsum(reward_stack[:, 1]), label="cum_mode")
    ax3.plot(np.cumsum(reward_stack[:, 2]), label="cum_safe")
    ax3.plot(np.cumsum(reward_stack[:, 3]), label="cum_smooth")
    ax3.set_title("Cumulative Reward Breakdown")
    ax3.grid(True)
    ax3.legend()

    fig_breakdown.tight_layout()
    fig_breakdown.savefig(os.path.join(save_dir, f"episode_{ep:03d}_reward_breakdown.png"))
    plt.close(fig_breakdown)

    fig3 = plt.figure(figsize=(10, 5))
    ax = fig3.add_subplot(1, 1, 1)
    ax.plot(steer_angle_deg_hist, label=steer_labels)
    ax.set_title("Actual Wheel Steering Angles (deg)")
    ax.set_xlabel("step")
    ax.set_ylabel("deg")
    ax.grid(True)
    ax.legend()
    fig3.tight_layout()
    fig3.savefig(os.path.join(save_dir, f"episode_{ep:03d}_steer_deg.png"))
    plt.close(fig3)

    fig4 = plt.figure(figsize=(10, 6))
    ax1 = fig4.add_subplot(2, 1, 1)
    ax1.plot(time_arr, vel_hist[:, 0], label="vx")
    ax1.plot(time_arr, vel_hist[:, 1], label="vy")
    ax1.plot(time_arr, speed_hist, label="speed")
    ax1.set_title("Robot Velocity")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("m/s")
    ax1.grid(True)
    ax1.legend()

    ax2 = fig4.add_subplot(2, 1, 2)
    ax2.plot(time_arr, np.asarray(pos_err_hist, dtype=np.float32), label="position error")
    ax2.plot(time_arr, np.rad2deg(np.asarray(yaw_err_hist, dtype=np.float32)), label="yaw error (deg)")
    ax2.set_title("Tracking Error")
    ax2.set_xlabel("time (s)")
    ax2.grid(True)
    ax2.legend()
    fig4.tight_layout()
    fig4.savefig(os.path.join(save_dir, f"episode_{ep:03d}_task1_velocity_error.png"))
    plt.close(fig4)


def save_episode_timeseries_csv(
    save_dir: str,
    ep: int,
    time_hist: List[float],
    pose_hist: List[Tuple[float, float, float]],
    vel_hist: List[Tuple[float, float]],
    traj_ref: List[Tuple[float, float]],
    traj_ball: List[Tuple[float, float]],
    pos_err_hist: List[float],
    yaw_err_hist: List[float],
    steer_angle_hist: List[np.ndarray],
    steer_rate_hist: List[np.ndarray],
    wheel_speed_hist: List[np.ndarray],
    wheel_acc_hist: List[np.ndarray],
) -> None:
    wheel_names = list(WHEEL_POS.keys())
    csv_path = os.path.join(save_dir, f"episode_{ep:03d}_timeseries.csv")
    fieldnames = [
        "time_s",
        "robot_x",
        "robot_y",
        "robot_yaw_rad",
        "robot_vx",
        "robot_vy",
        "robot_speed",
        "ref_x",
        "ref_y",
        "target_x",
        "target_y",
        "pos_err",
        "yaw_err_rad",
        "yaw_err_deg",
    ]
    for name in wheel_names:
        fieldnames.append(f"{name}_steer_deg")
    for name in wheel_names:
        fieldnames.append(f"{name}_steer_rate")
    for name in wheel_names:
        fieldnames.append(f"{name}_wheel_speed")
    for name in wheel_names:
        fieldnames.append(f"{name}_wheel_acc")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, time_s in enumerate(time_hist):
            px, py, yaw = pose_hist[idx]
            vx, vy = vel_hist[idx]
            rx, ry = traj_ref[idx]
            bx, by = traj_ball[idx]
            row = {
                "time_s": float(time_s),
                "robot_x": float(px),
                "robot_y": float(py),
                "robot_yaw_rad": float(yaw),
                "robot_vx": float(vx),
                "robot_vy": float(vy),
                "robot_speed": float(np.hypot(vx, vy)),
                "ref_x": float(rx),
                "ref_y": float(ry),
                "target_x": float(bx),
                "target_y": float(by),
                "pos_err": float(pos_err_hist[idx]),
                "yaw_err_rad": float(yaw_err_hist[idx]),
                "yaw_err_deg": float(np.rad2deg(yaw_err_hist[idx])),
            }
            steer = np.rad2deg(np.asarray(steer_angle_hist[idx], dtype=np.float32))
            steer_rate = np.asarray(steer_rate_hist[idx], dtype=np.float32)
            wheel_speed = np.asarray(wheel_speed_hist[idx], dtype=np.float32)
            wheel_acc = np.asarray(wheel_acc_hist[idx], dtype=np.float32)
            for j, name in enumerate(wheel_names):
                row[f"{name}_steer_deg"] = float(steer[j])
                row[f"{name}_steer_rate"] = float(steer_rate[j])
                row[f"{name}_wheel_speed"] = float(wheel_speed[j])
                row[f"{name}_wheel_acc"] = float(wheel_acc[j])
            writer.writerow(row)


def save_episode_animation(
    save_dir: str,
    ep: int,
    time_hist: List[float],
    pose_hist: List[Tuple[float, float, float]],
    vel_hist: List[Tuple[float, float]],
    traj_ref: List[Tuple[float, float]],
    traj_ball: List[Tuple[float, float]],
    steer_hist: List[np.ndarray],
    speed_hist: List[np.ndarray],
    slip_hist: List[np.ndarray],
    util_hist: List[np.ndarray],
    fps: int,
    anim_format: str,
    dt: float,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib import cm
    from matplotlib.patches import Circle, Polygon

    pose = np.array(pose_hist, dtype=np.float32)
    vel = np.array(vel_hist, dtype=np.float32)
    ref = np.array(traj_ref, dtype=np.float32)
    ball = np.array(traj_ball, dtype=np.float32)
    steer_hist = np.array(steer_hist, dtype=np.float32)
    speed_hist = np.array(speed_hist, dtype=np.float32)
    slip_hist = np.array(slip_hist, dtype=np.float32)
    util_hist = np.array(util_hist, dtype=np.float32)
    wheel_names = list(WHEEL_POS.keys())
    can_draw_wheels = (
        steer_hist.ndim == 2
        and speed_hist.ndim == 2
        and slip_hist.ndim == 2
        and util_hist.ndim == 2
        and steer_hist.shape[1] == len(wheel_names)
        and speed_hist.shape[1] == len(wheel_names)
        and slip_hist.shape[1] == len(wheel_names)
        and util_hist.shape[1] == len(wheel_names)
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.grid(True)
    ax.plot(ref[:, 0], ref[:, 1], color="tab:orange", linewidth=1.0, label="reference R")
    ax.plot(ball[:, 0], ball[:, 1], color="tab:red", linewidth=1.0, linestyle="--", alpha=0.45, label="target ball B")
    robot_line, = ax.plot([], [], color="tab:blue", linewidth=1.5, label="robot path")
    ball_line, = ax.plot([], [], color="tab:red", linewidth=1.2, label="ball path")
    ball_pt = Circle((0.0, 0.0), radius=0.05, color="tab:red")
    ax.add_patch(ball_pt)
    chassis = Polygon(np.zeros((4, 2)), closed=True, fill=False, edgecolor="k", lw=2)
    ax.add_patch(chassis)
    heading_line, = ax.plot([], [], "k-", lw=2, label="heading")
    vel_line, = ax.plot([], [], "g-", lw=2, label="velocity")
    wheel_patches = {}
    wheel_spokes = {}
    if can_draw_wheels:
        for name in wheel_names:
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

    xmin = min(np.min(pose[:, 0]), np.min(ref[:, 0]), np.min(ball[:, 0])) - 0.8
    xmax = max(np.max(pose[:, 0]), np.max(ref[:, 0]), np.max(ball[:, 0])) + 0.8
    ymin = min(np.min(pose[:, 1]), np.min(ref[:, 1]), np.min(ball[:, 1])) - 0.8
    ymax = max(np.max(pose[:, 1]), np.max(ref[:, 1]), np.max(ball[:, 1])) + 0.8
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    path_x: List[float] = []
    path_y: List[float] = []
    ball_x: List[float] = []
    ball_y: List[float] = []
    wheel_phi = {name: 0.0 for name in wheel_names}
    time_text = ax.text(
        0.02,
        0.96,
        "",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
    )
    metric_text = ax.text(
        0.02,
        0.90,
        "",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
    )
    cmap = cm.get_cmap("coolwarm")

    def init():
        robot_line.set_data([], [])
        ball_line.set_data([], [])
        heading_line.set_data([], [])
        vel_line.set_data([], [])
        time_text.set_text("")
        metric_text.set_text("")
        return (
            [robot_line, ball_line, heading_line, vel_line, time_text, metric_text, ball_pt, chassis]
            + list(wheel_patches.values())
            + list(wheel_spokes.values())
        )

    def update(i):
        px, py, yaw = pose[i]
        vx, vy = vel[i]
        bx, by = ball[i]

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

        if can_draw_wheels:
            for idx, name in enumerate(wheel_names):
                steer = steer_hist[i][idx]
                speed = speed_hist[i][idx]
                slip = float(slip_hist[i][idx])
                util = float(util_hist[i][idx])
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
                slip_mag = min(1.0, abs(slip))
                wheel_patches[name].set_facecolor(cmap(0.5 + 0.5 * np.sign(slip) * slip_mag))
                wheel_patches[name].set_edgecolor("k" if util > 0.7 else "darkgray")

                wheel_phi[name] = wrap_to_pi(
                    wheel_phi[name] + (speed / max(WHEEL_RADIUS, 1e-6)) * dt
                )
                phi = wheel_phi[name]
                spoke_len = WHEEL_WID * 0.9
                spoke_local = np.array([0.0, spoke_len / 2])
                spoke_rot = np.array(
                    [[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]]
                )
                spoke_local = spoke_rot @ spoke_local
                p1 = center - wheel_rot @ spoke_local
                p2 = center + wheel_rot @ spoke_local
                wheel_spokes[name].set_data([p1[0], p2[0]], [p1[1], p2[1]])

        if i < len(time_hist):
            time_text.set_text(f"t={time_hist[i]:.2f} s")
            mean_slip = float(np.mean(np.abs(slip_hist[i])))
            mean_util = float(np.mean(util_hist[i]))
            metric_text.set_text(
                f"mean|slip|={mean_slip:.2f}  mean|util|={mean_util:.2f}"
                + ("" if can_draw_wheels else "  wheel drawing skipped")
            )
        return (
            [robot_line, ball_line, heading_line, vel_line, time_text, metric_text, ball_pt, chassis]
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
        try:
            ani.save(anim_path, writer="pillow", fps=fps)
        except Exception:
            # Fallback: render frames manually and let PIL assemble the GIF.
            frames = []
            init()
            step = max(1, len(pose) // 300)
            for i in range(0, len(pose), step):
                update(i)
                fig.canvas.draw()
                w, h = fig.canvas.get_width_height()
                rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
                rgb = rgba[:, :, :3].copy()
                frame = Image.fromarray(rgb, mode="RGB").quantize(colors=255)
                frames.append(frame)
            if not frames:
                raise RuntimeError("No frames rendered for GIF export.")
            duration_ms = max(1, int(1000 * step / max(fps, 1)))
            frames[0].save(
                anim_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration_ms,
                loop=0,
                optimize=False,
                disposal=2,
            )
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
    _stat("mean_abs_yaw_err")
    _stat("completed_full")
    _stat("mean_ctrl_energy")


def save_eval_summary_csv(save_dir: str, results: Dict[str, List[float]]) -> None:
    fieldnames = ["episode"] + list(results.keys())
    max_len = max((len(v) for v in results.values()), default=0)
    csv_path = os.path.join(save_dir, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(max_len):
            row = {"episode": idx}
            for key, values in results.items():
                row[key] = float(values[idx]) if idx < len(values) else ""
            writer.writerow(row)


def build_default_save_dir(checkpoint_path: str) -> str:
    result_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    return os.path.join(result_dir, f"eval_{checkpoint_name}")


def build_default_baseline_save_dir(save_dir: str) -> str:
    return f"{save_dir}_baseline_mpc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ToRwheelsim policy checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=EvalConfig.checkpoint_path,
        help="Path to the policy checkpoint to evaluate.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save evaluation figures and animations. Defaults to a folder derived from checkpoint name.",
    )
    parser.add_argument("--episodes", type=int, default=3, help="Number of evaluation episodes.")
    parser.add_argument("--episode-steps", type=int, default=EvalConfig.episode_steps, help="Maximum environment steps per episode.")
    parser.add_argument("--seed", type=int, default=12345, help="Evaluation seed.")
    parser.add_argument("--ref-time", type=float, default=0.0, help="Fixed reference start time for reset.")
    parser.add_argument(
        "--traj-type",
        type=str,
        default=EvalConfig.traj_type,
        choices=["rounded_rect", "line_forward", "line_lateral", "line_diagonal", "circle", "s_curve"],
        help="Reference trajectory used by the Python simulation environment.",
    )
    parser.add_argument(
        "--render",
        dest="render",
        action="store_true",
        help="Render the environment while evaluating.",
    )
    parser.add_argument(
        "--no-render",
        dest="render",
        action="store_false",
        help="Disable live rendering while evaluating.",
    )
    parser.set_defaults(render=True)
    parser.add_argument(
        "--save-anim",
        dest="save_anim",
        action="store_true",
        help="Save episode animation.",
    )
    parser.add_argument(
        "--no-save-anim",
        dest="save_anim",
        action="store_false",
        help="Do not save episode animation.",
    )
    parser.set_defaults(save_anim=True)
    parser.add_argument(
        "--save-fig",
        dest="save_fig",
        action="store_true",
        help="Save episode summary figures.",
    )
    parser.add_argument(
        "--no-save-fig",
        dest="save_fig",
        action="store_false",
        help="Do not save episode summary figures.",
    )
    parser.set_defaults(save_fig=True)
    parser.add_argument(
        "--anim-format",
        type=str,
        default="gif",
        choices=["gif", "mp4"],
        help="Animation format.",
    )
    parser.add_argument("--open-dir", action="store_true", help="Open the output directory after evaluation.")
    return parser.parse_args()


def make_eval_config(cli_args: argparse.Namespace) -> EvalConfig:
    checkpoint_path = cli_args.checkpoint
    save_dir = cli_args.save_dir or build_default_save_dir(checkpoint_path)
    return EvalConfig(
        checkpoint_path=checkpoint_path,
        save_dir=save_dir,
        baseline_save_dir=build_default_baseline_save_dir(save_dir),
        traj_type=cli_args.traj_type,
        seed=cli_args.seed,
        ref_time=cli_args.ref_time,
        episode_steps=cli_args.episode_steps,
        num_episodes=cli_args.episodes,
        render=cli_args.render,
        save_fig=cli_args.save_fig,
        save_anim=cli_args.save_anim,
        anim_format=cli_args.anim_format,
        open_dir=cli_args.open_dir,
    )


if __name__ == "__main__":
    cli_args = parse_args()
    eval_jobs = [make_eval_config(cli_args)]
    for cfg in eval_jobs:
        evaluate_policy(cfg)
