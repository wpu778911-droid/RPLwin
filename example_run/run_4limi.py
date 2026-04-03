import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from matplotlib.patches import Circle, Polygon

from gops.create_pkg.create_alg import create_alg
from gops.create_pkg.create_env import create_env
from gops.utils.init_args import init_args


METHODS = ("nearest_angle", "rule_preview", "rl")
WHEEL_ORDER = ("FL", "FR", "RR", "RL")
ACTIVE_WHEEL_INDICES = [0, 2]
WHEEL_POS = {
    "FL": (+0.24, +0.175),
    "FR": (+0.24, -0.175),
    "RR": (-0.24, -0.175),
    "RL": (-0.24, +0.175),
}
CHASSIS_L = 0.48
CHASSIS_W = 0.35
WHEEL_LEN = 0.10
WHEEL_WID = 0.04


@dataclass
class EvalConfig:
    checkpoint_path: str = "results/PPO_paThi_sim2_gen_v1/apprfunc/apprfunc_550_opt.pkl"
    save_dir: str = "results/PPO_paThi_sim2_gen_v1/eval_apprfunc_550_opt_three_paths"
    env_id: str = "env_paThi_sim2"
    algorithm: str = "PPO"
    seed: int = 12345
    num_episodes: int = 1
    episode_steps: int = 900
    render: bool = False
    save_fig: bool = True
    save_anim: bool = True
    anim_fps: int = 16
    anim_format: str = "gif"
    traj_types: Optional[List[str]] = None
    ref_time: Optional[float] = None
    methods: Optional[List[str]] = None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def build_default_save_dir(checkpoint_path: str) -> str:
    result_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    return os.path.join(result_dir, f"eval_{checkpoint_name}_three_paths")


DISPLAY_NAME = {
    "nearest_angle": "nearest_angle",
    "rule_preview": "multi_solution",
    "rl": "rl",
    "snake5_easy": "snake5_easy",
    "snake5_medium": "snake5_medium",
    "snake5_hard": "snake5_hard",
}

TRAJ_DIFFICULTY_WEIGHT = {
    "snake5_easy": 1.0,
    "snake5_medium": 2.0,
    "snake5_hard": 3.0,
}


def load_train_args(checkpoint_path: str) -> Dict:
    result_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    config_path = os.path.join(result_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Training config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_eval_args(train_args: Dict, config: EvalConfig, mode: str, traj_type: Optional[str]) -> Dict:
    args = dict(train_args)
    args.update(
        {
            "env_id": config.env_id,
            "algorithm": config.algorithm,
            "seed": config.seed,
            "enable_cuda": False,
            "is_render": config.render,
            "mode": mode,
            "random_traj_on_reset": False,
            "episode_steps": config.episode_steps,
        }
    )
    if traj_type is not None:
        args["traj_type"] = traj_type
    return args


def create_eval_env(args: Dict):
    return create_env(**args)


def create_rl_alg(config: EvalConfig, base_args: Dict):
    env = create_env(**base_args)
    inited_args = init_args(env, **base_args)
    alg = create_alg(**inited_args)
    if not os.path.isfile(config.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {config.checkpoint_path}")
    state_dict = torch.load(config.checkpoint_path, map_location="cpu")
    alg.load_state_dict(state_dict)
    alg.eval()
    return alg


def policy_act_deterministic(networks, obs: np.ndarray) -> int:
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = networks.policy(obs_t)
        act_dist = networks.create_action_distributions(logits)
        if hasattr(act_dist, "mode"):
            act = act_dist.mode()
        elif hasattr(act_dist, "probs"):
            act = torch.argmax(act_dist.probs, dim=-1)
        else:
            act = torch.argmax(logits, dim=-1)
    return int(act.detach().cpu().numpy().reshape(-1)[0])


def format_action(action: int) -> np.ndarray:
    return np.asarray(action, dtype=np.int64)


def extract_env_states(env):
    base_env = getattr(env, "unwrapped", env)
    robot_state = np.array(base_env.robot.state, dtype=np.float32)
    ref_state = np.array(base_env.context.state.reference, dtype=np.float32)
    steer = np.array(base_env.robot._steer, dtype=np.float32)
    return robot_state, ref_state, steer


def get_reward_breakdown(info: Dict) -> Dict[str, float]:
    reward_breakdown = {"track": 0.0, "safe": 0.0, "branch": 0.0, "smooth": 0.0, "survive": 0.0, "total": 0.0}
    reward_breakdown.update(info.get("reward_breakdown", {}))
    return reward_breakdown


def run_single_episode(env, method: str, alg, seed: int, ref_time: Optional[float]):
    reset_kwargs = {"seed": seed}
    if ref_time is not None:
        reset_kwargs["ref_time"] = ref_time
    obs, info = env.reset(**reset_kwargs)
    done = False
    episode_return = 0.0

    robot_xy_hist = []
    robot_pose_hist = []
    ref_xy_hist = []
    track_error_hist = []
    yaw_error_hist = []
    active_margins_hist = []
    wheel_angles_hist = []
    selected_branch_hist = []
    preview_best_branch_hist = []
    baseline_branch_hist = []
    reward_total_hist = []
    reward_track_hist = []
    reward_safe_hist = []
    reward_branch_hist = []
    reward_smooth_hist = []

    while not done:
        if method == "rl":
            action = format_action(policy_act_deterministic(alg.networks, obs))
        else:
            action = format_action(0)

        step_out = env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = step_out

        robot_state, ref_state, _ = extract_env_states(env)
        px, py, yaw = robot_state
        rx, ry, _, _, ryaw = ref_state
        track_error = float(np.hypot(px - rx, py - ry))
        yaw_error = float(wrap_to_pi(yaw - ryaw))
        reward_breakdown = get_reward_breakdown(info)

        robot_xy_hist.append([px, py])
        robot_pose_hist.append([px, py, yaw])
        ref_xy_hist.append([rx, ry])
        track_error_hist.append(track_error)
        yaw_error_hist.append(yaw_error)
        active_margins_hist.append(np.asarray(info.get("active_wheel_margins", np.zeros(2)), dtype=np.float32))
        wheel_angles_hist.append(np.asarray(extract_env_states(env)[2], dtype=np.float32))
        selected_branch_hist.append(float(info.get("selected_branch_id", 0)))
        preview_best_branch_hist.append(float(info.get("preview_best_branch_id", 0)))
        baseline_branch_hist.append(float(info.get("baseline_branch_id", 0)))
        reward_total_hist.append(float(reward_breakdown["total"]))
        reward_track_hist.append(float(reward_breakdown["track"]))
        reward_safe_hist.append(float(reward_breakdown["safe"]))
        reward_branch_hist.append(float(reward_breakdown["branch"]))
        reward_smooth_hist.append(float(reward_breakdown["smooth"]))

        episode_return += float(reward)
        if getattr(env, "render", None) is not None and getattr(env.unwrapped if hasattr(env, "unwrapped") else env, "mode", "") and False:
            env.render()

    data = {
        "robot_xy": np.asarray(robot_xy_hist, dtype=np.float32),
        "robot_pose": np.asarray(robot_pose_hist, dtype=np.float32),
        "ref_xy": np.asarray(ref_xy_hist, dtype=np.float32),
        "track_error": np.asarray(track_error_hist, dtype=np.float32),
        "yaw_error": np.asarray(yaw_error_hist, dtype=np.float32),
        "active_margins_deg": np.rad2deg(np.asarray(active_margins_hist, dtype=np.float32)),
        "wheel_angles": np.asarray(wheel_angles_hist, dtype=np.float32),
        "selected_branch": np.asarray(selected_branch_hist, dtype=np.float32),
        "preview_best_branch": np.asarray(preview_best_branch_hist, dtype=np.float32),
        "baseline_branch": np.asarray(baseline_branch_hist, dtype=np.float32),
        "reward_total": np.asarray(reward_total_hist, dtype=np.float32),
        "reward_track": np.asarray(reward_track_hist, dtype=np.float32),
        "reward_safe": np.asarray(reward_safe_hist, dtype=np.float32),
        "reward_branch": np.asarray(reward_branch_hist, dtype=np.float32),
        "reward_smooth": np.asarray(reward_smooth_hist, dtype=np.float32),
        "episode_return": float(episode_return),
        "limit_hit_count": float(info.get("limit_hit_count", 0)),
        "branch_switch_count": float(info.get("branch_switch_count", 0)),
        "failure_reason": info.get("failure_reason", "none"),
    }
    return data


def get_reference_plot_bounds(env_args: Dict, seed: int, ref_time: Optional[float], episode_steps: int) -> Dict[str, float]:
    env = create_eval_env(env_args)
    reset_kwargs = {"seed": seed}
    if ref_time is not None:
        reset_kwargs["ref_time"] = ref_time
    env.reset(**reset_kwargs)

    base_env = getattr(env, "unwrapped", env)
    dt = float(getattr(base_env, "dt", env_args.get("dt", 0.05)))
    ref_time0 = float(getattr(base_env.context, "ref_time", ref_time if ref_time is not None else 0.0))

    ref_points = []
    for k in range(episode_steps + 1):
        ref_state = np.asarray(base_env.get_ref_state(ref_time0 + k * dt), dtype=np.float32)
        ref_points.append(ref_state[:2])
    ref_xy = np.asarray(ref_points, dtype=np.float32)
    pad = 0.9
    return {
        "xmin": float(np.min(ref_xy[:, 0]) - pad),
        "xmax": float(np.max(ref_xy[:, 0]) + pad),
        "ymin": float(np.min(ref_xy[:, 1]) - pad),
        "ymax": float(np.max(ref_xy[:, 1]) + pad),
    }


def plot_episode(
    save_dir: str, method: str, traj_name: str, ep: int, data: Dict[str, np.ndarray], plot_bounds: Dict[str, float]
) -> None:
    fig = plt.figure(figsize=(14, 10))
    method_label = DISPLAY_NAME.get(method, method)
    traj_label = DISPLAY_NAME.get(traj_name, traj_name)

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(data["robot_xy"][:, 0], data["robot_xy"][:, 1], label="robot")
    ax1.plot(data["ref_xy"][:, 0], data["ref_xy"][:, 1], label="reference")
    ax1.set_title(f"Trajectory | {method_label} | {traj_label}")
    ax1.set_aspect("equal")
    ax1.set_xlim(plot_bounds["xmin"], plot_bounds["xmax"])
    ax1.set_ylim(plot_bounds["ymin"], plot_bounds["ymax"])
    ax1.grid(True)
    ax1.legend()

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(data["track_error"], label="track")
    ax2.plot(np.abs(data["yaw_error"]), label="|yaw|")
    ax2.set_title("Tracking Error")
    ax2.grid(True)
    ax2.legend()

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.step(np.arange(len(data["selected_branch"])), data["selected_branch"], where="post", label="selected")
    ax3.step(np.arange(len(data["preview_best_branch"])), data["preview_best_branch"], where="post", label="preview")
    ax3.step(np.arange(len(data["baseline_branch"])), data["baseline_branch"], where="post", label="nearest")
    ax3.set_title("Branch Selection")
    ax3.grid(True)
    ax3.legend()

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(data["active_margins_deg"][:, 0], label="FL margin")
    ax4.plot(data["active_margins_deg"][:, 1], label="RR margin")
    ax4.set_title("Active Margin To Limit (deg)")
    ax4.grid(True)
    ax4.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{method}_{traj_name}_ep{ep:03d}.png"))
    plt.close(fig)


def save_episode_animation(
    save_dir: str,
    method: str,
    traj_name: str,
    ep: int,
    data: Dict[str, np.ndarray],
    plot_bounds: Dict[str, float],
    fps: int,
    anim_format: str,
) -> None:
    robot_xy = data["robot_xy"]
    ref_xy = data["ref_xy"]
    robot_pose = data["robot_pose"]
    wheel_angles = data["wheel_angles"]
    track_error = data["track_error"]
    selected_branch = data["selected_branch"]
    active_margins_deg = data["active_margins_deg"]
    method_label = DISPLAY_NAME.get(method, method)
    traj_label = DISPLAY_NAME.get(traj_name, traj_name)
    frame_stride = 1
    if anim_format == "gif":
        frame_stride = max(1, int(np.ceil(len(robot_xy) / 700.0)))
    frame_ids = np.arange(0, len(robot_xy), frame_stride, dtype=int)
    if frame_ids[-1] != len(robot_xy) - 1:
        frame_ids = np.append(frame_ids, len(robot_xy) - 1)

    fig = plt.figure(figsize=(15, 7.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[2.4, 2.4, 1.2], height_ratios=[1.0, 1.0])
    ax1 = fig.add_subplot(gs[:, :2])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 2])

    ax1.set_title(f"{method_label} | {traj_label}")
    ax1.set_aspect("equal")
    ax1.grid(True)
    ax1.plot(ref_xy[:, 0], ref_xy[:, 1], color="tab:orange", linewidth=1.2, label="reference")
    robot_line, = ax1.plot([], [], color="tab:blue", linewidth=1.8, label="robot")
    ref_pt, = ax1.plot([], [], "o", color="tab:orange")
    wheel_centers = {}
    wheel_patches = {}
    for name in WHEEL_ORDER:
        wheel_centers[name], = ax1.plot([], [], "ko", markersize=2)
        wheel_patches[name] = Polygon(np.zeros((4, 2)), closed=True, fill=True, alpha=0.8)
        ax1.add_patch(wheel_patches[name])
    chassis = Polygon(np.zeros((4, 2)), closed=True, fill=False, edgecolor="k", linewidth=2)
    ax1.add_patch(chassis)
    ax1.legend()

    ax2.set_title("Branch")
    ax2.grid(True)
    ax2.set_xlim(0, max(len(selected_branch) - 1, 1))
    ax2.set_ylim(-0.2, 3.2)
    branch_line, = ax2.step([], [], where="post", color="tab:green")

    ax3.set_title("Track Error / Margin")
    ax3.grid(True)
    ax3.set_xlim(0, max(len(track_error) - 1, 1))
    upper = max(float(np.max(track_error)) * 1.2, float(np.max(active_margins_deg)) * 0.1, 1.0)
    ax3.set_ylim(0.0, max(upper, 5.0))
    err_line, = ax3.plot([], [], color="tab:red", label="track error")
    margin_line, = ax3.plot([], [], color="tab:purple", label="min margin / 50")
    ax3.legend()

    ax1.set_xlim(plot_bounds["xmin"], plot_bounds["xmax"])
    ax1.set_ylim(plot_bounds["ymin"], plot_bounds["ymax"])

    def init():
        robot_line.set_data([], [])
        ref_pt.set_data([], [])
        branch_line.set_data([], [])
        err_line.set_data([], [])
        margin_line.set_data([], [])
        return [robot_line, ref_pt, branch_line, err_line, margin_line, chassis, *wheel_patches.values(), *wheel_centers.values()]

    def update(frame_idx):
        i = int(frame_ids[frame_idx])
        robot_line.set_data(robot_xy[: i + 1, 0], robot_xy[: i + 1, 1])
        ref_pt.set_data([ref_xy[i, 0]], [ref_xy[i, 1]])

        px, py, yaw = robot_pose[i]
        c = np.cos(yaw)
        s = np.sin(yaw)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)

        chassis_local = np.array(
            [
                [CHASSIS_L / 2, CHASSIS_W / 2],
                [CHASSIS_L / 2, -CHASSIS_W / 2],
                [-CHASSIS_L / 2, -CHASSIS_W / 2],
                [-CHASSIS_L / 2, CHASSIS_W / 2],
            ],
            dtype=np.float32,
        )
        chassis_world = (rot @ chassis_local.T).T + np.array([px, py], dtype=np.float32)
        chassis.set_xy(chassis_world)

        wheel_shape = np.array(
            [
                [WHEEL_LEN / 2, WHEEL_WID / 2],
                [WHEEL_LEN / 2, -WHEEL_WID / 2],
                [-WHEEL_LEN / 2, -WHEEL_WID / 2],
                [-WHEEL_LEN / 2, WHEEL_WID / 2],
            ],
            dtype=np.float32,
        )

        for idx, name in enumerate(WHEEL_ORDER):
            wx, wy = WHEEL_POS[name]
            center = rot @ np.array([wx, wy], dtype=np.float32) + np.array([px, py], dtype=np.float32)
            wheel_centers[name].set_data([center[0]], [center[1]])
            steer = float(wheel_angles[i, idx])
            wheel_rot_local = np.array(
                [[np.cos(steer), -np.sin(steer)], [np.sin(steer), np.cos(steer)]],
                dtype=np.float32,
            )
            wheel_world = (rot @ (wheel_rot_local @ wheel_shape.T)).T + center
            wheel_patches[name].set_xy(wheel_world)
            wheel_patches[name].set_facecolor("tab:blue" if idx in ACTIVE_WHEEL_INDICES else "silver")

        xs = np.arange(i + 1)
        branch_line.set_data(xs, selected_branch[: i + 1])
        err_line.set_data(xs, track_error[: i + 1])
        margin_line.set_data(xs, np.min(active_margins_deg[: i + 1], axis=1) / 50.0)
        return [robot_line, ref_pt, branch_line, err_line, margin_line, chassis, *wheel_patches.values(), *wheel_centers.values()]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_ids),
        init_func=init,
        interval=1000 / fps,
        blit=False,
    )
    anim_path = os.path.join(save_dir, f"{method}_{traj_name}_ep{ep:03d}.{anim_format}")
    if anim_format == "gif":
        ani.save(anim_path, writer="pillow", fps=max(1, int(round(fps / frame_stride))))
    else:
        if animation.writers.is_available("ffmpeg"):
            ani.save(anim_path, writer="ffmpeg", fps=fps)
        else:
            plt.close(fig)
            print(
                f"Skip animation for method={method}, traj={traj_name}, ep={ep:03d}: "
                "ffmpeg is not available, so mp4 cannot be written."
            )
            return
    plt.close(fig)


def summarize_episode(data: Dict[str, np.ndarray], method: str, traj_name: str, ep: int) -> Dict[str, object]:
    return {
        "method": method,
        "traj_type": traj_name,
        "episode": int(ep),
        "episode_return": float(data["episode_return"]),
        "mean_track_error": float(np.mean(data["track_error"])),
        "max_track_error": float(np.max(data["track_error"])),
        "rms_track_error": float(np.sqrt(np.mean(np.square(data["track_error"])))),
        "mean_margin_deg": float(np.mean(data["active_margins_deg"])),
        "min_margin_deg": float(np.min(data["active_margins_deg"])),
        "branch_switch_count": float(data["branch_switch_count"]),
        "limit_hit_count": float(data["limit_hit_count"]),
        "failure_reason": str(data["failure_reason"]),
    }


def save_summary_csv(save_dir: str, rows: List[Dict[str, object]]) -> None:
    csv_path = os.path.join(save_dir, "summary.csv")
    fieldnames = [
        "method",
        "traj_type",
        "episode",
        "episode_return",
        "mean_track_error",
        "max_track_error",
        "rms_track_error",
        "mean_margin_deg",
        "min_margin_deg",
        "branch_switch_count",
        "limit_hit_count",
        "failure_reason",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_json(save_dir: str, rows: List[Dict[str, object]]) -> None:
    json_path = os.path.join(save_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def save_method_comparison_plot(save_dir: str, rows: List[Dict[str, object]]) -> None:
    methods = sorted({row["method"] for row in rows})
    metrics = ["mean_track_error", "mean_margin_deg", "branch_switch_count", "limit_hit_count"]
    titles = ["Mean Track Error", "Mean Margin (deg)", "Branch Switch Count", "Limit Hit Count"]

    fig = plt.figure(figsize=(14, 10))
    for idx, (metric, title) in enumerate(zip(metrics, titles), start=1):
        ax = fig.add_subplot(2, 2, idx)
        vals = []
        for method in methods:
            method_rows = [row for row in rows if row["method"] == method]
            vals.append(float(np.mean([row[metric] for row in method_rows])))
        ax.bar(methods, vals, color=["tab:gray", "tab:orange", "tab:blue"][: len(methods)])
        ax.set_title(title)
        ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "method_comparison.png"))
    plt.close(fig)


def save_method_ranking_plot(save_dir: str, rows: List[Dict[str, object]]) -> None:
    methods = sorted({row["method"] for row in rows})
    traj_types = ["snake5_easy", "snake5_medium", "snake5_hard"]
    ranking_scores = {method: 0.0 for method in methods}

    for traj_name in traj_types:
        traj_rows = [row for row in rows if row["traj_type"] == traj_name]
        if not traj_rows:
            continue

        weight = TRAJ_DIFFICULTY_WEIGHT.get(traj_name, 1.0)
        err_vals = np.array([row["mean_track_error"] for row in traj_rows], dtype=np.float32)
        margin_vals = np.array([row["min_margin_deg"] for row in traj_rows], dtype=np.float32)
        hit_vals = np.array([row["limit_hit_count"] for row in traj_rows], dtype=np.float32)
        ret_vals = np.array([row["episode_return"] for row in traj_rows], dtype=np.float32)

        err_min, err_max = float(np.min(err_vals)), float(np.max(err_vals))
        margin_min, margin_max = float(np.min(margin_vals)), float(np.max(margin_vals))
        hit_min, hit_max = float(np.min(hit_vals)), float(np.max(hit_vals))
        ret_min, ret_max = float(np.min(ret_vals)), float(np.max(ret_vals))

        for row in traj_rows:
            method = row["method"]
            err_score = 1.0 if abs(err_max - err_min) < 1e-6 else 1.0 - (row["mean_track_error"] - err_min) / (err_max - err_min)
            margin_score = 1.0 if abs(margin_max - margin_min) < 1e-6 else (row["min_margin_deg"] - margin_min) / (margin_max - margin_min)
            hit_score = 1.0 if abs(hit_max - hit_min) < 1e-6 else 1.0 - (row["limit_hit_count"] - hit_min) / (hit_max - hit_min)
            ret_score = 1.0 if abs(ret_max - ret_min) < 1e-6 else (row["episode_return"] - ret_min) / (ret_max - ret_min)
            failure_penalty = 0.0 if row["failure_reason"] in {"none", "timeout"} else 0.20

            score = weight * (0.45 * err_score + 0.25 * margin_score + 0.20 * hit_score + 0.10 * ret_score - failure_penalty)
            ranking_scores[method] += score

    ordered = sorted(methods, key=lambda m: ranking_scores[m], reverse=True)
    labels = [DISPLAY_NAME.get(method, method) for method in ordered]
    vals = [ranking_scores[method] for method in ordered]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    bars = ax.bar(labels, vals, color=["tab:blue", "tab:orange", "tab:gray"][: len(labels)])
    ax.set_title("Overall Ranking Score")
    ax.set_ylabel("Weighted Score")
    ax.grid(True, axis="y")
    for idx, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"#{idx + 1}", ha="center", va="bottom", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "method_ranking.png"))
    plt.close(fig)


def evaluate_methods(config: EvalConfig) -> List[Dict[str, object]]:
    ensure_dir(config.save_dir)
    train_args = load_train_args(config.checkpoint_path)
    traj_types = config.traj_types or ["snake5_easy", "snake5_medium", "snake5_hard"]
    methods = config.methods or list(METHODS)

    if "rl" in methods:
        rl_args = merge_eval_args(train_args, config, "rl", traj_types[0] if traj_types else None)
        alg = create_rl_alg(config, rl_args)
    else:
        alg = None

    summary_rows: List[Dict[str, object]] = []

    # Build one shared evaluation plan and align all methods to the RL test cases.
    rl_case_args = merge_eval_args(train_args, config, "rl", traj_types[0] if traj_types else None)
    shared_cases = []
    for traj_index, traj_name in enumerate(traj_types):
        for ep in range(config.num_episodes):
            shared_seed = config.seed + traj_index * config.num_episodes + ep
            shared_cases.append(
                {
                    "traj_name": traj_name,
                    "episode": ep,
                    "seed": shared_seed,
                }
            )

    for traj_index, traj_name in enumerate(traj_types):
        plot_bounds_by_episode = []
        for ep in range(config.num_episodes):
            shared_seed = config.seed + traj_index * config.num_episodes + ep
            ref_env_args = dict(rl_case_args)
            ref_env_args["traj_type"] = traj_name
            plot_bounds_by_episode.append(
                get_reference_plot_bounds(ref_env_args, seed=shared_seed, ref_time=config.ref_time, episode_steps=config.episode_steps)
            )

        for method in methods:
            method_dir = os.path.join(config.save_dir, method)
            ensure_dir(method_dir)
            print(f"Evaluating method={method}, traj={traj_name}")

            env_args = merge_eval_args(train_args, config, method, traj_name)
            env = create_eval_env(env_args)

            for ep in range(config.num_episodes):
                case = shared_cases[traj_index * config.num_episodes + ep]
                seed = int(case["seed"])
                plot_bounds = plot_bounds_by_episode[ep]
                data = run_single_episode(env, method, alg, seed=seed, ref_time=config.ref_time)
                row = summarize_episode(data, method, traj_name, ep)
                summary_rows.append(row)

                if config.save_fig:
                    plot_episode(method_dir, method, traj_name, ep, data, plot_bounds)
                if config.save_anim:
                    save_episode_animation(
                        method_dir,
                        method,
                        traj_name,
                        ep,
                        data,
                        plot_bounds,
                        fps=config.anim_fps,
                        anim_format=config.anim_format,
                    )

                print(
                    f"method={method}, traj={traj_name}, ep={ep + 1}/{config.num_episodes}, "
                    f"ret={row['episode_return']:.3f}, mean_err={row['mean_track_error']:.4f}, "
                    f"min_margin_deg={row['min_margin_deg']:.2f}, limit_hits={row['limit_hit_count']:.0f}"
                )

    save_summary_csv(config.save_dir, summary_rows)
    save_summary_json(config.save_dir, summary_rows)
    save_method_comparison_plot(config.save_dir, summary_rows)
    save_method_ranking_plot(config.save_dir, summary_rows)
    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare nearest_angle, rule_preview, and PPO on paThi_sim2.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=r"E:\StudyApp\Gops\install\GOPS-dev\results\PPO_paThi_sim2_gen_v1\apprfunc\apprfunc_550_opt.pkl",
        help="Path to PPO checkpoint.",
    )
    parser.add_argument("--save-dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes per trajectory.")
    parser.add_argument("--episode-steps", type=int, default=900, help="Evaluation horizon for each fixed snake path.")
    parser.add_argument("--seed", type=int, default=12345, help="Evaluation seed.")
    parser.add_argument("--ref-time", type=float, default=None, help="Fixed reference time.")
    parser.add_argument("--traj-types", nargs="+", default=["snake5_easy", "snake5_medium", "snake5_hard"])
    parser.add_argument("--methods", nargs="+", default=["nearest_angle", "rule_preview", "rl"], choices=METHODS)
    parser.add_argument("--render", dest="render", action="store_true")
    parser.add_argument("--no-render", dest="render", action="store_false")
    parser.set_defaults(render=False)
    parser.add_argument("--save-fig", dest="save_fig", action="store_true")
    parser.add_argument("--no-save-fig", dest="save_fig", action="store_false")
    parser.set_defaults(save_fig=True)
    parser.add_argument("--save-anim", dest="save_anim", action="store_true")
    parser.add_argument("--no-save-anim", dest="save_anim", action="store_false")
    parser.set_defaults(save_anim=True)
    parser.add_argument("--anim-fps", type=int, default=16)
    parser.add_argument("--anim-format", type=str, default="gif", choices=["gif", "mp4"])
    return parser.parse_args()


def make_eval_config(cli_args: argparse.Namespace) -> EvalConfig:
    save_dir = cli_args.save_dir or build_default_save_dir(cli_args.checkpoint)
    return EvalConfig(
        checkpoint_path=cli_args.checkpoint,
        save_dir=save_dir,
        seed=cli_args.seed,
        num_episodes=cli_args.episodes,
        episode_steps=cli_args.episode_steps,
        render=cli_args.render,
        save_fig=cli_args.save_fig,
        save_anim=cli_args.save_anim,
        anim_fps=cli_args.anim_fps,
        anim_format=cli_args.anim_format,
        traj_types=cli_args.traj_types,
        ref_time=cli_args.ref_time,
        methods=cli_args.methods,
    )


if __name__ == "__main__":
    cli_args = parse_args()
    config = make_eval_config(cli_args)
    evaluate_methods(config)
