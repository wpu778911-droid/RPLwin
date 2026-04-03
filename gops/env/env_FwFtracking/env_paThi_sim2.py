from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from gym import spaces

from gops.env.env_gen_ocp.pyth_base import Context, ContextState, Env, Robot, State

STEER_LIMIT = np.deg2rad(170.0)
STEER_RATE_MAX = np.deg2rad(120.0)
WHEEL_ACC_MAX = 1.0

BALL_SPEED = 0.1
RECT_W = 2.5
RECT_H = 1.5
CORNER_R = 0.3

WHEEL_ORDER = ("FL", "FR", "RR", "RL")
WHEEL_POS = {
    "FL": (+0.24, +0.175),
    "FR": (+0.24, -0.175),
    "RR": (-0.24, -0.175),
    "RL": (-0.24, +0.175),
}
ACTIVE_WHEEL_INDICES = [0, 2]
PASSIVE_WHEEL_INDICES = [1, 3]
BRANCH_BITS = ((0, 0), (0, 1), (1, 0), (1, 1))


def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def angle_diff(a: float, b: float) -> float:
    return wrap_to_pi(a - b)


def world_to_body(px: float, py: float, yaw: float, xw: float, yw: float) -> Tuple[float, float]:
    dx = xw - px
    dy = yw - py
    c = np.cos(yaw)
    s = np.sin(yaw)
    xb = c * dx + s * dy
    yb = -s * dx + c * dy
    return float(xb), float(yb)


def body_to_world(yaw: float, xb: float, yb: float) -> Tuple[float, float]:
    c = np.cos(yaw)
    s = np.sin(yaw)
    xw = c * xb - s * yb
    yw = s * xb + c * yb
    return float(xw), float(yw)


def _rounded_rect_perimeter() -> float:
    straight_x = 2.0 * (RECT_W - CORNER_R)
    straight_y = 2.0 * (RECT_H - CORNER_R)
    arc = 0.5 * np.pi * CORNER_R
    return 2.0 * (straight_x + straight_y) + 4.0 * arc


def _rounded_rect_pos_vel(s: float) -> Tuple[float, float, float, float]:
    straight_x = 2.0 * (RECT_W - CORNER_R)
    straight_y = 2.0 * (RECT_H - CORNER_R)
    arc = 0.5 * np.pi * CORNER_R

    if s < straight_x:
        return -RECT_W + CORNER_R + s, -RECT_H, BALL_SPEED, 0.0
    s -= straight_x

    if s < arc:
        theta = -0.5 * np.pi + s / CORNER_R
        cx, cy = RECT_W - CORNER_R, -RECT_H + CORNER_R
        return (
            cx + CORNER_R * np.cos(theta),
            cy + CORNER_R * np.sin(theta),
            BALL_SPEED * (-np.sin(theta)),
            BALL_SPEED * np.cos(theta),
        )
    s -= arc

    if s < straight_y:
        return RECT_W, -RECT_H + CORNER_R + s, 0.0, BALL_SPEED
    s -= straight_y

    if s < arc:
        theta = s / CORNER_R
        cx, cy = RECT_W - CORNER_R, RECT_H - CORNER_R
        return (
            cx + CORNER_R * np.cos(theta),
            cy + CORNER_R * np.sin(theta),
            BALL_SPEED * (-np.sin(theta)),
            BALL_SPEED * np.cos(theta),
        )
    s -= arc

    if s < straight_x:
        return RECT_W - CORNER_R - s, RECT_H, -BALL_SPEED, 0.0
    s -= straight_x

    if s < arc:
        theta = 0.5 * np.pi + s / CORNER_R
        cx, cy = -RECT_W + CORNER_R, RECT_H - CORNER_R
        return (
            cx + CORNER_R * np.cos(theta),
            cy + CORNER_R * np.sin(theta),
            BALL_SPEED * (-np.sin(theta)),
            BALL_SPEED * np.cos(theta),
        )
    s -= arc

    if s < straight_y:
        return -RECT_W, RECT_H - CORNER_R - s, 0.0, -BALL_SPEED
    s -= straight_y

    theta = np.pi + s / CORNER_R
    cx, cy = -RECT_W + CORNER_R, -RECT_H + CORNER_R
    return (
        cx + CORNER_R * np.cos(theta),
        cy + CORNER_R * np.sin(theta),
        BALL_SPEED * (-np.sin(theta)),
        BALL_SPEED * np.cos(theta),
    )


def _make_progressive_snake_params(
    num_bends: int,
    segment_length: float,
    amp_start: float,
    amp_end: float,
    amp_power: float,
    x_start: float,
    sign_start: int = 1,
) -> Dict[str, float]:
    return {
        "num_bends": int(num_bends),
        "segment_length": float(segment_length),
        "amp_start": float(amp_start),
        "amp_end": float(amp_end),
        "amp_power": float(amp_power),
        "x_start": float(x_start),
        "sign_start": int(1 if sign_start >= 0 else -1),
    }


def _sample_train_generator_params(np_random) -> Dict[str, float]:
    family = str(
        np_random.choice(
            [
                "progressive_snake",
                "sine_mix",
                "wide_turns",
                "lissajous",
                "switchback_chain",
                "trap_chain",
            ]
        )
    )

    if family == "progressive_snake":
        num_bends = int(np_random.integers(8, 15))
        segment_length = float(np_random.uniform(0.65, 1.05))
        amp_start = float(np_random.uniform(0.08, 0.20))
        amp_end = float(np_random.uniform(0.75, 1.45))
        amp_power = float(np_random.uniform(0.9, 2.2))
        x_start = float(np_random.uniform(-3.2, -2.6))
        sign_start = 1 if float(np_random.uniform()) > 0.5 else -1
        params = _make_progressive_snake_params(
            num_bends=num_bends,
            segment_length=segment_length,
            amp_start=amp_start,
            amp_end=amp_end,
            amp_power=amp_power,
            x_start=x_start,
            sign_start=sign_start,
        )
    elif family == "sine_mix":
        params = {
            "family": family,
            "x_start": float(np_random.uniform(-3.2, -2.4)),
            "a1": float(np_random.uniform(0.35, 1.00)),
            "a2": float(np_random.uniform(0.18, 0.55)),
            "k1": float(np_random.uniform(0.7, 1.6)),
            "k2": float(np_random.uniform(1.8, 4.2)),
            "phase1": float(np_random.uniform(-np.pi, np.pi)),
            "phase2": float(np_random.uniform(-np.pi, np.pi)),
        }
    elif family == "wide_turns":
        params = {
            "family": family,
            "x_start": float(np_random.uniform(-3.0, -2.2)),
            "amp": float(np_random.uniform(0.8, 1.5)),
            "k": float(np_random.uniform(0.45, 0.9)),
            "phase": float(np_random.uniform(-np.pi, np.pi)),
        }
    elif family == "lissajous":
        params = {
            "family": family,
            "ax": float(np_random.uniform(1.2, 2.1)),
            "ay": float(np_random.uniform(0.7, 1.4)),
            "wx": float(np_random.uniform(0.45, 0.85)),
            "wy": float(np_random.uniform(0.9, 1.7)),
            "phase": float(np_random.uniform(-0.6 * np.pi, 0.6 * np.pi)),
        }
    elif family == "switchback_chain":
        params = {
            "family": family,
            "x_start": float(np_random.uniform(-3.0, -2.4)),
            "y_scale": float(np_random.uniform(0.7, 1.3)),
            "stretch": float(np_random.uniform(0.8, 1.3)),
        }
    else:
        params = {
            "family": "trap_chain",
            "x_start": float(np_random.uniform(-3.1, -2.5)),
            "y_scale": float(np_random.uniform(0.8, 1.25)),
            "stretch": float(np_random.uniform(0.9, 1.35)),
        }

    params["family"] = family
    return params


def _default_progressive_snake10_params() -> Dict[str, float]:
    return _make_progressive_snake_params(
        num_bends=10,
        segment_length=0.90,
        amp_start=0.10,
        amp_end=1.05,
        amp_power=1.35,
        x_start=-2.85,
        sign_start=1,
    )


def _default_snake5_easy_params() -> Dict[str, float]:
    return _make_progressive_snake_params(
        num_bends=5,
        segment_length=0.92,
        amp_start=0.16,
        amp_end=0.72,
        amp_power=1.12,
        x_start=-2.8,
        sign_start=1,
    )


def _default_snake5_medium_params() -> Dict[str, float]:
    return _make_progressive_snake_params(
        num_bends=5,
        segment_length=0.74,
        amp_start=0.24,
        amp_end=1.08,
        amp_power=1.35,
        x_start=-2.8,
        sign_start=1,
    )


def _default_snake5_hard_params() -> Dict[str, float]:
    return _make_progressive_snake_params(
        num_bends=5,
        segment_length=0.58,
        amp_start=0.34,
        amp_end=1.42,
        amp_power=1.65,
        x_start=-2.8,
        sign_start=1,
    )


def _progressive_snake_pos_vel(t: float, v: float, params: Dict[str, float]) -> Tuple[float, float, float, float]:
    x = float(params["x_start"] + v * t)
    local_x = x - float(params["x_start"])
    segment_length = max(float(params["segment_length"]), 1e-6)
    num_bends = max(int(params["num_bends"]), 1)
    total_length = num_bends * segment_length

    if local_x <= 0.0:
        return x, 0.0, v, 0.0
    if local_x >= total_length:
        return x, 0.0, v, 0.0

    seg_idx = min(int(local_x / segment_length), num_bends - 1)
    u = (local_x - seg_idx * segment_length) / segment_length
    amp_alpha = 0.0 if num_bends <= 1 else (seg_idx / float(num_bends - 1))
    amp = float(params["amp_start"]) + (float(params["amp_end"]) - float(params["amp_start"])) * (
        amp_alpha ** float(params["amp_power"])
    )
    sign = int(params["sign_start"]) * (1 if seg_idx % 2 == 0 else -1)

    sin_term = np.sin(np.pi * u)
    y = sign * amp * (sin_term**2)
    dy_dx = sign * amp * (np.pi / segment_length) * np.sin(2.0 * np.pi * u)
    dx = v
    dy = v * dy_dx
    return x, float(y), float(dx), float(dy)


def _sine_mix_pos_vel(t: float, v: float, params: Dict[str, float]) -> Tuple[float, float, float, float]:
    x = float(params["x_start"] + v * t)
    y = float(
        params["a1"] * np.sin(params["k1"] * x + params["phase1"])
        + params["a2"] * np.sin(params["k2"] * x + params["phase2"])
    )
    dy_dx = float(
        params["a1"] * params["k1"] * np.cos(params["k1"] * x + params["phase1"])
        + params["a2"] * params["k2"] * np.cos(params["k2"] * x + params["phase2"])
    )
    return x, y, v, float(v * dy_dx)


def _wide_turns_pos_vel(t: float, v: float, params: Dict[str, float]) -> Tuple[float, float, float, float]:
    x = float(params["x_start"] + v * t)
    y = float(params["amp"] * np.sin(params["k"] * x + params["phase"]))
    dy_dx = float(params["amp"] * params["k"] * np.cos(params["k"] * x + params["phase"]))
    return x, y, v, float(v * dy_dx)


def _lissajous_pos_vel(t: float, v: float, params: Dict[str, float]) -> Tuple[float, float, float, float]:
    # Use v to scale traversal rate so the family stays compatible with the rest of the env.
    omega = v / max(float(params["ax"]), 1e-6)
    tau = t * omega
    x = float(params["ax"] * np.sin(params["wx"] * tau))
    y = float(params["ay"] * np.sin(params["wy"] * tau + params["phase"]))
    dx = float(params["ax"] * params["wx"] * omega * np.cos(params["wx"] * tau))
    dy = float(params["ay"] * params["wy"] * omega * np.cos(params["wy"] * tau + params["phase"]))
    return x, y, dx, dy


def _switchback_chain_pos_vel(t: float, v: float, params: Dict[str, float]) -> Tuple[float, float, float, float]:
    x = float(params["x_start"] + v * t)
    z = float(params["stretch"] * x)
    y = float(
        params["y_scale"]
        * (
            0.95 * np.tanh(3.5 * (z + 1.8))
            - 1.15 * np.tanh(4.1 * (z + 0.55))
            + 1.25 * np.tanh(4.5 * (z - 0.65))
            - 1.05 * np.tanh(3.9 * (z - 1.85))
        )
    )
    dy_dz = float(
        params["y_scale"]
        * (
            0.95 * 3.5 * (1.0 - np.tanh(3.5 * (z + 1.8)) ** 2)
            - 1.15 * 4.1 * (1.0 - np.tanh(4.1 * (z + 0.55)) ** 2)
            + 1.25 * 4.5 * (1.0 - np.tanh(4.5 * (z - 0.65)) ** 2)
            - 1.05 * 3.9 * (1.0 - np.tanh(3.9 * (z - 1.85)) ** 2)
        )
    )
    dy_dx = float(params["stretch"] * dy_dz)
    return x, y, v, float(v * dy_dx)


def _trap_chain_pos_vel(t: float, v: float, params: Dict[str, float]) -> Tuple[float, float, float, float]:
    x = float(params["x_start"] + v * t)
    z = float(params["stretch"] * x)
    exp1 = np.exp(-((z + 1.6) ** 2) / 0.12)
    exp2 = np.exp(-((z + 0.2) ** 2) / 0.08)
    exp3 = np.exp(-((z - 1.0) ** 2) / 0.07)
    exp4 = np.exp(-((z - 2.05) ** 2) / 0.11)
    y = float(params["y_scale"] * (0.22 * np.sin(0.8 * z) + 0.85 * exp1 - 1.05 * exp2 + 1.15 * exp3 - 0.95 * exp4))
    dy_dz = float(
        params["y_scale"]
        * (
            0.22 * 0.8 * np.cos(0.8 * z)
            + 0.85 * exp1 * (-2.0 * (z + 1.6) / 0.12)
            - 1.05 * exp2 * (-2.0 * (z + 0.2) / 0.08)
            + 1.15 * exp3 * (-2.0 * (z - 1.0) / 0.07)
            - 0.95 * exp4 * (-2.0 * (z - 2.05) / 0.11)
        )
    )
    dy_dx = float(params["stretch"] * dy_dz)
    return x, y, v, float(v * dy_dx)


def get_ref_state_by_type(
    t: float, traj_type: str, speed_scale: float = 1.0, traj_params: Optional[Dict[str, float]] = None
) -> np.ndarray:
    v = BALL_SPEED * speed_scale
    if traj_type == "line":
        x, y, dx, dy = -2.5 + v * t, 0.0, v, 0.0
    elif traj_type == "circle":
        radius = 1.6
        omega = v / max(radius, 1e-6)
        phi = omega * t
        x = radius * np.cos(phi)
        y = radius * np.sin(phi)
        dx = -radius * omega * np.sin(phi)
        dy = radius * omega * np.cos(phi)
    elif traj_type == "s":
        x = -2.5 + v * t
        y = 0.8 * np.sin(1.2 * x)
        dx = v
        dy = 0.96 * v * np.cos(1.2 * x)
    elif traj_type == "eight":
        radius = 1.2
        omega = v / max(radius, 1e-6)
        phi = omega * t
        x = radius * np.sin(phi)
        y = 0.5 * radius * np.sin(2.0 * phi)
        dx = radius * omega * np.cos(phi)
        dy = radius * omega * np.cos(2.0 * phi)
    elif traj_type == "rounded_rect":
        period = _rounded_rect_perimeter() / max(v, 1e-6)
        s = (t % period) * v
        x, y, dx, dy = _rounded_rect_pos_vel(s)
    elif traj_type == "snake_dense":
        x = -2.5 + v * t
        y = 1.1 * np.sin(2.4 * x) + 0.35 * np.sin(5.2 * x)
        dx = v
        dy = v * (1.1 * 2.4 * np.cos(2.4 * x) + 0.35 * 5.2 * np.cos(5.2 * x))
    elif traj_type == "tight_eight":
        radius = 0.72
        omega = 2.4 * v / max(radius, 1e-6)
        phi = omega * t
        x = radius * np.sin(phi)
        y = 0.82 * radius * np.sin(2.0 * phi)
        dx = radius * omega * np.cos(phi)
        dy = 1.64 * radius * omega * np.cos(2.0 * phi)
    elif traj_type == "switchback":
        x = -2.4 + v * t
        y = (
            0.95 * np.tanh(4.2 * (x + 1.45))
            - 1.2 * np.tanh(4.8 * (x + 0.25))
            + 1.05 * np.tanh(4.5 * (x - 0.95))
        )
        dx = v
        dy = v * (
            0.95 * 4.2 * (1.0 - np.tanh(4.2 * (x + 1.45)) ** 2)
            - 1.2 * 4.8 * (1.0 - np.tanh(4.8 * (x + 0.25)) ** 2)
            + 1.05 * 4.5 * (1.0 - np.tanh(4.5 * (x - 0.95)) ** 2)
        )
    elif traj_type == "trap":
        x = -2.5 + v * t
        y = (
            0.15 * np.sin(1.0 * x)
            + 0.95 * np.exp(-((x + 0.95) ** 2) / 0.08)
            - 1.1 * np.exp(-((x - 0.15) ** 2) / 0.06)
            + 0.95 * np.exp(-((x - 1.05) ** 2) / 0.08)
        )
        dx = v
        dy = v * (
            0.15 * np.cos(1.0 * x)
            + 0.95 * np.exp(-((x + 0.95) ** 2) / 0.08) * (-2.0 * (x + 0.95) / 0.08)
            - 1.1 * np.exp(-((x - 0.15) ** 2) / 0.06) * (-2.0 * (x - 0.15) / 0.06)
            + 0.95 * np.exp(-((x - 1.05) ** 2) / 0.08) * (-2.0 * (x - 1.05) / 0.08)
        )
    elif traj_type == "progressive_snake10":
        params = _default_progressive_snake10_params() if traj_params is None else traj_params
        x, y, dx, dy = _progressive_snake_pos_vel(t, v, params)
    elif traj_type == "snake5_easy":
        params = _default_snake5_easy_params() if traj_params is None else traj_params
        x, y, dx, dy = _progressive_snake_pos_vel(t, v, params)
    elif traj_type == "snake5_medium":
        params = _default_snake5_medium_params() if traj_params is None else traj_params
        x, y, dx, dy = _progressive_snake_pos_vel(t, v, params)
    elif traj_type == "snake5_hard":
        params = _default_snake5_hard_params() if traj_params is None else traj_params
        x, y, dx, dy = _progressive_snake_pos_vel(t, v, params)
    elif traj_type == "train_generator":
        params = _sample_train_generator_params(np.random.default_rng()) if traj_params is None else traj_params
        family = str(params.get("family", "progressive_snake"))
        if family == "progressive_snake":
            x, y, dx, dy = _progressive_snake_pos_vel(t, v, params)
        elif family == "sine_mix":
            x, y, dx, dy = _sine_mix_pos_vel(t, v, params)
        elif family == "wide_turns":
            x, y, dx, dy = _wide_turns_pos_vel(t, v, params)
        elif family == "lissajous":
            x, y, dx, dy = _lissajous_pos_vel(t, v, params)
        elif family == "switchback_chain":
            x, y, dx, dy = _switchback_chain_pos_vel(t, v, params)
        elif family == "trap_chain":
            x, y, dx, dy = _trap_chain_pos_vel(t, v, params)
        else:
            x, y, dx, dy = _progressive_snake_pos_vel(t, v, _default_progressive_snake10_params())
    else:
        seg = 2.0
        omega = 0.5
        if t < seg:
            x, y, dx, dy = -2.0 + v * t, -1.0, v, 0.0
        elif t < 2.0 * seg:
            tau = t - seg
            x = 0.8 * np.sin(omega * tau)
            y = -1.0 + v * tau
            dx = 0.8 * omega * np.cos(omega * tau)
            dy = v
        else:
            tau = t - 2.0 * seg
            x = 0.8 + v * tau
            y = 1.0 - 0.5 * np.sin(omega * tau)
            dx = v
            dy = -0.5 * omega * np.cos(omega * tau)

    yaw = float(np.arctan2(dy, dx)) if np.hypot(dx, dy) > 1e-8 else 0.0
    return np.array([x, y, dx, dy, yaw], dtype=np.float32)


class PathRefContext(Context):
    def __init__(self, dt: float, t_total: float, traj_type: str, random_traj_on_reset: bool):
        self.dt = dt
        self.t_total = t_total
        self.traj_type = traj_type
        self.random_traj_on_reset = random_traj_on_reset
        self.ref_time = 0.0
        self.traj_params: Optional[Dict[str, float]] = None
        self.state = ContextState(reference=np.zeros(5, dtype=np.float32), t=0)
        self._traj_candidates = (
            "line",
            "circle",
            "s",
            "eight",
            "rounded_rect",
            "snake_dense",
            "tight_eight",
            "switchback",
            "trap",
            "snake5_easy",
            "snake5_medium",
            "snake5_hard",
            "progressive_snake10",
            "poly",
        )

    def _prepare_traj_params(self) -> None:
        if self.traj_type == "train_generator":
            self.traj_params = _sample_train_generator_params(self.np_random)
        elif self.traj_type == "progressive_snake10":
            self.traj_params = _default_progressive_snake10_params()
        elif self.traj_type == "snake5_easy":
            self.traj_params = _default_snake5_easy_params()
        elif self.traj_type == "snake5_medium":
            self.traj_params = _default_snake5_medium_params()
        elif self.traj_type == "snake5_hard":
            self.traj_params = _default_snake5_hard_params()
        else:
            self.traj_params = None

    def reset(self, ref_time: float = 0.0, traj_type: Optional[str] = None) -> ContextState[np.ndarray]:
        self.ref_time = ref_time
        if traj_type is not None:
            self.traj_type = traj_type
        elif self.random_traj_on_reset or self.traj_type == "random":
            idx = int(self.np_random.integers(0, len(self._traj_candidates)))
            self.traj_type = self._traj_candidates[idx]
        self._prepare_traj_params()
        self.state = ContextState(
            reference=get_ref_state_by_type(self.ref_time, self.traj_type, traj_params=self.traj_params), t=0
        )
        return self.state

    def step(self) -> ContextState[np.ndarray]:
        self.ref_time += self.dt
        self.state = ContextState(
            reference=get_ref_state_by_type(self.ref_time, self.traj_type, traj_params=self.traj_params), t=0
        )
        return self.state

    def get_zero_state(self) -> ContextState[np.ndarray]:
        return ContextState(reference=np.zeros(5, dtype=np.float32), t=0)


class Diag2ActiveRobot(Robot):
    def __init__(
        self,
        dt: float,
        v_wheel_max: float,
        steer_rate_max: Optional[float],
        wheel_acc_max: Optional[float],
        passive_steer_rate_max: Optional[float],
        passive_wheel_acc_max: Optional[float],
        w_max: float,
        v_max: float,
        a_max: float,
        lsq_reg: float = 1e-4,
    ):
        self.dt = float(dt)
        self.v_wheel_max = float(v_wheel_max)
        self.steer_rate_max = STEER_RATE_MAX if steer_rate_max is None else float(steer_rate_max)
        self.wheel_acc_max = WHEEL_ACC_MAX if wheel_acc_max is None else float(wheel_acc_max)
        self.passive_steer_rate_max = (
            3.0 * self.steer_rate_max if passive_steer_rate_max is None else float(passive_steer_rate_max)
        )
        self.passive_wheel_acc_max = (
            3.0 * self.wheel_acc_max if passive_wheel_acc_max is None else float(passive_wheel_acc_max)
        )
        self.w_max = float(w_max)
        self.v_max = float(v_max)
        self.a_max = float(a_max)
        self.lsq_reg = float(lsq_reg)
        self.state_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.pi], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.pi], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(BRANCH_BITS))
        self._steer = np.zeros(4, dtype=np.float32)
        self._speed = np.zeros(4, dtype=np.float32)
        self._vel_world = np.zeros(2, dtype=np.float32)
        self._w_body = 0.0

    @property
    def vel_world(self) -> np.ndarray:
        return self._vel_world

    @property
    def w_body(self) -> float:
        return self._w_body

    def reset(self, state: np.ndarray) -> np.ndarray:
        self._steer = np.zeros(4, dtype=np.float32)
        self._speed = np.zeros(4, dtype=np.float32)
        self._vel_world = np.zeros(2, dtype=np.float32)
        self._w_body = 0.0
        return super().reset(state)

    def step(self, action: np.ndarray) -> np.ndarray:
        return self.state


class Diag2ActiveBranchEnv(Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        *,
        dt: float = 0.05,
        t_total: float = 30.0,
        episode_steps: int = 600,
        traj_type: str = "rounded_rect",
        random_traj_on_reset: bool = True,
        mode: str = "rl",
        n_preview: int = 10,
        v_wheel_max: float = 0.35,
        w_max: float = 1.5,
        v_max: float = 0.35,
        a_max: float = 0.35,
        steer_rate_max: Optional[float] = None,
        wheel_acc_max: Optional[float] = None,
        passive_steer_rate_max: Optional[float] = None,
        passive_wheel_acc_max: Optional[float] = None,
        kp_pos_x: float = 1.2,
        kp_pos_y: float = 1.6,
        kd_vel_x: float = 0.6,
        kd_vel_y: float = 0.6,
        kp_yaw: float = 2.2,
        preview_cost_scale: float = 0.2,
        danger_margin_deg: float = 12.0,
        init_pos_noise: float = 0.15,
        init_yaw_noise_deg: float = 12.0,
        init_steer_noise_deg: float = 20.0,
        random_ref_time: bool = True,
        **kwargs,
    ):
        if mode not in {"nearest_angle", "rule_preview", "rl"}:
            raise ValueError("mode must be one of {'nearest_angle', 'rule_preview', 'rl'}")

        self.robot = Diag2ActiveRobot(
            dt=dt,
            v_wheel_max=v_wheel_max,
            steer_rate_max=steer_rate_max,
            wheel_acc_max=wheel_acc_max,
            passive_steer_rate_max=passive_steer_rate_max,
            passive_wheel_acc_max=passive_wheel_acc_max,
            w_max=w_max,
            v_max=v_max,
            a_max=a_max,
        )
        self.context = PathRefContext(
            dt=dt,
            t_total=t_total,
            traj_type=traj_type,
            random_traj_on_reset=random_traj_on_reset,
        )

        self.dt = float(dt)
        self.t_total = float(t_total)
        self.max_episode_steps = int(episode_steps)
        self.mode = mode
        self.n_preview = int(n_preview)
        self.v_max = float(v_max)
        self.w_max = float(w_max)
        self.kp_pos_x = float(kp_pos_x)
        self.kp_pos_y = float(kp_pos_y)
        self.kd_vel_x = float(kd_vel_x)
        self.kd_vel_y = float(kd_vel_y)
        self.kp_yaw = float(kp_yaw)
        self.preview_cost_scale = float(preview_cost_scale)
        self.danger_margin = np.deg2rad(float(danger_margin_deg))
        self.init_pos_noise = float(init_pos_noise)
        self.init_yaw_noise = np.deg2rad(float(init_yaw_noise_deg))
        self.init_steer_noise = np.deg2rad(float(init_steer_noise_deg))
        self.random_ref_time = bool(random_ref_time)

        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * 38, dtype=np.float32),
            high=np.array([np.inf] * 38, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(BRANCH_BITS))
        self.robot.action_space = self.action_space

        self.termination_penalty = 25.0
        self._elapsed_steps = 0
        self._selected_branch_id = 0
        self._baseline_branch_id = 0
        self._preview_best_branch_id = 0
        self._branch_switched = 0.0
        self._prev_branch_id = 0
        self._prev_active_steer = np.zeros(2, dtype=np.float32)
        self._prev_robot_state = np.zeros(3, dtype=np.float32)
        self._prev_speed = np.zeros(4, dtype=np.float32)
        self._preview_results: List[Dict] = []
        self._last_action_branch = 0
        self._reward_breakdown: Dict[str, float] = {}
        self._failure_reason = "none"
        self._branch_switch_count = 0
        self._limit_hit_count = 0
        self.seed()

    @property
    def additional_info(self) -> Dict[str, Dict]:
        return {}

    def get_ref_state(self, t: float) -> np.ndarray:
        return get_ref_state_by_type(t, self.context.traj_type, traj_params=self.context.traj_params)

    def get_ref_horizon(self) -> List[np.ndarray]:
        return [self.get_ref_state(self.context.ref_time + (k + 1) * self.dt) for k in range(self.n_preview)]

    def local_twist_controller(self, state: np.ndarray, ref_state: np.ndarray, vel_world: np.ndarray) -> np.ndarray:
        px, py, yaw = state
        rx, ry, rvx, rvy, ryaw = ref_state
        ex_b, ey_b = world_to_body(px, py, yaw, rx, ry)
        v_ref_bx, v_ref_by = world_to_body(0.0, 0.0, yaw, rvx, rvy)
        v_cur_bx, v_cur_by = world_to_body(0.0, 0.0, yaw, vel_world[0], vel_world[1])
        evx = v_ref_bx - v_cur_bx
        evy = v_ref_by - v_cur_by
        eyaw = wrap_to_pi(ryaw - yaw)

        twist = np.array(
            [
                v_ref_bx + self.kp_pos_x * ex_b + self.kd_vel_x * evx,
                v_ref_by + self.kp_pos_y * ey_b + self.kd_vel_y * evy,
                self.kp_yaw * eyaw,
            ],
            dtype=np.float32,
        )
        twist[:2] = np.clip(twist[:2], -self.v_max, self.v_max)
        twist[2] = np.clip(twist[2], -self.w_max, self.w_max)
        return twist

    def solve_wheel_targets_from_twist(self, twist: np.ndarray) -> Dict[str, np.ndarray]:
        vx, vy, wz = twist
        steer_main = np.zeros(4, dtype=np.float32)
        steer_dual = np.zeros(4, dtype=np.float32)
        speed_main = np.zeros(4, dtype=np.float32)
        speed_dual = np.zeros(4, dtype=np.float32)
        local_v = np.zeros((4, 2), dtype=np.float32)

        for i, name in enumerate(WHEEL_ORDER):
            x_i, y_i = WHEEL_POS[name]
            vix = vx - wz * y_i
            viy = vy + wz * x_i
            local_v[i] = np.array([vix, viy], dtype=np.float32)
            speed = float(np.hypot(vix, viy))
            theta = float(self.robot._steer[i]) if speed < 1e-8 else float(np.arctan2(viy, vix))
            steer_main[i] = theta
            steer_dual[i] = wrap_to_pi(theta + np.pi)
            speed_main[i] = speed
            speed_dual[i] = -speed

        return {
            "steer_main": steer_main,
            "steer_dual": steer_dual,
            "speed_main": speed_main,
            "speed_dual": speed_dual,
            "local_v": local_v,
        }

    def build_branch_targets(
        self, branch_bits: Tuple[int, int], twist: np.ndarray, current_steer: np.ndarray
    ) -> Dict[str, np.ndarray]:
        solved = self.solve_wheel_targets_from_twist(twist)
        steer_targets = np.zeros(4, dtype=np.float32)
        speed_targets = np.zeros(4, dtype=np.float32)

        for local_idx, wheel_idx in enumerate(ACTIVE_WHEEL_INDICES):
            if branch_bits[local_idx] == 0:
                steer_targets[wheel_idx] = solved["steer_main"][wheel_idx]
                speed_targets[wheel_idx] = solved["speed_main"][wheel_idx]
            else:
                steer_targets[wheel_idx] = solved["steer_dual"][wheel_idx]
                speed_targets[wheel_idx] = solved["speed_dual"][wheel_idx]

        for wheel_idx in PASSIVE_WHEEL_INDICES:
            vx_i, vy_i = solved["local_v"][wheel_idx]
            if np.hypot(vx_i, vy_i) < 1e-8:
                steer_targets[wheel_idx] = current_steer[wheel_idx]
                speed_targets[wheel_idx] = 0.0
            else:
                steer_targets[wheel_idx] = wrap_to_pi(np.arctan2(vy_i, vx_i))
                speed_targets[wheel_idx] = float(np.hypot(vx_i, vy_i))

        return {
            "steer_targets": steer_targets.astype(np.float32),
            "wheel_speed_targets": np.clip(speed_targets, -self.robot.v_wheel_max, self.robot.v_wheel_max).astype(
                np.float32
            ),
        }

    def propagate_wheels(
        self,
        steer: np.ndarray,
        speed: np.ndarray,
        steer_targets: np.ndarray,
        speed_targets: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        steer_next = steer.copy()
        speed_next = speed.copy()
        for i in range(4):
            if i in ACTIVE_WHEEL_INDICES:
                steer_rate_lim = self.robot.steer_rate_max
                speed_acc_lim = self.robot.wheel_acc_max
            else:
                steer_rate_lim = self.robot.passive_steer_rate_max
                speed_acc_lim = self.robot.passive_wheel_acc_max

            steer_next[i] = steer[i] + np.clip(
                angle_diff(steer_targets[i], steer[i]), -steer_rate_lim * self.dt, steer_rate_lim * self.dt
            )
            speed_next[i] = speed[i] + np.clip(
                speed_targets[i] - speed[i], -speed_acc_lim * self.dt, speed_acc_lim * self.dt
            )
            steer_next[i] = wrap_to_pi(steer_next[i])
            if i in ACTIVE_WHEEL_INDICES:
                steer_next[i] = np.clip(steer_next[i], -STEER_LIMIT, STEER_LIMIT)
            speed_next[i] = np.clip(speed_next[i], -self.robot.v_wheel_max, self.robot.v_wheel_max)
        return steer_next.astype(np.float32), speed_next.astype(np.float32)

    def estimate_body_twist_from_wheels(self, steer: np.ndarray, speed: np.ndarray) -> np.ndarray:
        a_rows = []
        b_rows = []
        for i, name in enumerate(WHEEL_ORDER):
            x_i, y_i = WHEEL_POS[name]
            c = np.cos(steer[i])
            s = np.sin(steer[i])
            a_rows.append([c, s, -c * y_i + s * x_i])
            b_rows.append(speed[i])
        a_mat = np.array(a_rows, dtype=np.float32)
        b_vec = np.array(b_rows, dtype=np.float32)
        ata = a_mat.T @ a_mat + self.robot.lsq_reg * np.eye(3, dtype=np.float32)
        atb = a_mat.T @ b_vec
        twist = np.linalg.solve(ata, atb)
        twist[:2] = np.clip(twist[:2], -2.0 * self.robot.v_wheel_max, 2.0 * self.robot.v_wheel_max)
        twist[2] = np.clip(twist[2], -self.w_max, self.w_max)
        return twist.astype(np.float32)

    def propagate_robot(
        self,
        state: np.ndarray,
        vel_world: np.ndarray,
        steer: np.ndarray,
        speed: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        px, py, yaw = state
        vx_body, vy_body, w_body = self.estimate_body_twist_from_wheels(steer, speed)
        yaw_mid = yaw + 0.5 * w_body * self.dt
        vx_world, vy_world = body_to_world(yaw_mid, vx_body, vy_body)
        vel_new = np.array([vx_world, vy_world], dtype=np.float32)
        vel_norm = float(np.hypot(vel_new[0], vel_new[1]))
        if vel_norm > self.v_max and vel_norm > 1e-8:
            vel_new *= self.v_max / vel_norm

        delta_v = vel_new - vel_world
        delta_n = float(np.hypot(delta_v[0], delta_v[1]))
        a_limit = self.robot.a_max * self.dt
        if delta_n > a_limit and delta_n > 1e-8:
            delta_v *= a_limit / delta_n
        vel_new = vel_world + delta_v

        px_next = px + vel_new[0] * self.dt
        py_next = py + vel_new[1] * self.dt
        yaw_next = wrap_to_pi(yaw + w_body * self.dt)
        return np.array([px_next, py_next, yaw_next], dtype=np.float32), vel_new.astype(np.float32), float(w_body)

    def active_margins(self, steer: np.ndarray) -> np.ndarray:
        return np.array([float(STEER_LIMIT - abs(steer[i])) for i in ACTIVE_WHEEL_INDICES], dtype=np.float32)

    def baseline_branch_selection(self) -> int:
        twist = self.local_twist_controller(self.robot.state, self.context.state.reference, self.robot.vel_world)
        solved = self.solve_wheel_targets_from_twist(twist)
        bits = []
        for wheel_idx in ACTIVE_WHEEL_INDICES:
            cur = float(self.robot._steer[wheel_idx])
            d_main = abs(angle_diff(solved["steer_main"][wheel_idx], cur))
            d_dual = abs(angle_diff(solved["steer_dual"][wheel_idx], cur))
            bits.append(0 if d_main <= d_dual else 1)
        return BRANCH_BITS.index(tuple(bits))

    def preview_branch(self, branch_id: int, ref_horizon: List[np.ndarray]) -> Dict:
        branch_bits = BRANCH_BITS[branch_id]
        pred_state = self.robot.state.copy()
        pred_steer = self.robot._steer.copy()
        pred_speed = self.robot._speed.copy()
        pred_vel_world = self.robot.vel_world.copy()

        feasible_len = 0
        total_cost = 0.0
        min_margin = float(np.min(self.active_margins(pred_steer)))
        predicted_limit_hit = False
        predicted_flip_risk = 0.0
        first_targets = None
        first_steer_delta = 0.0

        for k, ref_state in enumerate(ref_horizon):
            twist_cmd = self.local_twist_controller(pred_state, ref_state, pred_vel_world)
            targets = self.build_branch_targets(branch_bits, twist_cmd, pred_steer)
            if k == 0:
                first_targets = targets
                for wheel_idx in ACTIVE_WHEEL_INDICES:
                    first_steer_delta += abs(angle_diff(targets["steer_targets"][wheel_idx], pred_steer[wheel_idx]))

            pred_steer, pred_speed = self.propagate_wheels(
                pred_steer, pred_speed, targets["steer_targets"], targets["wheel_speed_targets"]
            )
            pred_state, pred_vel_world, _ = self.propagate_robot(pred_state, pred_vel_world, pred_steer, pred_speed)

            ex_b, ey_b = world_to_body(pred_state[0], pred_state[1], pred_state[2], ref_state[0], ref_state[1])
            v_ref_bx, v_ref_by = world_to_body(0.0, 0.0, pred_state[2], ref_state[2], ref_state[3])
            v_cur_bx, v_cur_by = world_to_body(0.0, 0.0, pred_state[2], pred_vel_world[0], pred_vel_world[1])
            eyaw = wrap_to_pi(ref_state[4] - pred_state[2])
            margins = self.active_margins(pred_steer)
            min_margin = min(min_margin, float(np.min(margins)))
            predicted_flip_risk += float(np.sum(margins < self.danger_margin))

            total_cost += float(
                1.8 * (ex_b**2 + ey_b**2)
                + 0.35 * ((v_ref_bx - v_cur_bx) ** 2 + (v_ref_by - v_cur_by) ** 2)
                + 0.6 * (eyaw**2)
                + 0.12 * np.sum(np.abs(pred_speed[ACTIVE_WHEEL_INDICES]))
                + 0.6 * np.sum(np.square(np.clip(self.danger_margin - margins, 0.0, None)))
            )

            if np.any(np.abs(pred_steer[ACTIVE_WHEEL_INDICES]) >= STEER_LIMIT - 1e-6):
                predicted_limit_hit = True
                break
            feasible_len += 1

        if first_targets is None:
            first_targets = {
                "steer_targets": self.robot._steer.copy(),
                "wheel_speed_targets": self.robot._speed.copy(),
            }
        return {
            "branch_id": int(branch_id),
            "branch_bits": branch_bits,
            "feasible_len": int(feasible_len),
            "total_cost": float(total_cost),
            "min_margin": float(min_margin),
            "predicted_limit_hit": bool(predicted_limit_hit),
            "predicted_flip_risk": float(predicted_flip_risk),
            "first_targets": first_targets,
            "first_steer_delta": float(first_steer_delta),
        }

    def select_branch_rule(self, previews: List[Dict]) -> int:
        best_id = 0
        best_score = -np.inf
        for item in previews:
            score = (
                4.0 * item["feasible_len"]
                + 0.8 * item["min_margin"]
                - item["total_cost"]
                - 6.0 * float(item["predicted_limit_hit"])
                - 0.1 * item["predicted_flip_risk"]
            )
            if score > best_score:
                best_score = score
                best_id = item["branch_id"]
        return int(best_id)

    def _decode_action(self, action) -> int:
        if isinstance(action, np.ndarray):
            action = int(np.asarray(action).reshape(-1)[0])
        return int(np.clip(int(action), 0, len(BRANCH_BITS) - 1))

    def _branch_gap(self, selected_preview: Dict, best_preview: Dict) -> float:
        feasible_gap = float(best_preview["feasible_len"] - selected_preview["feasible_len"])
        cost_gap = max(0.0, float(selected_preview["total_cost"] - best_preview["total_cost"]))
        margin_gap = max(0.0, float(best_preview["min_margin"] - selected_preview["min_margin"]))
        return feasible_gap + 0.5 * cost_gap + 0.5 * margin_gap

    def _compute_preview_results(self) -> List[Dict]:
        ref_horizon = self.get_ref_horizon()
        return [self.preview_branch(i, ref_horizon) for i in range(len(BRANCH_BITS))]

    def _future_infeasible(self, previews: List[Dict]) -> bool:
        return max(item["feasible_len"] for item in previews) <= 0

    def get_obs(self) -> np.ndarray:
        ref = self.context.state.reference
        px, py, yaw = self.robot.state
        ex_b, ey_b = world_to_body(px, py, yaw, ref[0], ref[1])
        ref_vx_b, ref_vy_b = world_to_body(0.0, 0.0, yaw, ref[2], ref[3])
        cur_vx_b, cur_vy_b = world_to_body(0.0, 0.0, yaw, self.robot.vel_world[0], self.robot.vel_world[1])
        evx = ref_vx_b - cur_vx_b
        evy = ref_vy_b - cur_vy_b
        eyaw = wrap_to_pi(ref[4] - yaw)

        active_feats: List[float] = []
        margins = self.active_margins(self.robot._steer)
        for idx, wheel_idx in enumerate(ACTIVE_WHEEL_INDICES):
            theta = self.robot._steer[wheel_idx]
            active_feats.extend(
                [
                    np.cos(theta),
                    np.sin(theta),
                    float(self.robot._speed[wheel_idx] / max(self.robot.v_wheel_max, 1e-6)),
                    float(margins[idx] / max(STEER_LIMIT, 1e-6)),
                ]
            )

        branch_one_hot = np.zeros(len(BRANCH_BITS), dtype=np.float32)
        branch_one_hot[self._selected_branch_id] = 1.0

        preview_feats: List[float] = []
        for item in self._preview_results:
            preview_feats.extend(
                [
                    float(item["feasible_len"] / max(self.n_preview, 1)),
                    float(np.tanh(item["total_cost"] * self.preview_cost_scale)),
                    float(item["min_margin"] / max(STEER_LIMIT, 1e-6)),
                    float(item["first_steer_delta"] / max(STEER_LIMIT, 1e-6)),
                ]
            )

        return np.array(
            [
                ex_b,
                ey_b,
                evx,
                evy,
                eyaw,
                float(ref_vx_b / max(self.v_max, 1e-6)),
                float(ref_vy_b / max(self.v_max, 1e-6)),
                float(np.hypot(ref[2], ref[3]) / max(self.v_max, 1e-6)),
                *active_feats,
                *branch_one_hot.tolist(),
                float(self._branch_switched),
                float(self._prev_branch_id / max(len(BRANCH_BITS) - 1, 1)),
                *preview_feats,
            ],
            dtype=np.float32,
        )

    def _get_obs(self) -> np.ndarray:
        return self.get_obs()

    def compute_reward(self, selected_preview: Dict, best_preview: Dict) -> Tuple[float, Dict[str, float]]:
        ref = self.context.state.reference
        px, py, yaw = self.robot.state
        prev_px, prev_py, prev_yaw = self._prev_robot_state
        ex_b, ey_b = world_to_body(px, py, yaw, ref[0], ref[1])
        ref_vx_b, ref_vy_b = world_to_body(0.0, 0.0, yaw, ref[2], ref[3])
        cur_vx_b, cur_vy_b = world_to_body(0.0, 0.0, yaw, self.robot.vel_world[0], self.robot.vel_world[1])
        evx = ref_vx_b - cur_vx_b
        evy = ref_vy_b - cur_vy_b
        eyaw = wrap_to_pi(ref[4] - yaw)

        # Tracking remains the dominant term: position, velocity, and heading all matter.
        r_track = -1.45 * (ex_b**2 + ey_b**2) - 0.30 * (evx**2 + evy**2) - 0.55 * (eyaw**2)

        # Reward forward progress along the reference tangent instead of merely surviving.
        ref_dir = np.array([np.cos(ref[4]), np.sin(ref[4])], dtype=np.float32)
        robot_delta = np.array([px - prev_px, py - prev_py], dtype=np.float32)
        progress_along_ref = float(np.dot(robot_delta, ref_dir))
        backward_pen = float(max(0.0, -progress_along_ref))
        r_progress = 0.90 * progress_along_ref - 0.35 * backward_pen

        margins = self.active_margins(self.robot._steer)
        near_pen = float(np.sum(np.square(np.clip(self.danger_margin - margins, 0.0, None))))
        limit_hit = float(np.any(np.abs(self.robot._steer[ACTIVE_WHEEL_INDICES]) >= STEER_LIMIT - 1e-6))
        # Penalize large yaw mismatch with velocity direction to suppress side-slip-like motion.
        vel_norm = float(np.hypot(self.robot.vel_world[0], self.robot.vel_world[1]))
        vel_heading = float(np.arctan2(self.robot.vel_world[1], self.robot.vel_world[0])) if vel_norm > 1e-6 else yaw
        vel_heading_err = float(abs(wrap_to_pi(vel_heading - yaw)))
        r_safe = -1.2 * near_pen - 8.0 * limit_hit - 0.22 * (vel_heading_err**2)

        gap = self._branch_gap(selected_preview, best_preview)
        r_branch = (
            0.12 * float(selected_preview["feasible_len"] / max(self.n_preview, 1))
            + 0.08 * float(selected_preview["min_margin"] / max(STEER_LIMIT, 1e-6))
            - 0.06 * gap
            - 0.18 * float(selected_preview["predicted_limit_hit"])
        )

        steer_change = float(
            np.sum(
                [
                    abs(angle_diff(self.robot._steer[idx], self._prev_active_steer[local_idx]))
                    for local_idx, idx in enumerate(ACTIVE_WHEEL_INDICES)
                ]
            )
            / max(STEER_LIMIT, 1e-6)
        )
        speed_change = float(np.mean(np.abs(self.robot._speed - self._prev_speed)) / max(self.robot.v_wheel_max, 1e-6))
        wheel_effort = float(np.mean(np.abs(self.robot._speed)) / max(self.robot.v_wheel_max, 1e-6))
        r_control = -0.08 * wheel_effort - 0.10 * speed_change
        r_smooth = -0.15 * steer_change - 0.08 * float(self._branch_switched)
        r_survive = 0.03
        total = float(r_track + r_progress + r_safe + r_branch + r_control + r_smooth + r_survive)

        return total, {
            "track": float(r_track),
            "progress": float(r_progress),
            "safe": float(r_safe),
            "branch": float(r_branch),
            "control": float(r_control),
            "smooth": float(r_smooth),
            "survive": float(r_survive),
            "total": float(total),
        }

    def _get_reward(self, action: np.ndarray) -> float:
        return float(self._reward_breakdown.get("total", 0.0))

    def _get_terminated(self) -> bool:
        if self._elapsed_steps >= self.max_episode_steps:
            self._failure_reason = "timeout"
            return True

        ref = self.context.state.reference
        px, py = self.robot.state[:2]
        if float(np.hypot(px - ref[0], py - ref[1])) > 2.5:
            self._failure_reason = "track_error"
            return True

        if np.any(np.abs(self.robot._steer[ACTIVE_WHEEL_INDICES]) >= STEER_LIMIT - 1e-6):
            self._failure_reason = "steer_limit"
            return True

        if self._future_infeasible(self._preview_results):
            self._failure_reason = "all_branches_infeasible"
            return True

        self._failure_reason = "none"
        return False

    def _build_info(self) -> dict:
        track_error = float(
            np.hypot(
                *world_to_body(
                    self.robot.state[0],
                    self.robot.state[1],
                    self.robot.state[2],
                    self.context.state.reference[0],
                    self.context.state.reference[1],
                )
            )
        )
        return {
            "state": self._state,
            "selected_branch_id": int(self._selected_branch_id),
            "selected_branch_bits": np.array(BRANCH_BITS[self._selected_branch_id], dtype=np.int32),
            "baseline_branch_id": int(self._baseline_branch_id),
            "preview_best_branch_id": int(self._preview_best_branch_id),
            "last_action_branch": int(self._last_action_branch),
            "feasible_lens": np.array([p["feasible_len"] for p in self._preview_results], dtype=np.float32),
            "total_costs": np.array([p["total_cost"] for p in self._preview_results], dtype=np.float32),
            "min_margins": np.array([p["min_margin"] for p in self._preview_results], dtype=np.float32),
            "predicted_limit_hits": np.array(
                [float(p["predicted_limit_hit"]) for p in self._preview_results], dtype=np.float32
            ),
            "branch_switched": float(self._branch_switched),
            "active_wheel_angles": self.robot._steer[ACTIVE_WHEEL_INDICES].copy(),
            "active_wheel_margins": self.active_margins(self.robot._steer),
            "track_error": track_error,
            "reward_breakdown": dict(self._reward_breakdown),
            "failure_reason": self._failure_reason,
            "branch_switch_count": int(self._branch_switch_count),
            "limit_hit_count": int(self._limit_hit_count),
            "traj_type": self.context.traj_type,
            "traj_params": None if self.context.traj_params is None else dict(self.context.traj_params),
        }

    def _get_info(self) -> dict:
        return self._build_info()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        init_state: Optional[Sequence[float]] = None,
        ref_time: Optional[float] = None,
    ) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self.seed(seed)

        if ref_time is None:
            if self.random_ref_time:
                ref_time = float(self.np_random.uniform(0.0, max(self.t_total - 2.0, 0.0)))
            else:
                ref_time = 0.0

        traj_type = None
        if options is not None:
            traj_type = options.get("traj_type", None)

        context_state = self.context.reset(ref_time=ref_time, traj_type=traj_type)
        ref = context_state.reference

        if init_state is None:
            dx = float(self.np_random.uniform(-self.init_pos_noise, self.init_pos_noise))
            dy = float(self.np_random.uniform(-self.init_pos_noise, self.init_pos_noise))
            dyaw = float(self.np_random.uniform(-self.init_yaw_noise, self.init_yaw_noise))
            init_state = [float(ref[0] - 0.7 + dx), float(ref[1] - 0.4 + dy), float(ref[4] + dyaw)]
        else:
            init_state = np.asarray(init_state, dtype=np.float32)

        self._state = State(robot_state=self.robot.reset(np.asarray(init_state, dtype=np.float32)), context_state=context_state)

        self.robot._steer[ACTIVE_WHEEL_INDICES] = self.np_random.uniform(
            -self.init_steer_noise, self.init_steer_noise, size=len(ACTIVE_WHEEL_INDICES)
        ).astype(np.float32)
        self.robot._steer[PASSIVE_WHEEL_INDICES] = 0.0
        self.robot._speed[:] = 0.0
        self.robot._vel_world[:] = 0.0
        self.robot._w_body = 0.0

        self._elapsed_steps = 0
        self._selected_branch_id = 0
        self._baseline_branch_id = 0
        self._preview_best_branch_id = 0
        self._prev_branch_id = 0
        self._branch_switched = 0.0
        self._prev_active_steer = self.robot._steer[ACTIVE_WHEEL_INDICES].copy()
        self._prev_robot_state = self.robot.state.copy()
        self._prev_speed = self.robot._speed.copy()
        self._preview_results = self._compute_preview_results()
        self._reward_breakdown = {
            "track": 0.0,
            "progress": 0.0,
            "safe": 0.0,
            "branch": 0.0,
            "control": 0.0,
            "smooth": 0.0,
            "survive": 0.0,
            "total": 0.0,
        }
        self._last_action_branch = 0
        self._failure_reason = "none"
        self._branch_switch_count = 0
        self._limit_hit_count = 0
        return self.get_obs(), self._build_info()

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        self._baseline_branch_id = self.baseline_branch_selection()
        current_previews = self._compute_preview_results()
        self._preview_results = current_previews
        self._preview_best_branch_id = self.select_branch_rule(current_previews)

        if self.mode == "nearest_angle":
            selected_branch_id = self._baseline_branch_id
        elif self.mode == "rule_preview":
            selected_branch_id = self._preview_best_branch_id
        else:
            selected_branch_id = self._decode_action(action)

        self._last_action_branch = selected_branch_id
        prev_branch_id = self._selected_branch_id
        self._selected_branch_id = selected_branch_id
        self._branch_switched = float(prev_branch_id != selected_branch_id)
        if self._branch_switched > 0.5:
            self._branch_switch_count += 1

        selected_preview = current_previews[selected_branch_id]
        best_preview = current_previews[self._preview_best_branch_id]

        steer_targets = selected_preview["first_targets"]["steer_targets"]
        speed_targets = selected_preview["first_targets"]["wheel_speed_targets"]
        self._prev_robot_state = self.robot.state.copy()
        self._prev_active_steer = self.robot._steer[ACTIVE_WHEEL_INDICES].copy()
        self._prev_speed = self.robot._speed.copy()

        self.robot._steer, self.robot._speed = self.propagate_wheels(
            self.robot._steer, self.robot._speed, steer_targets, speed_targets
        )
        self.robot.state, self.robot._vel_world, self.robot._w_body = self.propagate_robot(
            self.robot.state, self.robot.vel_world, self.robot._steer, self.robot._speed
        )

        if np.any(np.abs(self.robot._steer[ACTIVE_WHEEL_INDICES]) >= STEER_LIMIT - 1e-6):
            self._limit_hit_count += 1

        context_state_next = self.context.step()
        self._state = State(robot_state=self.robot.state.copy(), context_state=context_state_next)
        self._elapsed_steps += 1
        self._prev_branch_id = prev_branch_id

        self._preview_results = self._compute_preview_results()
        reward, self._reward_breakdown = self.compute_reward(selected_preview, best_preview)
        done = self._get_terminated()
        if done:
            reward -= self.termination_penalty
            self._reward_breakdown["total"] = float(reward)

        return self.get_obs(), float(reward), bool(done), self._build_info()

    def render(self, mode: str = "human"):
        import matplotlib.pyplot as plt
        import matplotlib.patches as pc

        plt.ion()
        fig = plt.figure(num=0, figsize=(6.4, 6.4))
        plt.clf()
        px, py, yaw = self.robot.state
        ref = self.context.state.reference

        ax = plt.axes(xlim=(px - 4, px + 4), ylim=(py - 4, py + 4))
        ax.set_aspect("equal")
        ax.grid(True)

        veh_length = 0.48
        veh_width = 0.35
        x_offset = veh_length / 2.0 * np.cos(yaw) - veh_width / 2.0 * np.sin(yaw)
        y_offset = veh_length / 2.0 * np.sin(yaw) + veh_width / 2.0 * np.cos(yaw)
        ax.add_patch(
            pc.Rectangle(
                (px - x_offset, py - y_offset),
                veh_length,
                veh_width,
                angle=np.rad2deg(yaw),
                facecolor="w",
                edgecolor="r",
                zorder=2,
            )
        )

        wheel_len = 0.08
        wheel_rad = 0.025
        for i, name in enumerate(WHEEL_ORDER):
            x_i, y_i = WHEEL_POS[name]
            theta = yaw + self.robot._steer[i]
            wx = px + np.cos(yaw) * x_i - np.sin(yaw) * y_i
            wy = py + np.sin(yaw) * x_i + np.cos(yaw) * y_i
            dx = 0.5 * wheel_len * np.cos(theta)
            dy = 0.5 * wheel_len * np.sin(theta)
            color = "tab:blue" if i in ACTIVE_WHEEL_INDICES else "tab:gray"
            ax.add_patch(pc.Circle((wx, wy), radius=wheel_rad, color=color, zorder=4))
            ax.plot([wx - dx, wx + dx], [wy - dy, wy + dy], color=color, linewidth=2, zorder=5)

        ax.add_patch(pc.Circle((ref[0], ref[1]), radius=0.05, color="red", zorder=3))
        ax.arrow(ref[0], ref[1], 0.3 * np.cos(ref[4]), 0.3 * np.sin(ref[4]), color="tab:red", width=0.01)
        ax.set_title(f"Diag2Active Branch Env | mode={self.mode} | traj={self.context.traj_type}")
        plt.tight_layout()

        if mode == "rgb_array":
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.pause(0.01)
            return image
        plt.pause(0.01)
        plt.show(block=False)


def env_creator(**kwargs):
    return Diag2ActiveBranchEnv(**kwargs)


if __name__ == "__main__":
    env = Diag2ActiveBranchEnv(mode="rl", traj_type="rounded_rect")
    obs, info = env.reset(seed=0)
    for _ in range(20):
        obs, rew, done, info = env.step(env.action_space.sample())
        print("branch", info["selected_branch_id"], "reward", rew, "track_error", info["track_error"])
        if done:
            obs, info = env.reset(seed=0)
