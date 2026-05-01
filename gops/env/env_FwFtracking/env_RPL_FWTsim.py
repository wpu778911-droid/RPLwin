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
WHEEL_POS = {
    "FR": (+0.24, -0.175),
    "RR": (-0.24, -0.175),
    "RL": (-0.24, +0.175),
    "FL": (+0.24, +0.175),
}
WHEEL_ORDER = ("FR", "RR", "RL", "FL")


def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def clip(x: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, low), high)


def _circular_mean(angles: np.ndarray) -> float:
    return float(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))))


def _angle_rms_to(angles: np.ndarray, target: float) -> float:
    err = wrap_to_pi(angles - target)
    return float(np.sqrt(np.mean(err**2)))


def _soft_gauss(err: float, scale: float) -> float:
    x = float(err) / max(float(scale), 1e-6)
    return float(np.exp(-0.5 * min(x * x, 60.0)))


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
        x = -RECT_W + CORNER_R + s
        y = -RECT_H
        dx, dy = BALL_SPEED, 0.0
        return x, y, dx, dy
    s -= straight_x

    if s < arc:
        theta = -0.5 * np.pi + s / CORNER_R
        cx, cy = RECT_W - CORNER_R, -RECT_H + CORNER_R
        x = cx + CORNER_R * np.cos(theta)
        y = cy + CORNER_R * np.sin(theta)
        dx = BALL_SPEED * (-np.sin(theta))
        dy = BALL_SPEED * (np.cos(theta))
        return x, y, dx, dy
    s -= arc

    if s < straight_y:
        x = RECT_W
        y = -RECT_H + CORNER_R + s
        dx, dy = 0.0, BALL_SPEED
        return x, y, dx, dy
    s -= straight_y

    if s < arc:
        theta = 0.0 + s / CORNER_R
        cx, cy = RECT_W - CORNER_R, RECT_H - CORNER_R
        x = cx + CORNER_R * np.cos(theta)
        y = cy + CORNER_R * np.sin(theta)
        dx = BALL_SPEED * (-np.sin(theta))
        dy = BALL_SPEED * (np.cos(theta))
        return x, y, dx, dy
    s -= arc

    if s < straight_x:
        x = RECT_W - CORNER_R - s
        y = RECT_H
        dx, dy = -BALL_SPEED, 0.0
        return x, y, dx, dy
    s -= straight_x

    if s < arc:
        theta = 0.5 * np.pi + s / CORNER_R
        cx, cy = -RECT_W + CORNER_R, RECT_H - CORNER_R
        x = cx + CORNER_R * np.cos(theta)
        y = cy + CORNER_R * np.sin(theta)
        dx = BALL_SPEED * (-np.sin(theta))
        dy = BALL_SPEED * (np.cos(theta))
        return x, y, dx, dy
    s -= arc

    if s < straight_y:
        x = -RECT_W
        y = RECT_H - CORNER_R - s
        dx, dy = 0.0, -BALL_SPEED
        return x, y, dx, dy
    s -= straight_y

    theta = np.pi + s / CORNER_R
    cx, cy = -RECT_W + CORNER_R, -RECT_H + CORNER_R
    x = cx + CORNER_R * np.cos(theta)
    y = cy + CORNER_R * np.sin(theta)
    dx = BALL_SPEED * (-np.sin(theta))
    dy = BALL_SPEED * (np.cos(theta))
    return x, y, dx, dy


def ball_rect_traj(t: float, t_total: float) -> Tuple[float, float]:
    period = _rounded_rect_perimeter() / BALL_SPEED
    s = (t % period) * BALL_SPEED
    x, y, _, _ = _rounded_rect_pos_vel(s)
    return x, y


def ball_rect_vel(t: float, t_total: float) -> Tuple[float, float]:
    period = _rounded_rect_perimeter() / BALL_SPEED
    s = (t % period) * BALL_SPEED
    _, _, dx, dy = _rounded_rect_pos_vel(s)
    return dx, dy


def _line_pos_vel(
    t: float,
    direction: Tuple[float, float],
    start: Tuple[float, float],
    speed: float = BALL_SPEED,
) -> Tuple[float, float, float, float]:
    dx, dy = direction
    norm = float(np.hypot(dx, dy))
    if norm < 1e-6:
        ux, uy = 1.0, 0.0
    else:
        ux, uy = dx / norm, dy / norm
    return (
        float(start[0] + speed * t * ux),
        float(start[1] + speed * t * uy),
        float(speed * ux),
        float(speed * uy),
    )


def _circle_pos_vel(t: float, radius: float = 1.2, speed: float = BALL_SPEED) -> Tuple[float, float, float, float]:
    omega = speed / max(radius, 1e-6)
    theta = omega * t
    return (
        float(radius * np.cos(theta)),
        float(radius * np.sin(theta)),
        float(-radius * omega * np.sin(theta)),
        float(radius * omega * np.cos(theta)),
    )


def _s_curve_pos_vel(t: float, speed: float = BALL_SPEED) -> Tuple[float, float, float, float]:
    x_start = -1.8
    amp = 0.55
    wavelength = 2.4
    k = 2.0 * np.pi / wavelength
    x = x_start + speed * t
    phase = k * (x - x_start)
    y = amp * np.sin(phase)
    dx = speed
    dy = amp * k * np.cos(phase) * dx
    return float(x), float(y), float(dx), float(dy)


def get_ref_state_by_type(t: float, t_total: float, traj_type: str = "rounded_rect") -> Tuple[float, float, float, float]:
    traj = str(traj_type or "rounded_rect").lower()
    if traj in {"rounded_rect", "round_rect", "rect", "default"}:
        bx, by = ball_rect_traj(t, t_total)
        bdx, bdy = ball_rect_vel(t, t_total)
        return bx, by, bdx, bdy
    if traj in {"line_forward", "line_x", "straight", "straight_x"}:
        return _line_pos_vel(t, direction=(1.0, 0.0), start=(-1.8, 0.0))
    if traj in {"line_lateral", "line_y", "lateral", "straight_y"}:
        return _line_pos_vel(t, direction=(0.0, 1.0), start=(0.0, -1.8))
    if traj in {"line_diagonal", "diag", "diagonal"}:
        return _line_pos_vel(t, direction=(1.0, 1.0), start=(-1.3, -1.3))
    if traj in {"circle", "circular"}:
        return _circle_pos_vel(t)
    if traj in {"s_curve", "sine", "s"}:
        return _s_curve_pos_vel(t)
    raise ValueError(
        "traj_type must be one of rounded_rect, line_forward, line_lateral, "
        "line_diagonal, circle, s_curve."
    )


def compute_scan_follow_target(
    bx: float,
    by: float,
    bdx: float,
    bdy: float,
    follow_dist: float,
) -> Tuple[float, float]:
    # Right-side normal of target velocity direction: n_r = [sin(phi), -cos(phi)]
    speed = np.hypot(bdx, bdy)
    if speed > 1e-6:
        ux, uy = bdx / speed, bdy / speed
    else:
        ux, uy = 1.0, 0.0
    n_r = np.array([uy, -ux], dtype=np.float32)
    rx = bx + follow_dist * n_r[0]
    ry = by + follow_dist * n_r[1]
    return float(rx), float(ry)


def world_to_body(px: float, py: float, yaw: float, xw: float, yw: float) -> Tuple[float, float]:
    dx = xw - px
    dy = yw - py
    c = np.cos(yaw)
    s = np.sin(yaw)
    xb = c * dx + s * dy
    yb = -s * dx + c * dy
    return float(xb), float(yb)


def huber_loss(x: float, delta: float = 0.25) -> float:
    ax = abs(float(x))
    if ax <= delta:
        return 0.5 * ax * ax
    return delta * (ax - 0.5 * delta)


def smoothstep(x: float, lo: float, hi: float) -> float:
    z = float(np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0))
    return z * z * (3.0 - 2.0 * z)


class ToRwheelsimRobot(Robot):
    def __init__(
        self,
        dt: float,
        v_wheel_max: float,
        steer_rate_max: Optional[float],
        wheel_acc_max: Optional[float],
        w_max: float,
        v_max: float,
        a_max: float,
        lsq_reg: float = 1e-4,
    ):
        self.dt = dt
        self.v_wheel_max = v_wheel_max
        self.steer_rate_max = STEER_RATE_MAX if steer_rate_max is None else steer_rate_max
        self.wheel_acc_max = WHEEL_ACC_MAX if wheel_acc_max is None else wheel_acc_max
        self.w_max = w_max
        self.v_max = v_max
        self.a_max = a_max
        self.lsq_reg = lsq_reg
        self.state_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.pi], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.pi], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-self.steer_rate_max, -self.wheel_acc_max] * 4, dtype=np.float32),
            high=np.array([self.steer_rate_max, self.wheel_acc_max] * 4, dtype=np.float32),
            dtype=np.float32,
        )
        self._steer = {name: 0.0 for name in WHEEL_ORDER}
        self._speed = {name: 0.0 for name in WHEEL_ORDER}
        self._prev_steer = {name: 0.0 for name in WHEEL_ORDER}
        self._vel_world = np.zeros(2, dtype=np.float32)
        self._w_body = 0.0
        self.last_flip_count = 0
        self.last_flip_triggered = False

    @property
    def vel_world(self) -> np.ndarray:
        return self._vel_world

    @property
    def w_body(self) -> float:
        return self._w_body

    def reset(self, state: np.ndarray) -> np.ndarray:
        self._steer = {name: 0.0 for name in WHEEL_ORDER}
        self._speed = {name: 0.0 for name in WHEEL_ORDER}
        self._prev_steer = {name: 0.0 for name in WHEEL_ORDER}
        self._vel_world = np.zeros(2, dtype=np.float32)
        self._w_body = 0.0
        self._step_count = 0
        self.last_flip_count = 0
        self.last_flip_triggered = False
        return super().reset(state)

    def _maybe_flip(self, steer: float, speed: float, prev_steer: float) -> Tuple[float, float, bool]:
        # Traditional steering-limit shell is kept as a safety guard.
        # RL is rewarded to reduce dependence on this shell rather than remove it.
        steer = wrap_to_pi(steer)
        steer = float(np.clip(steer, -STEER_LIMIT, STEER_LIMIT))
        steer_alt = wrap_to_pi(steer + np.pi)

        margin = np.deg2rad(5.0)
        near_limit = abs(steer) > (STEER_LIMIT - margin)
        flip_cooldown_steps = int(max(1.0, 0.3 / self.dt))
        if not hasattr(self, "_flip_counter"):
            self._flip_counter = 0
        can_flip = self._flip_counter == 0
        flipped = False

        if can_flip and near_limit:
            steer = steer_alt
            speed = -speed
            self._flip_counter = flip_cooldown_steps
            flipped = True
        elif self._flip_counter > 0:
            self._flip_counter -= 1

        steer = float(np.clip(steer, -STEER_LIMIT, STEER_LIMIT))
        return steer, speed, flipped

    def step(self, action: np.ndarray) -> np.ndarray:
        action = clip(action, self.action_space.low, self.action_space.high)
        px, py, yaw = self.state
        if not hasattr(self, "_step_count"):
            self._step_count = 0
        self._step_count += 1

        flip_count = 0
        for i, name in enumerate(WHEEL_ORDER):
            d_steer = float(action[2 * i])
            d_speed = float(action[2 * i + 1])

            steer = self._steer[name] + d_steer * self.dt
            speed = self._speed[name] + d_speed * self.dt
            speed = float(np.clip(speed, -self.v_wheel_max, self.v_wheel_max))

            steer, speed, flipped = self._maybe_flip(steer, speed, self._prev_steer[name])
            if flipped:
                flip_count += 1

            self._steer[name] = steer
            self._speed[name] = speed
            self._prev_steer[name] = steer

        A_rows = []
        b_rows = []
        for name in WHEEL_ORDER:
            x_i, y_i = WHEEL_POS[name]
            steer = self._steer[name]
            speed = self._speed[name]
            c = np.cos(steer)
            s = np.sin(steer)
            A_rows.append([c, s, -c * y_i + s * x_i])
            b_rows.append(speed)

        A = np.array(A_rows, dtype=np.float32)
        b = np.array(b_rows, dtype=np.float32)
        AtA = A.T @ A + self.lsq_reg * np.eye(3, dtype=np.float32)
        Atb = A.T @ b
        vx_body, vy_body, w_body = np.linalg.solve(AtA, Atb)

        v_body_limit = 2.0 * self.v_wheel_max
        vx_body = float(np.clip(vx_body, -v_body_limit, v_body_limit))
        vy_body = float(np.clip(vy_body, -v_body_limit, v_body_limit))
        w_body = float(np.clip(w_body, -self.w_max, self.w_max))

        yaw_mid = yaw + 0.5 * w_body * self.dt
        cos_yaw = np.cos(yaw_mid)
        sin_yaw = np.sin(yaw_mid)
        vx_world = vx_body * cos_yaw - vy_body * sin_yaw
        vy_world = vx_body * sin_yaw + vy_body * cos_yaw
        vel_new = np.array([vx_world, vy_world], dtype=np.float32)
        speed_world = float(np.hypot(vel_new[0], vel_new[1]))
        if speed_world > self.v_max and speed_world > 1e-9:
            vel_new = vel_new * (self.v_max / speed_world)

        delta_v = vel_new - self._vel_world
        delta_norm = float(np.hypot(delta_v[0], delta_v[1]))
        a_limit = self.a_max * self.dt
        if delta_norm > a_limit and delta_norm > 1e-9:
            delta_v = delta_v * (a_limit / delta_norm)
        vel_new = self._vel_world + delta_v
        self._vel_world = vel_new
        self._w_body = w_body

        px_next = px + self._vel_world[0] * self.dt
        py_next = py + self._vel_world[1] * self.dt
        yaw_next = wrap_to_pi(yaw + self._w_body * self.dt)
        self.state = np.array([px_next, py_next, yaw_next], dtype=np.float32)

        self.last_flip_count = int(flip_count)
        self.last_flip_triggered = bool(flip_count > 0)
        return self.state


class BallRefContext(Context):
    def __init__(self, dt: float, t_total: float, traj_type: str = "rounded_rect"):
        self.dt = dt
        self.t_total = t_total
        self.traj_type = str(traj_type or "rounded_rect")
        self.ref_time = 0.0
        self.state = ContextState(reference=np.zeros(4, dtype=np.float32), t=0)

    def reference_at(self, ref_time: float) -> Tuple[float, float, float, float]:
        return get_ref_state_by_type(ref_time, self.t_total, self.traj_type)

    def sample_ref_time(self, np_random) -> float:
        traj = self.traj_type.lower()
        if traj in {"rounded_rect", "round_rect", "rect", "default"}:
            period = _rounded_rect_perimeter() / BALL_SPEED
            return float(period * np_random.uniform(0.0, 1.0))
        if traj in {"circle", "circular"}:
            period = 2.0 * np.pi * 1.2 / BALL_SPEED
            return float(period * np_random.uniform(0.0, 1.0))
        return 0.0

    def reset(self, ref_time: float = 0.0, traj_type: Optional[str] = None) -> ContextState[np.ndarray]:
        if traj_type is not None:
            self.traj_type = str(traj_type)
        self.ref_time = ref_time
        bx, by, bdx, bdy = self.reference_at(self.ref_time)
        self.state = ContextState(reference=np.array([bx, by, bdx, bdy], dtype=np.float32), t=0)
        return self.state

    def step(self) -> ContextState[np.ndarray]:
        self.ref_time += self.dt
        bx, by, bdx, bdy = self.reference_at(self.ref_time)
        self.state = ContextState(reference=np.array([bx, by, bdx, bdy], dtype=np.float32), t=0)
        return self.state

    def get_zero_state(self) -> ContextState[np.ndarray]:
        return ContextState(reference=np.zeros(4, dtype=np.float32), t=0)


class ToRwheelsimSeqScanEnv(Env):
    termination_penalty = 50.0
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        *,
        dt: float = 0.05,
        t_total: float = 30.0,
        episode_steps: int = 1500,
        follow_dist: float = 0.3,
        v_wheel_max: float = 0.3,
        w_max: float = 1.5,
        v_max: float = 0.3,
        a_max: float = 0.3,
        steer_rate_max: Optional[float] = None,
        wheel_acc_max: Optional[float] = None,
        action_horizon: int = 3,
        safety_margin_ratio: float = 0.12,
        traj_type: str = "rounded_rect",
        yaw_huber_weight: float = 5.5,
        yaw_cos_weight: float = 3.0,
        yaw_progress_weight: float = 3.0,
        mode_reward_weight: float = 0.28,
        mode_motion_weight: float = 0.10,
        mode2_proto_scale: float = 1.0,
        mode1_proto_peak_weight: float = 3.4,
        mode1_target_pen_weight: float = 1.15,
        mode1_spread_pen_weight: float = 0.65,
        mode1_speed_pen_weight: float = 0.25,
        mode1_safe_target_weight: float = 0.18,
        mode1_safe_speed_weight: float = 0.10,
        mode1_safe_steer_weight: float = 0.08,
        plan_shift_weight: float = 0.0,
        include_prev_plan_in_obs: bool = False,
        prev_plan_obs_steps: int = 2,
        temporal_ensemble_alpha: float = 0.0,
        temporal_ensemble_decay: float = 0.5,
        **kwargs,
    ):
        self.robot = ToRwheelsimRobot(
            dt=dt,
            v_wheel_max=v_wheel_max,
            steer_rate_max=steer_rate_max,
            wheel_acc_max=wheel_acc_max,
            w_max=w_max,
            v_max=v_max,
            a_max=a_max,
        )
        self.context = BallRefContext(dt=dt, t_total=t_total, traj_type=traj_type)
        self.traj_type = str(traj_type or "rounded_rect")
        self.dt = dt
        self.t_total = t_total
        self.follow_dist = follow_dist
        self.max_episode_steps = int(episode_steps)
        self.v_max = float(v_max)
        self.w_max = float(w_max)
        self.action_horizon = int(action_horizon)
        self.action_dim_per_step = 8
        if self.action_horizon < 1:
            raise ValueError("action_horizon must be at least 1.")
        self.yaw_huber_weight = float(yaw_huber_weight)
        self.yaw_cos_weight = float(yaw_cos_weight)
        self.yaw_progress_weight = float(yaw_progress_weight)
        self.mode_reward_weight = float(mode_reward_weight)
        self.mode_motion_weight = float(mode_motion_weight)
        self.mode2_proto_scale = float(mode2_proto_scale)
        self.mode1_proto_peak_weight = float(mode1_proto_peak_weight)
        self.mode1_target_pen_weight = float(mode1_target_pen_weight)
        self.mode1_spread_pen_weight = float(mode1_spread_pen_weight)
        self.mode1_speed_pen_weight = float(mode1_speed_pen_weight)
        self.mode1_safe_target_weight = float(mode1_safe_target_weight)
        self.mode1_safe_speed_weight = float(mode1_safe_speed_weight)
        self.mode1_safe_steer_weight = float(mode1_safe_steer_weight)
        self.plan_shift_weight = float(plan_shift_weight)
        self.include_prev_plan_in_obs = bool(include_prev_plan_in_obs)
        self.prev_plan_obs_steps = int(max(0, min(prev_plan_obs_steps, self.action_horizon - 1)))
        if not self.include_prev_plan_in_obs:
            self.prev_plan_obs_steps = 0
        self.temporal_ensemble_alpha = float(
            np.clip(temporal_ensemble_alpha, 0.0, 0.95)
        )
        self.temporal_ensemble_decay = float(
            np.clip(temporal_ensemble_decay, 0.0, 1.0)
        )

        # Obs layout (32D):
        # A(4): B and R in body frame
        # B(1): yaw tracking error from reference target direction
        # C(2): target velocity in body frame
        # D(3): chassis vx, vy, omega (body)
        # E(6): future R1/R2/R3 in body frame
        # F(16): per-wheel [cos(steer), sin(steer), norm_speed, margin]
        # Optional tail plan: previous unexecuted planned actions, normalized.
        obs_dim = 32 + self.prev_plan_obs_steps * self.action_dim_per_step
        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * obs_dim, dtype=np.float32),
            high=np.array([np.inf] * obs_dim, dtype=np.float32),
            dtype=np.float32,
        )

        low_step = np.array([-self.robot.steer_rate_max, -self.robot.wheel_acc_max] * 4, dtype=np.float32)
        high_step = np.array([self.robot.steer_rate_max, self.robot.wheel_acc_max] * 4, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.tile(low_step, self.action_horizon),
            high=np.tile(high_step, self.action_horizon),
            dtype=np.float32,
        )

        self.init_dist = 1.0
        self._safety_margin_ratio = float(safety_margin_ratio)
        self._debug_reward_terms: Dict[str, float] = {}
        self._last_exec_action = np.zeros(8, dtype=np.float32)
        self._prev_action_seq = np.zeros(
            (self.action_horizon, self.action_dim_per_step), dtype=np.float32
        )
        self._prev_action_seq_valid = False
        self._action_plan_history: List[np.ndarray] = []
        self._last_temporal_ensemble_sources = 1
        self._last_temporal_ensemble_delta_pen = 0.0
        self._prev_ref_pos_err = 0.0
        self._prev_ref_vel_err = 0.0
        self._prev_yaw_err = 0.0
        self._yaw_ref_prev = 0.0
        self._prev_w_body = 0.0
        self._prev_speed_sign = np.zeros(4, dtype=np.float32)
        self._flip_total = 0
        self._last_mode_alpha_1 = 0.5
        self._last_mode_alpha_2 = 0.5
        self._elapsed_steps = 0
        self.seed()

    def _split_action(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = clip(action, self.action_space.low, self.action_space.high)
        action_seq = action.reshape(self.action_horizon, self.action_dim_per_step)
        # This base env executes only the first 8D action. If action_horizon > 1,
        # the remaining planned actions can be used as a receding-horizon plan.
        action_exec = action_seq[0].copy()
        return action_seq, action_exec

    def _action_norm(self) -> np.ndarray:
        return np.array([
            self.robot.steer_rate_max,
            self.robot.wheel_acc_max,
            self.robot.steer_rate_max,
            self.robot.wheel_acc_max,
            self.robot.steer_rate_max,
            self.robot.wheel_acc_max,
            self.robot.steer_rate_max,
            self.robot.wheel_acc_max,
        ], dtype=np.float32)

    def _temporal_ensemble_action(
        self, action_seq: np.ndarray, action_first: np.ndarray
    ) -> np.ndarray:
        alpha = self.temporal_ensemble_alpha
        if alpha <= 0.0 or not self._action_plan_history:
            self._last_temporal_ensemble_sources = 1
            self._last_temporal_ensemble_delta_pen = 0.0
            return action_first.copy()

        preds = []
        weights = []
        for age, past_seq in enumerate(self._action_plan_history, start=1):
            if age >= self.action_horizon:
                break
            preds.append(past_seq[age].copy())
            weights.append(self.temporal_ensemble_decay ** (age - 1))

        if not preds or float(np.sum(weights)) <= 1e-9:
            self._last_temporal_ensemble_sources = 1
            self._last_temporal_ensemble_delta_pen = 0.0
            return action_first.copy()

        weights_arr = np.asarray(weights, dtype=np.float32)
        weights_arr = weights_arr / max(float(np.sum(weights_arr)), 1e-9)
        history_action = np.sum(
            np.stack(preds, axis=0) * weights_arr[:, None], axis=0
        )
        action_exec = (1.0 - alpha) * action_first + alpha * history_action
        action_exec = clip(action_exec, self.robot.action_space.low, self.robot.action_space.high)

        diff = (action_exec - action_first) / np.maximum(self._action_norm(), 1e-6)
        self._last_temporal_ensemble_sources = 1 + len(preds)
        self._last_temporal_ensemble_delta_pen = float(np.sum(diff**2))
        return action_exec.astype(np.float32)

    def _target_and_ref_world(self) -> Tuple[float, float, float, float, float, float, float]:
        px, py, yaw = self.robot.state
        bx, by, bdx, bdy = self.context.state.reference
        rx, ry = compute_scan_follow_target(bx, by, bdx, bdy, self.follow_dist)
        yaw_ref = np.arctan2(by - py, bx - px)
        return float(bx), float(by), float(bdx), float(bdy), float(rx), float(ry), float(yaw_ref)

    def _task_geometry(self) -> Dict[str, float]:
        px, py, yaw = self.robot.state
        bx, by, bdx, bdy = self.context.state.reference
        speed = float(np.hypot(bdx, bdy))
        if speed > 1e-6:
            ux, uy = bdx / speed, bdy / speed
        else:
            ux, uy = 1.0, 0.0
        t_hat = np.array([ux, uy], dtype=np.float32)
        n_r = np.array([uy, -ux], dtype=np.float32)
        rx = float(bx + self.follow_dist * n_r[0])
        ry = float(by + self.follow_dist * n_r[1])
        x_R_body, y_R_body = world_to_body(px, py, yaw, rx, ry)
        x_B_body, y_B_body = world_to_body(px, py, yaw, bx, by)
        c = np.cos(yaw)
        s = np.sin(yaw)
        v_ref_bx = float(c * bdx + s * bdy)
        v_ref_by = float(-s * bdx + c * bdy)
        v_cur_bx = float(c * self.robot.vel_world[0] + s * self.robot.vel_world[1])
        v_cur_by = float(-s * self.robot.vel_world[0] + c * self.robot.vel_world[1])
        ref_pos_err = float(np.hypot(x_R_body, y_R_body) / max(self.follow_dist, 1e-6))
        ref_vel_err = float(np.hypot(v_cur_bx - v_ref_bx, v_cur_by - v_ref_by) / max(self.v_max, 1e-6))
        yaw_face_raw = float(np.arctan2(by - py, bx - px))
        if not hasattr(self, "_yaw_ref_prev"):
            self._yaw_ref_prev = yaw_face_raw
        speed_scale = float(np.clip(speed / max(self.v_max, 1e-6), 0.0, 1.0))
        beta = 0.12 + 0.18 * speed_scale
        dyaw = wrap_to_pi(yaw_face_raw - self._yaw_ref_prev)
        yaw_face = float(wrap_to_pi(self._yaw_ref_prev + beta * dyaw))
        yaw_err = float(wrap_to_pi(yaw - yaw_face))
        return {
            "bx": float(bx),
            "by": float(by),
            "bdx": float(bdx),
            "bdy": float(bdy),
            "rx": rx,
            "ry": ry,
            "x_B_body": x_B_body,
            "y_B_body": y_B_body,
            "x_R_body": x_R_body,
            "y_R_body": y_R_body,
            "v_ref_bx": v_ref_bx,
            "v_ref_by": v_ref_by,
            "v_cur_bx": v_cur_bx,
            "v_cur_by": v_cur_by,
            "ref_pos_err": ref_pos_err,
            "ref_vel_err": ref_vel_err,
            "yaw_face_raw": yaw_face_raw,
            "yaw_face": yaw_face,
            "ux": float(ux),
            "uy": float(uy),
            "nr_x": float(n_r[0]),
            "nr_y": float(n_r[1]),
        }

    def _future_ref_preview_body(self, steps: int = 3) -> np.ndarray:
        px, py, yaw = self.robot.state
        preview = []
        for k in range(1, steps + 1):
            t_k = self.context.ref_time + k * self.dt
            bx_k, by_k, bdx_k, bdy_k = self.context.reference_at(t_k)
            rx_k, ry_k = compute_scan_follow_target(bx_k, by_k, bdx_k, bdy_k, self.follow_dist)
            xr_k, yr_k = world_to_body(px, py, yaw, rx_k, ry_k)
            preview.extend([xr_k, yr_k])
        return np.array(preview, dtype=np.float32)

    def _compute_wheel_metrics(self) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        px, py, yaw = self.robot.state
        vx_w, vy_w = self.robot.vel_world
        c = np.cos(yaw)
        s = np.sin(yaw)
        vx_b = vx_w * c + vy_w * s
        vy_b = -vx_w * s + vy_w * c
        w_b = self.robot.w_body
        slip = []
        util = []
        margins = []
        vmax = max(self.robot.v_wheel_max, 1e-6)
        for name in WHEEL_ORDER:
            x_i, y_i = WHEEL_POS[name]
            steer = self.robot._steer[name]
            speed = self.robot._speed[name]
            wheel_vx = vx_b - w_b * y_i
            wheel_vy = vy_b + w_b * x_i
            slip_i = (-np.sin(steer) * wheel_vx + np.cos(steer) * wheel_vy) / vmax
            util_i = abs(speed) / vmax
            margin_i = (STEER_LIMIT - abs(steer)) / max(STEER_LIMIT, 1e-6)
            slip.append(float(slip_i))
            util.append(float(util_i))
            margins.append(float(margin_i))
        if abs(vx_b) < 1e-6:
            vx_b = 1e-6
        sideslip = float(np.arctan2(vy_b, vx_b))
        return (
            np.array(slip, dtype=np.float32),
            np.array(util, dtype=np.float32),
            sideslip,
            np.array(margins, dtype=np.float32),
        )

    def _get_obs(self) -> np.ndarray:
        px, py, yaw = self.robot.state
        vx_w, vy_w = self.robot.vel_world
        w_b = self.robot.w_body
        geom = self._task_geometry()
        x_B_body = geom["x_B_body"]
        y_B_body = geom["y_B_body"]
        x_R_body = geom["x_R_body"]
        y_R_body = geom["y_R_body"]
        yaw_ref = geom["yaw_face"]
        e_psi = wrap_to_pi(yaw - yaw_ref)
        v_Bx_body = geom["v_ref_bx"]
        v_By_body = geom["v_ref_by"]
        v_x_body = geom["v_cur_bx"]
        v_y_body = geom["v_cur_by"]

        preview_body = self._future_ref_preview_body(steps=3)
        _, _, _, margins = self._compute_wheel_metrics()

        wheel_feats = []
        for i, name in enumerate(WHEEL_ORDER):
            steer = self.robot._steer[name]
            speed = self.robot._speed[name]
            wheel_feats.extend(
                [
                    np.cos(steer),
                    np.sin(steer),
                    speed / max(self.robot.v_wheel_max, 1e-6),
                    margins[i],
                ]
            )

        prev_plan_feats = []
        if self.prev_plan_obs_steps > 0:
            if self._prev_action_seq_valid:
                tail = self._prev_action_seq[1 : 1 + self.prev_plan_obs_steps]
            else:
                tail = np.zeros(
                    (0, self.action_dim_per_step), dtype=np.float32
                )
            if tail.shape[0] < self.prev_plan_obs_steps:
                pad = np.zeros(
                    (self.prev_plan_obs_steps - tail.shape[0], self.action_dim_per_step),
                    dtype=np.float32,
                )
                tail = np.vstack([tail, pad])
            tail_norm = tail / np.maximum(self._action_norm(), 1e-6)
            prev_plan_feats = np.clip(tail_norm, -1.0, 1.0).reshape(-1).tolist()

        obs = np.array(
            [
                x_B_body,
                y_B_body,
                x_R_body,
                y_R_body,
                e_psi,
                v_Bx_body,
                v_By_body,
                v_x_body,
                v_y_body,
                w_b,
                *preview_body.tolist(),
                *wheel_feats,
                *prev_plan_feats,
            ],
            dtype=np.float32,
        )
        return obs

    def _mode_proto_1(self, tangent_body: np.ndarray, steer_angles: np.ndarray, speeds: np.ndarray) -> float:
        # Mode-1: all wheels should collapse to the same tangent-aligned steering basin.
        tan = tangent_body / (np.linalg.norm(tangent_body) + 1e-6)
        target_angle = float(np.arctan2(tan[1], tan[0]))
        mean_angle = _circular_mean(steer_angles)
        target_rms = _angle_rms_to(steer_angles, target_angle)
        spread_rms = _angle_rms_to(steer_angles, mean_angle)
        speed_rms = float(np.sqrt(np.mean((speeds - np.mean(speeds)) ** 2)))

        target_score = _soft_gauss(target_rms, np.deg2rad(18.0))
        target_dense_score = 1.0 / (
            1.0 + (target_rms / max(np.deg2rad(24.0), 1e-6)) ** 2
        )
        spread_score = _soft_gauss(spread_rms, np.deg2rad(5.5))
        speed_score = _soft_gauss(speed_rms, 0.08)

        target_pen = min((target_rms / max(np.deg2rad(45.0), 1e-6)) ** 1.25, 3.0)
        coherence_score = 0.65 * spread_score + 0.35 * speed_score
        target_mix = 0.65 * target_score + 0.35 * target_dense_score
        peak = target_mix * coherence_score
        miss = (
            self.mode1_target_pen_weight * target_pen
            + self.mode1_spread_pen_weight * (1.0 - spread_score)
            + self.mode1_speed_pen_weight * (1.0 - speed_score)
        )
        return float(self.mode1_proto_peak_weight * peak - miss)

    def _mode_proto_2(
        self,
        tangent_body: np.ndarray,
        steer_angles: np.ndarray,
        speeds: np.ndarray,
        mode2_focus: float,
    ) -> float:
        # Mode-2: each side must be coherent, with opposite deviations around
        # the tangent-aligned Mode-1 steering direction.
        tan = tangent_body / (np.linalg.norm(tangent_body) + 1e-6)
        base_angle = float(np.arctan2(tan[1], tan[0]))
        g12 = _circular_mean(steer_angles[[0, 1]])
        g34 = _circular_mean(steer_angles[[2, 3]])

        group12_rms = _angle_rms_to(steer_angles[[0, 1]], g12)
        group34_rms = _angle_rms_to(steer_angles[[2, 3]], g34)
        internal_rms = 0.5 * (group12_rms + group34_rms)

        d12 = float(wrap_to_pi(g12 - base_angle))
        d34 = float(wrap_to_pi(g34 - base_angle))
        symmetry_err = abs(float(wrap_to_pi(d12 + d34)))
        half_sep = 0.5 * abs(float(wrap_to_pi(d12 - d34)))
        desired_sep = np.deg2rad(18.0 + 14.0 * float(np.clip(mode2_focus, 0.0, 1.0)))

        sp12 = float(abs(speeds[0] - speeds[1]))
        sp34 = float(abs(speeds[2] - speeds[3]))
        group_speed_rms = float(np.sqrt(0.5 * (sp12**2 + sp34**2)))

        internal_score = _soft_gauss(internal_rms, np.deg2rad(5.5))
        symmetry_score = _soft_gauss(symmetry_err, np.deg2rad(10.0))
        sep_center_score = _soft_gauss(abs(half_sep - desired_sep), np.deg2rad(12.0))
        sep_nonzero_score = smoothstep(half_sep, np.deg2rad(8.0), np.deg2rad(18.0))
        speed_score = _soft_gauss(group_speed_rms, 0.09)
        sep_score = float(sep_center_score * sep_nonzero_score)

        peak = (
            internal_score ** 0.35
            * symmetry_score ** 0.30
            * sep_score ** 0.25
            * speed_score ** 0.10
        )
        miss = (
            0.75 * (1.0 - internal_score)
            + 0.55 * (1.0 - symmetry_score)
            + 0.50 * (1.0 - sep_score)
            + 0.30 * (1.0 - speed_score)
        )
        return float(3.2 * peak - miss)

    def _mode_proto_3(self, steer_angles: np.ndarray, speeds: np.ndarray) -> float:
        # Placeholder extensible prototype.
        return float(0.0 - 0.05 * np.mean(speeds**2))

    def _mode_proto_4(self, steer_angles: np.ndarray, speeds: np.ndarray) -> float:
        # Placeholder extensible prototype.
        return float(0.0 - 0.05 * np.var(steer_angles))

    def _mode_gating(self) -> Tuple[float, float, float, float, np.ndarray]:
        # Use a longer heading preview so rounded-corner turns activate Mode-2
        # before the robot is already inside the corner.
        t0 = float(self.context.ref_time)
        lookahead_time = 1.2
        _, _, vx0, vy0 = self.context.reference_at(t0)
        _, _, vx1, vy1 = self.context.reference_at(t0 + lookahead_time)
        heading0 = float(np.arctan2(vy0, vx0))
        heading1 = float(np.arctan2(vy1, vx1))
        turn = abs(wrap_to_pi(heading1 - heading0))
        k = smoothstep(turn, np.deg2rad(2.5), np.deg2rad(13.0))
        k = float(k * k / (k * k + (1.0 - k) * (1.0 - k) + 1e-9))
        alpha1 = 1.0 - k
        alpha2 = k
        alpha3 = 0.1 * (1.0 - k)
        alpha4 = 0.1 * k
        z = alpha1 + alpha2 + alpha3 + alpha4 + 1e-9
        alpha1, alpha2, alpha3, alpha4 = alpha1 / z, alpha2 / z, alpha3 / z, alpha4 / z
        c = np.cos(self.robot.state[2])
        s = np.sin(self.robot.state[2])
        tangent = np.array([c * vx0 + s * vy0, -s * vx0 + c * vy0], dtype=np.float32)
        tangent = tangent / (np.linalg.norm(tangent) + 1e-6)
        return float(alpha1), float(alpha2), float(alpha3), float(alpha4), tangent

    def _compute_reward(
        self,
        action_seq: np.ndarray,
        action_exec: np.ndarray,
        *,
        update_cache: bool = True,
    ) -> Tuple[float, Dict[str, float]]:
        geom = self._task_geometry()
        bx = geom["bx"]
        by = geom["by"]
        pos_err = float(geom["ref_pos_err"])
        vel_err = float(geom["ref_vel_err"])
        yaw_err = float(abs(wrap_to_pi(self.robot.state[2] - geom["yaw_face"])))
        w_pos = 7.5
        w_vel = 1.2
        w_yaw_huber = self.yaw_huber_weight
        w_yaw_cos = self.yaw_cos_weight
        w_prog_pos = 1.2
        w_prog_vel = 0.4
        w_prog_yaw = self.yaw_progress_weight
        r_pos = -w_pos * huber_loss(pos_err, delta=0.30)
        r_vel = -w_vel * huber_loss(vel_err, delta=0.20)
        r_yaw = (
            -w_yaw_huber * huber_loss(yaw_err, delta=0.08)
            -w_yaw_cos * (1.0 - np.cos(yaw_err))
        )
        r_prog = (
            w_prog_pos * (self._prev_ref_pos_err - pos_err)
            + w_prog_vel * (self._prev_ref_vel_err - vel_err)
            + w_prog_yaw * (abs(self._prev_yaw_err) - abs(yaw_err))
        )
        r_task = r_pos + r_vel + r_yaw + r_prog

        steer_angles = np.array([self.robot._steer[n] for n in WHEEL_ORDER], dtype=np.float32)
        speeds = np.array([self.robot._speed[n] / max(self.robot.v_wheel_max, 1e-6) for n in WHEEL_ORDER], dtype=np.float32)
        a1, a2, a3, a4, tangent = self._mode_gating()
        mode_norm = max(a1 + a2, 1e-6)
        a1 = a1 / mode_norm
        a2 = a2 / mode_norm
        m1 = self._mode_proto_1(tangent, steer_angles, speeds)
        m2 = self.mode2_proto_scale * self._mode_proto_2(tangent, steer_angles, speeds, a2)
        mode_raw = float(a1 * m1 + a2 * m2)
        mode1_focus = smoothstep(a1, 0.80, 0.95)
        mode1_target_angle = float(np.arctan2(tangent[1], tangent[0]))
        mode1_target_rms = _angle_rms_to(steer_angles, mode1_target_angle)
        mode1_speed_var = float(np.mean((speeds - np.mean(speeds)) ** 2))
        mode1_mean_angle = _circular_mean(steer_angles)
        mode1_steer_spread = float(
            np.mean(wrap_to_pi(steer_angles - mode1_mean_angle) ** 2)
        )
        self._last_mode_alpha_1 = a1
        self._last_mode_alpha_2 = a2

        slip, _, sideslip, margins = self._compute_wheel_metrics()
        near_penalty = float(np.sum(np.square(np.clip(0.2 - margins, 0.0, None))))
        margin_min = float(np.min(margins))
        slip_rms = float(np.sqrt(np.mean(slip**2)))

        thr = 1.0 - self._safety_margin_ratio
        outward_penalty = 0.0
        for i, name in enumerate(WHEEL_ORDER):
            steer = self.robot._steer[name]
            if abs(steer) > thr * STEER_LIMIT:
                dsteer = float(action_exec[2 * i])
                if steer * dsteer > 0.0:
                    outward_penalty += abs(dsteer) / max(self.robot.steer_rate_max, 1e-6)

        flip_penalty = float(self.robot.last_flip_count)

        cur_sign = np.array([np.sign(self.robot._speed[n]) for n in WHEEL_ORDER], dtype=np.float32)
        sign_changed = (cur_sign != 0.0) & (self._prev_speed_sign != 0.0) & (cur_sign != self._prev_speed_sign)
        reverse_penalty = float(np.sum(sign_changed.astype(np.float32)))

        side_penalty = float(sideslip**2 + 0.5 * np.mean(slip**2))

        # Mode reward should serve tracking quality rather than dominate it.
        task_gate_pos = float(np.clip(1.0 - pos_err / 0.30, 0.0, 1.0))
        task_gate_yaw = float(np.clip(1.0 - yaw_err / 0.18, 0.0, 1.0))
        task_gate = float(0.15 + 0.85 * task_gate_pos * task_gate_yaw)

        safety_gate_margin = float(np.clip((margin_min - 0.08) / 0.32, 0.0, 1.0))
        safety_gate_slip = float(np.clip(1.0 - slip_rms / 0.35, 0.0, 1.0))
        safety_gate_flip = float(1.0 / (1.0 + flip_penalty))
        safety_gate = float(0.20 + 0.80 * safety_gate_margin * safety_gate_slip * safety_gate_flip)

        steer_usage = float(np.mean(np.abs(steer_angles) / max(STEER_LIMIT, 1e-6)))
        steer_rate_usage = float(
            np.mean(np.abs(action_exec[0::2]) / max(self.robot.steer_rate_max, 1e-6))
        )
        mode_motion_penalty = float(
            0.70 * steer_usage**2
            + 0.55 * steer_rate_usage**2
            + 0.35 * outward_penalty
            + 0.45 * flip_penalty
            + 0.30 * slip_rms**2
        )

        r_mode = (
            self.mode_reward_weight * task_gate * safety_gate * mode_raw
            - self.mode_motion_weight * mode_motion_penalty
        )

        r_safe = (
            -0.45 * near_penalty
            -0.25 * outward_penalty
            -0.6 * flip_penalty
            -0.05 * reverse_penalty
            -0.12 * side_penalty
        )
        straight_speed_pen = float(mode1_focus * mode1_speed_var)
        straight_steer_pen = float(
            mode1_focus * mode1_steer_spread / max(np.deg2rad(20.0) ** 2, 1e-6)
        )
        straight_target_pen = float(
            mode1_focus
            * min((mode1_target_rms / max(np.deg2rad(45.0), 1e-6)) ** 2, 4.0)
        )

        act_norm = self._action_norm()
        a0n = action_exec / np.maximum(act_norm, 1e-6)
        mag_pen = float(np.sum(a0n**2))

        da = (action_exec - self._last_exec_action) / np.maximum(act_norm, 1e-6)
        delta_pen = float(np.sum(da**2))

        seq_pen = 0.0
        for k in range(self.action_horizon - 1):
            ak = action_seq[k] / np.maximum(act_norm, 1e-6)
            ak1 = action_seq[k + 1] / np.maximum(act_norm, 1e-6)
            seq_pen += float(np.sum((ak1 - ak) ** 2))

        plan_shift_pen = 0.0
        if (
            self.plan_shift_weight > 0.0
            and self._prev_action_seq_valid
            and self.action_horizon > 1
        ):
            prev_tail = self._prev_action_seq[1:]
            curr_head = action_seq[:-1]
            n_shift = min(prev_tail.shape[0], curr_head.shape[0])
            for k in range(n_shift):
                weight = 1.0 if k == 0 else 0.5
                prev_k = prev_tail[k] / np.maximum(act_norm, 1e-6)
                curr_k = curr_head[k] / np.maximum(act_norm, 1e-6)
                plan_shift_pen += float(weight * np.sum((curr_k - prev_k) ** 2))

        w_body = float(self.robot.w_body)
        yaw_acc = (w_body - self._prev_w_body) / max(self.dt, 1e-6)
        yaw_smooth_pen = float((w_body / max(self.w_max, 1e-6)) ** 2 + 0.2 * yaw_acc**2)

        r_safe += (
            -self.mode1_safe_target_weight * straight_target_pen
            -self.mode1_safe_speed_weight * straight_speed_pen
            -self.mode1_safe_steer_weight * straight_steer_pen
        )

        r_smooth = (
            -0.012 * mag_pen
            -0.015 * delta_pen
            -0.008 * seq_pen
            -self.plan_shift_weight * plan_shift_pen
            -0.004 * yaw_smooth_pen
        )

        total = r_task + r_mode + r_safe + r_smooth
        terms = {
            "reward_task": float(r_task),
            "reward_pos": float(r_pos),
            "reward_vel": float(r_vel),
            "reward_yaw": float(r_yaw),
            "reward_progress": float(r_prog),
            "reward_mode": float(r_mode),
            "reward_safe": float(r_safe),
            "reward_smooth": float(r_smooth),
            "mode_raw": float(mode_raw),
            "mode_proto_1": float(m1),
            "mode_proto_2": float(m2),
            "mode_task_gate": float(task_gate),
            "mode_safety_gate": float(safety_gate),
            "mode_motion_penalty": float(mode_motion_penalty),
            "mode1_focus": float(mode1_focus),
            "straight_speed_pen": float(straight_speed_pen),
            "straight_steer_pen": float(straight_steer_pen),
            "straight_target_pen": float(straight_target_pen),
            "straight_speed_var": float(mode1_speed_var),
            "straight_steer_spread": float(mode1_steer_spread),
            "mode1_target_rms": float(mode1_target_rms),
            "seq_action_pen": float(seq_pen),
            "plan_shift_pen": float(plan_shift_pen),
            "temporal_ensemble_alpha": float(self.temporal_ensemble_alpha),
            "temporal_ensemble_sources": float(self._last_temporal_ensemble_sources),
            "temporal_ensemble_delta_pen": float(self._last_temporal_ensemble_delta_pen),
            "flip_count": float(self.robot.last_flip_count),
            "min_margin": float(margin_min),
            "ref_pos_error": float(pos_err),
            "ref_vel_error": float(vel_err),
            "pos_error": float(pos_err),
            "yaw_error": float(yaw_err),
            "mode_alpha_1": float(a1),
            "mode_alpha_2": float(a2),
            "target_x": float(bx),
            "target_y": float(by),
        }

        if update_cache:
            self._prev_ref_pos_err = pos_err
            self._prev_ref_vel_err = vel_err
            self._prev_yaw_err = yaw_err
            self._yaw_ref_prev = float(geom["yaw_face"])
            self._last_exec_action = action_exec.copy()
            self._prev_action_seq = action_seq.copy()
            self._prev_action_seq_valid = True
            self._action_plan_history.insert(0, action_seq.copy())
            max_history = max(0, self.action_horizon - 1)
            if len(self._action_plan_history) > max_history:
                del self._action_plan_history[max_history:]
            self._prev_speed_sign = cur_sign
            self._prev_w_body = w_body
            self._flip_total += int(self.robot.last_flip_count)
        return float(total), terms

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        init_state: Optional[Sequence[float]] = None,
        ref_time: Optional[float] = None,
    ) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self.seed(seed)

        if options is not None and options.get("traj_type") is not None:
            self.context.traj_type = str(options["traj_type"])

        if ref_time is None:
            ref_time = self.context.sample_ref_time(self.np_random)

        self.traj_type = self.context.traj_type
        context_state = self.context.reset(ref_time=ref_time)
        bx, by, bdx, bdy = context_state.reference
        speed = float(np.hypot(bdx, bdy))
        if speed > 1e-6:
            ux, uy = bdx / speed, bdy / speed
        else:
            ux, uy = 1.0, 0.0
        right_normal = np.array([uy, -ux], dtype=np.float32)

        if init_state is None:
            dist0 = self.follow_dist + self.np_random.uniform(-0.15, 0.15)
            tang_offset = self.np_random.uniform(-0.25, 0.25)
            px0 = bx + dist0 * right_normal[0] + tang_offset * ux
            py0 = by + dist0 * right_normal[1] + tang_offset * uy
            yaw0 = np.arctan2(by - py0, bx - px0)
            yaw0 += self.np_random.uniform(-0.25, 0.25)
            init_state = [px0, py0, yaw0]
        else:
            init_state = np.array(init_state, dtype=np.float32)

        self._state = State(
            robot_state=self.robot.reset(np.array(init_state, dtype=np.float32)),
            context_state=context_state,
        )
        px0, py0, yaw0 = np.array(init_state, dtype=np.float32)
        rx0 = float(bx + self.follow_dist * right_normal[0])
        ry0 = float(by + self.follow_dist * right_normal[1])
        xR_b0, yR_b0 = world_to_body(px0, py0, yaw0, rx0, ry0)
        pos_err0 = float(np.hypot(xR_b0, yR_b0) / max(self.follow_dist, 1e-6))
        c0 = np.cos(yaw0)
        s0 = np.sin(yaw0)
        v_ref_bx0 = float(c0 * bdx + s0 * bdy)
        v_ref_by0 = float(-s0 * bdx + c0 * bdy)
        vel_err0 = float(np.hypot(v_ref_bx0, v_ref_by0) / max(self.v_max, 1e-6))
        yaw_ref = np.arctan2(by - py0, bx - px0)

        # Per-reset caches required by progress/smooth/safety rewards.
        self._prev_ref_pos_err = pos_err0
        self._prev_ref_vel_err = vel_err0
        self._prev_yaw_err = float(abs(wrap_to_pi(init_state[2] - yaw_ref)))
        self._yaw_ref_prev = float(yaw_ref)
        self._last_exec_action = np.zeros(8, dtype=np.float32)
        self._prev_action_seq = np.zeros(
            (self.action_horizon, self.action_dim_per_step), dtype=np.float32
        )
        self._prev_action_seq_valid = False
        self._action_plan_history = []
        self._last_temporal_ensemble_sources = 1
        self._last_temporal_ensemble_delta_pen = 0.0
        self._prev_speed_sign = np.zeros(4, dtype=np.float32)
        self._prev_w_body = 0.0
        self._flip_total = 0
        self._last_mode_alpha_1 = 0.5
        self._last_mode_alpha_2 = 0.5
        self._elapsed_steps = 0
        self._debug_reward_terms = {}

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        # Required step flow:
        # 1) receive action_horizon * 8D action
        # 2) reshape to (action_horizon, 8)
        # 3) execute only first action
        # 4) robot applies traditional steering-limit safety shell
        # 5) update context
        # 6) update env state
        # 7) compute reward
        # 8) return obs, reward, done, info
        action_seq, action_first = self._split_action(action)
        action_exec = self._temporal_ensemble_action(action_seq, action_first)

        robot_state_next = self.robot.step(action_exec)
        context_state_next = self.context.step()
        self._state = State(robot_state=robot_state_next, context_state=context_state_next)

        self._elapsed_steps += 1
        reward, terms = self._compute_reward(action_seq, action_exec, update_cache=True)
        terminated = self._get_terminated()
        if terminated:
            reward -= self.termination_penalty
        self._debug_reward_terms = terms

        obs = self._get_obs()
        info = self._get_info()
        info.update(terms)
        info["flip_total"] = int(self._flip_total)
        info["elapsed_steps"] = int(self._elapsed_steps)
        return obs, reward, terminated, info

    def _get_reward(self, action: np.ndarray) -> float:
        # Unused because step() is overridden to follow the required sequence.
        action_seq, action_first = self._split_action(action)
        action_exec = self._temporal_ensemble_action(action_seq, action_first)
        reward, _ = self._compute_reward(action_seq, action_exec, update_cache=False)
        return reward

    def _get_terminated(self) -> bool:
        px, py = self.robot.state[:2]
        bx, by = self.context.state.reference[:2]
        dist_to_ball = np.hypot(px - bx, py - by)
        timeout = self._elapsed_steps >= self.max_episode_steps
        out_of_range = dist_to_ball > 6.0
        return bool(timeout or out_of_range)

    def render(self, mode: str = "human"):
        import matplotlib.pyplot as plt
        import matplotlib.patches as pc

        plt.ion()
        fig = plt.figure(num=0, figsize=(6.4, 6.4))
        plt.clf()
        px, py, yaw = self.robot.state
        bx, by, bdx, bdy = self.context.state.reference
        rx, ry = compute_scan_follow_target(bx, by, bdx, bdy, self.follow_dist)

        ax = plt.axes(xlim=(px - 4, px + 4), ylim=(py - 4, py + 4))
        ax.set_aspect("equal")

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
        for name in WHEEL_ORDER:
            x_i, y_i = WHEEL_POS[name]
            steer = self.robot._steer[name]
            wx = px + np.cos(yaw) * x_i - np.sin(yaw) * y_i
            wy = py + np.sin(yaw) * x_i + np.cos(yaw) * y_i
            theta = yaw + steer
            dx = 0.5 * wheel_len * np.cos(theta)
            dy = 0.5 * wheel_len * np.sin(theta)
            ax.add_patch(pc.Circle((wx, wy), radius=wheel_rad, color="k", zorder=4))
            ax.plot([wx - dx, wx + dx], [wy - dy, wy + dy], color="k", linewidth=2, zorder=5)

        ax.add_patch(pc.Circle((bx, by), radius=0.05, color="red", zorder=3))
        ax.plot([rx], [ry], marker="x", color="blue", zorder=3)

        ax.set_title("ToRwheelsim Seq-Scan RL Env")
        ax.grid(True)
        plt.tight_layout()

        if mode == "rgb_array":
            fig.canvas.draw()
            image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.pause(0.01)
            return image_from_plot
        if mode == "human":
            plt.pause(0.01)
            plt.show(block=False)


def env_creator(**kwargs):
    return ToRwheelsimSeqScanEnv(**kwargs)


# Compared with env_ToRwheelsim.py:
# 1) task target changed: right-side follow point + heading to the moving target;
# 2) action changed from single-step 8D to 3-step sequence 24D (execute first step only);
# 3) reward changed to task + mode + safety + smooth structure;
# 4) added mode-layer reward with dynamic gating by future trajectory geometry;
# 5) kept traditional steering-limit flip logic as a safety shell, while penalizing over-reliance.
