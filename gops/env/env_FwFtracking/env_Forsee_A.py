from typing import Optional, Sequence, Tuple

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


def compute_follow_target(
    bx: float,
    by: float,
    bdx: float,
    bdy: float,
    follow_dist: float,
    side: float,
) -> Tuple[float, float, float]:
    speed = np.hypot(bdx, bdy)
    if speed > 1e-6:
        ux, uy = bdx / speed, bdy / speed
    else:
        ux, uy = 1.0, 0.0
    perp = np.array([-uy, ux])
    pnorm = np.hypot(perp[0], perp[1])
    if pnorm < 1e-6:
        perp_unit = np.array([0.0, 1.0])
    else:
        perp_unit = perp / pnorm
    rx = bx + perp_unit[0] * follow_dist * side
    ry = by + perp_unit[1] * follow_dist * side
    yaw_ref = np.arctan2(by - ry, bx - rx)
    return rx, ry, yaw_ref


class ForseeARobot(Robot):
    def __init__(
        self,
        dt: float,
        v_wheel_max: float,
        steer_rate_max: Optional[float],
        wheel_acc_max: Optional[float],
        w_max: float,
        v_max: float,
        a_max: float,
        lookahead: int = 8,
        lsq_reg: float = 1e-4,
        w_steer: float = 1.0,
        w_speed: float = 0.5,
        w_limit: float = 2.0,
        w_rate: float = 1.0,
    ):
        self.dt = dt
        self.v_wheel_max = v_wheel_max
        self.steer_rate_max = STEER_RATE_MAX if steer_rate_max is None else steer_rate_max
        self.wheel_acc_max = WHEEL_ACC_MAX if wheel_acc_max is None else wheel_acc_max
        self.w_max = w_max
        self.v_max = v_max
        self.a_max = a_max
        self.lookahead = int(np.clip(lookahead, 6, 10))
        self.lsq_reg = lsq_reg
        self.w_steer = w_steer
        self.w_speed = w_speed
        self.w_limit = w_limit
        self.w_rate = w_rate
        self.state_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.pi], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.pi], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-self.v_max, -self.v_max, -self.w_max], dtype=np.float32),
            high=np.array([self.v_max, self.v_max, self.w_max], dtype=np.float32),
            dtype=np.float32,
        )
        self._steer = {name: 0.0 for name in WHEEL_ORDER}
        self._speed = {name: 0.0 for name in WHEEL_ORDER}
        self._vel_world = np.zeros(2, dtype=np.float32)
        self._w_body = 0.0
        self._vel_body_cmd = np.zeros(2, dtype=np.float32)

    @property
    def vel_world(self) -> np.ndarray:
        return self._vel_world

    @property
    def w_body(self) -> float:
        return self._w_body

    def reset(self, state: np.ndarray) -> np.ndarray:
        self._steer = {name: 0.0 for name in WHEEL_ORDER}
        self._speed = {name: 0.0 for name in WHEEL_ORDER}
        self._vel_world = np.zeros(2, dtype=np.float32)
        self._w_body = 0.0
        self._vel_body_cmd = np.zeros(2, dtype=np.float32)
        return super().reset(state)

    def _smooth_body_vel(self, target: np.ndarray) -> np.ndarray:
        delta = target - self._vel_body_cmd
        delta_norm = float(np.hypot(delta[0], delta[1]))
        a_limit = self.a_max * self.dt
        if delta_norm > a_limit and delta_norm > 1e-9:
            delta = delta * (a_limit / delta_norm)
        self._vel_body_cmd = self._vel_body_cmd + delta
        return self._vel_body_cmd.copy()

    def _predict_body_vel_seq(self, target: np.ndarray) -> np.ndarray:
        seq = np.zeros((self.lookahead, 2), dtype=np.float32)
        tmp = self._vel_body_cmd.copy()
        for k in range(self.lookahead):
            delta = target - tmp
            delta_norm = float(np.hypot(delta[0], delta[1]))
            a_limit = self.a_max * self.dt
            if delta_norm > a_limit and delta_norm > 1e-9:
                delta = delta * (a_limit / delta_norm)
            tmp = tmp + delta
            seq[k] = tmp
        return seq

    def _branch_cost(
        self,
        steer_seq: np.ndarray,
        speed_seq: np.ndarray,
        prev_steer: float,
        prev_speed: float,
    ) -> float:
        cost = 0.0
        last_steer = prev_steer
        last_speed = prev_speed
        margin = np.deg2rad(5.0)
        for k in range(steer_seq.shape[0]):
            steer = steer_seq[k]
            speed = speed_seq[k]
            dsteer = wrap_to_pi(steer - last_steer)
            dspeed = speed - last_speed
            cost += self.w_steer * float(dsteer ** 2) + self.w_speed * float(dspeed ** 2)

            over = max(0.0, abs(steer) - (STEER_LIMIT - margin))
            cost += self.w_limit * float(over ** 2)

            steer_rate = abs(dsteer) / max(self.dt, 1e-9)
            if steer_rate > self.steer_rate_max:
                cost += self.w_rate * float((steer_rate - self.steer_rate_max) ** 2)
            acc = abs(dspeed) / max(self.dt, 1e-9)
            if acc > self.wheel_acc_max:
                cost += self.w_rate * float((acc - self.wheel_acc_max) ** 2)

            last_steer = steer
            last_speed = speed
        return cost

    def step(self, action: np.ndarray) -> np.ndarray:
        action = clip(action, self.action_space.low, self.action_space.high)
        vx_des, vy_des, w_des = float(action[0]), float(action[1]), float(action[2])
        w_cmd = float(np.clip(w_des, -self.w_max, self.w_max))
        vx_des = float(np.clip(vx_des, -self.v_max, self.v_max))
        vy_des = float(np.clip(vy_des, -self.v_max, self.v_max))
        v_cmd = self._smooth_body_vel(np.array([vx_des, vy_des], dtype=np.float32))
        v_seq = self._predict_body_vel_seq(np.array([vx_des, vy_des], dtype=np.float32))

        # ---- prelook branch selection ----
        for name in WHEEL_ORDER:
            x_i, y_i = WHEEL_POS[name]
            prev_steer = self._steer[name]
            prev_speed = self._speed[name]

            steer_seq = np.zeros(self.lookahead, dtype=np.float32)
            speed_seq = np.zeros(self.lookahead, dtype=np.float32)
            for k in range(self.lookahead):
                vx_body, vy_body = v_seq[k]
                v_ix = vx_body - w_cmd * y_i
                v_iy = vy_body + w_cmd * x_i
                speed = float(np.hypot(v_ix, v_iy))
                if speed < 1e-6:
                    steer = steer_seq[k - 1] if k > 0 else prev_steer
                    speed = 0.0
                else:
                    steer = float(np.arctan2(v_iy, v_ix))
                steer_seq[k] = wrap_to_pi(steer)
                speed_seq[k] = speed

            steer_seq_flip = wrap_to_pi(steer_seq + np.pi)
            speed_seq_flip = -speed_seq

            cost_norm = self._branch_cost(steer_seq, speed_seq, prev_steer, prev_speed)
            cost_flip = self._branch_cost(steer_seq_flip, speed_seq_flip, prev_steer, prev_speed)
            use_flip = cost_flip < cost_norm

            v_ix_now = v_cmd[0] - w_cmd * y_i
            v_iy_now = v_cmd[1] + w_cmd * x_i
            speed_now = float(np.hypot(v_ix_now, v_iy_now))
            if speed_now < 1e-6:
                steer_now = prev_steer
                speed_now = 0.0
            else:
                steer_now = float(np.arctan2(v_iy_now, v_ix_now))
            if use_flip:
                steer_now = wrap_to_pi(steer_now + np.pi)
                speed_now = -speed_now
            else:
                steer_now = wrap_to_pi(steer_now)

            # ---- rate limits ----
            dsteer = wrap_to_pi(steer_now - prev_steer)
            max_dsteer = self.steer_rate_max * self.dt
            dsteer = float(np.clip(dsteer, -max_dsteer, max_dsteer))
            steer = wrap_to_pi(prev_steer + dsteer)
            steer = float(np.clip(steer, -STEER_LIMIT, STEER_LIMIT))

            dspeed = speed_now - prev_speed
            max_dspeed = self.wheel_acc_max * self.dt
            dspeed = float(np.clip(dspeed, -max_dspeed, max_dspeed))
            speed = prev_speed + dspeed
            speed = float(np.clip(speed, -self.v_wheel_max, self.v_wheel_max))

            self._steer[name] = steer
            self._speed[name] = speed

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

        vx_body, vy_body, w_body = np.linalg.lstsq(A, b, rcond=None)[0]
        vx_body = float(np.clip(vx_body, -self.v_max, self.v_max))
        vy_body = float(np.clip(vy_body, -self.v_max, self.v_max))
        w_body = float(np.clip(w_body, -self.w_max, self.w_max))

        px, py, yaw = self.state
        yaw_mid = yaw + 0.5 * w_body * self.dt
        cos_yaw = np.cos(yaw_mid)
        sin_yaw = np.sin(yaw_mid)
        vx_world = vx_body * cos_yaw - vy_body * sin_yaw
        vy_world = vx_body * sin_yaw + vy_body * cos_yaw
        self._vel_world = np.array([vx_world, vy_world], dtype=np.float32)
        self._w_body = w_body

        px_next = px + self._vel_world[0] * self.dt
        py_next = py + self._vel_world[1] * self.dt
        yaw_next = wrap_to_pi(yaw + self._w_body * self.dt)
        self.state = np.array([px_next, py_next, yaw_next], dtype=np.float32)
        return self.state


class BallRefContext(Context):
    def __init__(self, dt: float, t_total: float):
        self.dt = dt
        self.t_total = t_total
        self.ref_time = 0.0
        self.state = ContextState(reference=np.zeros(4, dtype=np.float32), t=0)

    def reset(self, ref_time: float = 0.0) -> ContextState[np.ndarray]:
        self.ref_time = ref_time
        bx, by = ball_rect_traj(self.ref_time, self.t_total)
        bdx, bdy = ball_rect_vel(self.ref_time, self.t_total)
        self.state = ContextState(reference=np.array([bx, by, bdx, bdy], dtype=np.float32), t=0)
        return self.state

    def step(self) -> ContextState[np.ndarray]:
        self.ref_time += self.dt
        bx, by = ball_rect_traj(self.ref_time, self.t_total)
        bdx, bdy = ball_rect_vel(self.ref_time, self.t_total)
        self.state = ContextState(reference=np.array([bx, by, bdx, bdy], dtype=np.float32), t=0)
        return self.state

    def get_zero_state(self) -> ContextState[np.ndarray]:
        return ContextState(reference=np.zeros(4, dtype=np.float32), t=0)


class ForseeAEnv(Env):
    termination_penalty = 50.0

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
        lookahead: int = 8,
        **kwargs,
    ):
        self.robot = ForseeARobot(
            dt=dt,
            v_wheel_max=v_wheel_max,
            steer_rate_max=steer_rate_max,
            wheel_acc_max=wheel_acc_max,
            w_max=w_max,
            v_max=v_max,
            a_max=a_max,
            lookahead=lookahead,
        )
        self.context = BallRefContext(dt=dt, t_total=t_total)
        self.dt = dt
        self.t_total = t_total
        self.follow_dist = follow_dist
        self.max_episode_steps = int(episode_steps)

        self.state_dim = 3
        obs_dim = 7
        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * obs_dim, dtype=np.float32),
            high=np.array([np.inf] * obs_dim, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = self.robot.action_space
        self.side = -1.0
        self.init_dist = 1.0
        self._yaw_ref_prev = 0.0
        self._prev_action = np.zeros(3, dtype=np.float32)
        self._last_ref_time = None
        self._step_count = 0
        self.seed()

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
            period = _rounded_rect_perimeter() / BALL_SPEED
            ref_time = period * self.np_random.uniform(0.0, 1.0)
        context_state = self.context.reset(ref_time=ref_time)

        bx, by, bdx, bdy = context_state.reference
        if init_state is None:
            base_angle = -3.0 * np.pi / 4.0
            angle = base_angle + self.np_random.uniform(-0.2, 0.2)
            px0 = bx + self.init_dist * np.cos(angle)
            py0 = by + self.init_dist * np.sin(angle)
            yaw0 = np.arctan2(by - py0, bx - px0)
            init_state = [px0, py0, yaw0]
        else:
            init_state = np.array(init_state, dtype=np.float32)

        self._state = State(
            robot_state=self.robot.reset(np.array(init_state, dtype=np.float32)),
            context_state=context_state,
        )
        rx, ry, yaw_ref = compute_follow_target(bx, by, bdx, bdy, self.follow_dist, self.side)
        self._yaw_ref_prev = yaw_ref
        self._prev_action = np.zeros(3, dtype=np.float32)
        self._last_ref_time = self.context.ref_time
        self._step_count = 0
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        self._step_count += 1
        return super().step(action)

    def _get_obs(self) -> np.ndarray:
        px, py, yaw = self.robot.state
        vx, vy = self.robot.vel_world
        bx, by, bdx, bdy = self.context.state.reference
        rx, ry, yaw_ref = self._get_ref_target()
        rel_ref = np.array([px - rx, py - ry], dtype=np.float32)
        rel_ball = np.array([px - bx, py - by], dtype=np.float32)
        vel_err = np.array([vx - bdx, vy - bdy], dtype=np.float32)
        yaw_err = np.array([wrap_to_pi(yaw - yaw_ref)], dtype=np.float32)
        return np.concatenate((rel_ref, rel_ball, vel_err, yaw_err), axis=0)

    def _get_reward(self, action: np.ndarray) -> float:
        px, py, yaw = self.robot.state
        vx, vy = self.robot.vel_world
        bx, by, bdx, bdy = self.context.state.reference
        rx, ry, yaw_ref = self._get_ref_target()
        pos_err = float(np.hypot(px - rx, py - ry) / max(self.follow_dist, 1e-6))
        vel_err = float(np.hypot(vx - bdx, vy - bdy) / max(BALL_SPEED, 1e-6))
        yaw_err = abs(wrap_to_pi(yaw - yaw_ref)) / np.pi

        action = clip(action, self.action_space.low, self.action_space.high)
        delta_u = float(np.linalg.norm(action - self._prev_action))
        self._prev_action = np.asarray(action, dtype=np.float32)

        w_pos = 2.0
        w_vel = 0.5
        w_yaw = 0.5
        w_u = 0.1
        return -w_pos * pos_err - w_vel * vel_err - w_yaw * yaw_err - w_u * (delta_u ** 2)

    def _get_ref_target(self) -> Tuple[float, float, float]:
        bx, by, bdx, bdy = self.context.state.reference
        rx, ry, yaw_ref_raw = compute_follow_target(bx, by, bdx, bdy, self.follow_dist, self.side)
        if self._last_ref_time != self.context.ref_time:
            beta = 0.2
            dyaw = wrap_to_pi(yaw_ref_raw - self._yaw_ref_prev)
            self._yaw_ref_prev = wrap_to_pi(self._yaw_ref_prev + beta * dyaw)
            self._last_ref_time = self.context.ref_time
        return rx, ry, self._yaw_ref_prev

    def _get_terminated(self) -> bool:
        px, py = self.robot.state[:2]
        bx, by = self.context.state.reference[:2]
        dist_to_ball = np.hypot(px - bx, py - by)
        return dist_to_ball > 5.0 or self._step_count >= self.max_episode_steps


def env_creator(**kwargs):
    return ForseeAEnv(**kwargs)
