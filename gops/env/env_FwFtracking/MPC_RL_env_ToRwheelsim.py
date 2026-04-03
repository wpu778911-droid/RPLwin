from typing import Optional, Sequence, Tuple

import numpy as np
from gym import spaces

from gops.env.env_gen_ocp.pyth_base import Context, ContextState, Env, Robot, State
#物理约束
STEER_LIMIT = np.deg2rad(170.0)#舵轮角度限制
STEER_RATE_MAX = np.deg2rad(120.0)#舵轮最大转角角速度
WHEEL_ACC_MAX = 1.0#车轮最大加速度
#小球轨迹参数
BALL_SPEED = 0.1#速度
RECT_W = 2.5#轨迹长
RECT_H = 1.5#轨迹宽
CORNER_R = 0.3#四个圆角半径
WHEEL_POS = {
    "FR": (+0.24, -0.175),
    "RR": (-0.24, -0.175),
    "RL": (-0.24, +0.175),
    "FL": (+0.24, +0.175),
}#四舵轮位置
WHEEL_ORDER = ("FR", "RR", "RL", "FL")


def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi
#角度归一化，把角度映射到[-pi, pi]范围内

def clip(x: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, low), high)
#clip函数，把变量限制到指定范围内，超过范围的部分被截断为边界值
def _rounded_rect_perimeter() -> float:
    straight_x = 2.0 * (RECT_W - CORNER_R)
    straight_y = 2.0 * (RECT_H - CORNER_R)
    arc = 0.5 * np.pi * CORNER_R
    return 2.0 * (straight_x + straight_y) + 4.0 * arc
#计算圆角矩形周长，计算小球轨迹的总长
#第40-109行：给定弧长 s，求小球位置和速度
def _rounded_rect_pos_vel(s: float) -> Tuple[float, float, float, float]:
    straight_x = 2.0 * (RECT_W - CORNER_R)
    straight_y = 2.0 * (RECT_H - CORNER_R)
    arc = 0.5 * np.pi * CORNER_R
#根据弧长 s，计算小球在圆角矩形轨迹上的位置 (x, y) 和速度 (dx, dy)。轨迹由四条直线段和四个圆弧段组成，函数通过判断 s 落在哪个段上来计算对应的位置和速度。
    if s < straight_x:
        x = -RECT_W + CORNER_R + s
        y = -RECT_H
        dx, dy = BALL_SPEED, 0.0
        return x, y, dx, dy
    s -= straight_x
#在底边直线段运动时候的位置和速度
    if s < arc:
        theta = -0.5 * np.pi + s / CORNER_R
        cx, cy = RECT_W - CORNER_R, -RECT_H + CORNER_R
        x = cx + CORNER_R * np.cos(theta)
        y = cy + CORNER_R * np.sin(theta)
        dx = BALL_SPEED * (-np.sin(theta))
        dy = BALL_SPEED * (np.cos(theta))
        return x, y, dx, dy
    s -= arc
#在右边圆弧段运动时候的位置和速度
    if s < straight_y:
        x = RECT_W
        y = -RECT_H + CORNER_R + s
        dx, dy = 0.0, BALL_SPEED
        return x, y, dx, dy
    s -= straight_y
#在右边直线段运动时候的位置和速度
    if s < arc:
        theta = 0.0 + s / CORNER_R
        cx, cy = RECT_W - CORNER_R, RECT_H - CORNER_R
        x = cx + CORNER_R * np.cos(theta)
        y = cy + CORNER_R * np.sin(theta)
        dx = BALL_SPEED * (-np.sin(theta))
        dy = BALL_SPEED * (np.cos(theta))
        return x, y, dx, dy
    s -= arc
#在上边圆弧段运动时候的位置和速度
    if s < straight_x:
        x = RECT_W - CORNER_R - s
        y = RECT_H
        dx, dy = -BALL_SPEED, 0.0
        return x, y, dx, dy
    s -= straight_x
#在左边直线段运动时候的位置和速度
    if s < arc:
        theta = 0.5 * np.pi + s / CORNER_R
        cx, cy = -RECT_W + CORNER_R, RECT_H - CORNER_R
        x = cx + CORNER_R * np.cos(theta)
        y = cy + CORNER_R * np.sin(theta)
        dx = BALL_SPEED * (-np.sin(theta))
        dy = BALL_SPEED * (np.cos(theta))
        return x, y, dx, dy
    s -= arc
#在左边圆弧段运动时候的位置和速度
    if s < straight_y:
        x = -RECT_W
        y = RECT_H - CORNER_R - s
        dx, dy = 0.0, -BALL_SPEED
        return x, y, dx, dy
    s -= straight_y
#在底边圆弧段运动时候的位置和速度
    theta = np.pi + s / CORNER_R
    cx, cy = -RECT_W + CORNER_R, -RECT_H + CORNER_R
    x = cx + CORNER_R * np.cos(theta)
    y = cy + CORNER_R * np.sin(theta)
    dx = BALL_SPEED * (-np.sin(theta))
    dy = BALL_SPEED * (np.cos(theta))
    return x, y, dx, dy
#根据弧长 s，计算小球在圆角矩形轨迹上的位置 (x, y) 和速度 (dx, dy)。轨迹由四条直线段和四个圆弧段组成，函数通过判断 s 落在哪个段上来计算对应的位置和速度。

def ball_rect_traj(t: float, t_total: float) -> Tuple[float, float]:
    period = _rounded_rect_perimeter() / BALL_SPEED
    s = (t % period) * BALL_SPEED
    x, y, _, _ = _rounded_rect_pos_vel(s)
    return x, y
#给定时间 t，计算小球在圆角矩形轨迹上的位置 (x, y)。函数首先计算小球运动一个完整周期所需的时间 period，然后根据当前时间 t 计算小球已经运动的弧长 s，最后调用 _rounded_rect_pos_vel 函数获取对应位置的坐标 (x, y)。

def ball_rect_vel(t: float, t_total: float) -> Tuple[float, float]:
    period = _rounded_rect_perimeter() / BALL_SPEED
    s = (t % period) * BALL_SPEED
    _, _, dx, dy = _rounded_rect_pos_vel(s)
    return dx, dy
#给定时间 t，计算小球在圆角矩形轨迹上的速度 (dx, dy)。函数的计算方式与 ball_rect_traj 类似，但最终返回的是速度分量 dx 和 dy。
#这两个函数共同定义了小球在圆角矩形轨迹上的运动，分别提供了位置和速度信息，供后续的 MPC 规划和环境交互使用。
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
    #求小球速度方向单位向量和垂直方向单位向量
    pnorm = np.hypot(perp[0], perp[1])
    if pnorm < 1e-6:
        perp_unit = np.array([0.0, 1.0])
    else:
        perp_unit = perp / pnorm
    rx = bx + perp_unit[0] * follow_dist * side
    ry = by + perp_unit[1] * follow_dist * side
    yaw_ref = np.arctan2(by - ry, bx - rx)#车头朝向小球
    #生成侧向跟随点和朝向参考，follow_dist 是跟随距离，side 是侧向偏移方向（-1 或 1），rx 和 ry 是机器人应该跟随小球的目标位置，yaw_ref 是机器人应该保持的朝向，使其面向小球。
    return rx, ry, yaw_ref
#compute_follow_target函数计算机器人应该跟随小球的目标位置和朝向。它根据小球的位置 (bx, by) 和速度 (bdx, bdy)，以及期望的跟随距离 follow_dist 和侧向偏移方向 side，计算出机器人应该达到的目标位置 (rx, ry) 和朝向 yaw_ref。函数首先计算小球的单位速度向量，然后计算一个垂直于速度方向的单位向量，最后根据这些信息计算出目标位置和朝向。
class ChassisMPCPlanner:
    def __init__(
        self,
        dt: float,#采样周期
        t_total: float,#小球运动总时间
        horizon: int = 5,#预测步长
        v_max: float = 0.3,#最大速度
        w_max: float = 1.5,#最大角速度
        dv_max: float = 0.15,#最大速度变化率
        dw_max: float = 0.3,#最大角速度变化率
        k_pos: float = 1.2,#位置控制增益
        k_yaw: float = 2.0,#航向控制增益
    ):
        self.dt = float(dt)#采样周期
        self.t_total = float(t_total)#小球运动总时间
        self.horizon = int(horizon)#预测步长
        self.v_max = float(v_max)#最大速度
        self.w_max = float(w_max)#最大角速度
        self.dv_max = float(dv_max)#最大速度变化率
        self.dw_max = float(dw_max)#最大角速度变化率
        self.k_pos = float(k_pos)#位置控制增益
        self.k_yaw = float(k_yaw) #航向控制增益
        self._last_u = np.zeros(3, dtype=np.float32)

    def reset(self) -> None:
        self._last_u = np.zeros(3, dtype=np.float32)

    def _limit_vec_rate(self, v: np.ndarray, v_prev: np.ndarray) -> np.ndarray:
        dv = v - v_prev
        dv_norm = float(np.hypot(dv[0], dv[1]))
        if dv_norm > self.dv_max and dv_norm > 1e-9:
            dv = dv * (self.dv_max / dv_norm)
        return v_prev + dv
#_limit_vec_rate函数限制速度向量的变化率，确保每个时间步的速度变化不超过 dv_max。它计算当前速度 v 与前一速度 v_prev 之间的差值 dv，并根据 dv 的范数来判断是否需要进行缩放，以满足最大变化率的约束。
    def _limit_scalar_rate(self, value: float, prev: float) -> float:
        dv = value - prev
        if abs(dv) > self.dw_max:
            dv = np.sign(dv) * self.dw_max
        return prev + dv
#_limit_scalar_rate函数限制标量值（如角速度）的变化率，确保每个时间步的变化不超过 dw_max。它计算当前值 value 与前一值 prev 之间的差值 dv，并根据 dv 的绝对值来判断是否需要进行缩放，以满足最大变化率的约束。
    def plan(
        self,
        state: np.ndarray,
        ref_time: float,
        follow_dist: float,
        side: float,
    ) -> np.ndarray:
        x, y, yaw = float(state[0]), float(state[1]), float(state[2])
        v_prev = self._last_u[:2].copy()
        w_prev = float(self._last_u[2])
        traj = []
 
        for k in range(self.horizon):
            t_k = ref_time + k * self.dt
            bx, by = ball_rect_traj(t_k, self.t_total)
            bdx, bdy = ball_rect_vel(t_k, self.t_total)
            rx, ry, yaw_ref = compute_follow_target(bx, by, bdx, bdy, follow_dist, side)
#预测未来第k步的小球参考
            to_ref = np.array([rx - x, ry - y], dtype=np.float32)
            v_des = self.k_pos * to_ref
            v_mag = float(np.hypot(v_des[0], v_des[1]))
            if v_mag > self.v_max and v_mag > 1e-9:
                v_des = v_des * (self.v_max / v_mag)
            v_des = self._limit_vec_rate(v_des, v_prev)
#根据当前位置和目标位置计算期望速度，并应用位置控制增益 k_pos 来调整速度大小。然后，函数检查期望速度的大小是否超过最大速度 v_max，如果超过则进行缩放。最后，函数调用 _limit_vec_rate 来确保速度变化率不超过 dv_max。
            yaw_err = wrap_to_pi(yaw_ref - yaw)
            omega_des = float(np.clip(self.k_yaw * yaw_err, -self.w_max, self.w_max))
            omega_des = self._limit_scalar_rate(omega_des, w_prev)
#根据当前朝向和目标朝向计算航向误差 yaw_err，并应用航向控制增益 k_yaw 来计算期望角速度 omega_des。然后，函数将 omega_des 限制在最大角速度 w_max 范围内，并调用 _limit_scalar_rate 来确保角速度变化率不超过 dw_max。
            x += v_des[0] * self.dt
            y += v_des[1] * self.dt
            yaw = wrap_to_pi(yaw + omega_des * self.dt)

            traj.append([x, y, yaw_ref, v_des[0], v_des[1], omega_des])
            v_prev = v_des
            w_prev = omega_des
#plan函数是 MPC 规划器的核心方法。它根据当前状态、参考时间、跟随距离和侧向偏移方向，生成一个未来轨迹，包含每个时间步的目标位置、朝向和速度。函数通过迭代计算未来每个时间步的目标状态，并应用速度和角速度限制来确保生成的轨迹可行。在每个时间步 k 中，函数首先计算小球在未来时间 t_k 的位置和速度，然后根据这些信息计算机器人应该跟随小球的目标位置和朝向。接着，函数计算从当前状态到目标状态的期望速度和角速度，并应用限制来确保它们在可行范围内。最后，函数更新机器人状态并将当前时间步的轨迹点添加到 traj 列表中。完成所有时间步的计算后，函数将 traj 转换为 numpy 数组并返回，同时更新 _last_u 以供下一次规划使用。
        self._last_u = np.array([v_prev[0], v_prev[1], w_prev], dtype=np.float32)
        return np.array(traj, dtype=np.float32)
#ChassisMPCPlanner类实现了一个基于模型预测控制（MPC）的路径规划器，用于计算机器人在未来一段时间内应该如何移动以跟随小球。它根据当前状态、参考时间、跟随距离和侧向偏移方向，生成一个未来轨迹，包含每个时间步的目标位置、朝向和速度。函数 plan 是核心方法，它通过迭代计算未来每个时间步的目标状态，并应用速度和角速度限制来确保生成的轨迹可行。

class ToRwheelsimRobot(Robot):
    def __init__(
        self,
        dt: float,#采样周期
        v_wheel_max: float,#车轮最大速度
        steer_rate_max: Optional[float],#舵轮最大转角角速度
        wheel_acc_max: Optional[float],#车轮最大加速度
        w_max: float,#最大角速度
        v_max: float,#最大速度
        a_max: float,#最大加速度
        ema_alpha: float = 0.2,#指数移动平均滤波的平滑系数
        lsq_reg: float = 1e-4,#最小二乘法求解中的正则化参数
        slip_weight_start: Optional[float] = None,#滑移惩罚权重的起始值
        slip_weight_end: Optional[float] = None,#滑移惩罚权重的结束值
        slip_weight_warm_steps: int = 0,#滑移惩罚权重的预热步数
    ):
        self.dt = dt#采样周期
        self.v_wheel_max = v_wheel_max#车轮最大速度
        self.steer_rate_max = STEER_RATE_MAX if steer_rate_max is None else steer_rate_max
        self.wheel_acc_max = WHEEL_ACC_MAX if wheel_acc_max is None else wheel_acc_max
        self.w_max = w_max
        self.v_max = v_max
        self.a_max = a_max
        self.ema_alpha = ema_alpha
        self.lsq_reg = lsq_reg
        if slip_weight_end is None:
            slip_weight_end = 0.2#默认滑移惩罚权重结束值
        if slip_weight_start is None:
            slip_weight_start = slip_weight_end#默认滑移惩罚权重起始值
        self._slip_weight_start = float(slip_weight_start)#滑移惩罚权重的起始值
        self._slip_weight_end = float(slip_weight_end)#滑移惩罚权重的结束值
        self._slip_weight_warm_steps = int(slip_weight_warm_steps)#滑移惩罚权重的预热步数
        self.state_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.pi], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.pi], dtype=np.float32),
            dtype=np.float32,
        )#状态空间定义为一个包含位置 (x, y) 和朝向 (yaw) 的连续空间，其中 x 和 y 没有明确的边界，而 yaw 的范围被限制在 [-pi, pi] 之间。
        self.action_space = spaces.Box(
            low=np.array([-self.steer_rate_max, -self.wheel_acc_max] * 4, dtype=np.float32),
            high=np.array([self.steer_rate_max, self.wheel_acc_max] * 4, dtype=np.float32),
            dtype=np.float32,
        )#动作空间定义为一个连续空间，包含四个舵轮的转角速度和加速度，每个舵轮对应两个维度。转角速度的范围由 steer_rate_max 定义，加速度的范围由 wheel_acc_max 定义。总共有 8 个维度，分别对应四个舵轮的转角速度和加速度。
        self._steer = {name: 0.0 for name in WHEEL_ORDER}#当前每个舵轮的转角，初始值为0
        self._speed = {name: 0.0 for name in WHEEL_ORDER}#当前每个舵轮的速度，初始值为0
        self._prev_steer = {name: 0.0 for name in WHEEL_ORDER}#上一个时间步每个舵轮的转角，用于判断是否需要翻转舵轮，初始值为0
        self._vel_world = np.zeros(2, dtype=np.float32)#机器人在世界坐标系下的速度，包含 x 和 y 方向的分量，初始值为零。
        self._w_body = 0.0#机器人在自身坐标系下的角速度，初始值为零。
#ToRwheelsimRobot类实现了一个模拟四舵轮机器人动力学的类。它定义了机器人的状态空间和动作空间，并实现了 reset 和 step 方法来更新机器人的状态。机器人通过控制四个舵轮的转角速度和加速度来移动，step 方法根据当前状态和动作计算新的位置、朝向、速度和角速度，并应用物理约束来确保运动的可行性。
    @property
    def vel_world(self) -> np.ndarray:
        return self._vel_world
#vel_world 属性返回机器人在世界坐标系下的速度，包含 x 和 y 方向的分量。
    @property
    def w_body(self) -> float:
        return self._w_body
#w_body 属性返回机器人在自身坐标系下的角速度。
    def reset(self, state: np.ndarray) -> np.ndarray:
        self._steer = {name: 0.0 for name in WHEEL_ORDER}
        self._speed = {name: 0.0 for name in WHEEL_ORDER}
        self._prev_steer = {name: 0.0 for name in WHEEL_ORDER}
        self._vel_world = np.zeros(2, dtype=np.float32)
        self._w_body = 0.0
        self._step_count = 0
        return super().reset(state)
#reset 方法重置机器人的状态，包括舵轮转角、速度、世界坐标系下的速度和自身坐标系下的角速度。它还重置了步数计数器，并调用父类的 reset 方法来设置初始状态。
    def _maybe_flip(self, steer: float, speed: float, prev_steer: float) -> Tuple[float, float]:
        steer = wrap_to_pi(steer)
        steer = float(np.clip(steer, -STEER_LIMIT, STEER_LIMIT))
        steer_alt = wrap_to_pi(steer + np.pi)

        margin = np.deg2rad(5.0)
        near_limit = abs(steer) > (STEER_LIMIT - margin)
        flip_cooldown_steps = int(max(1.0, 0.3 / self.dt))
        if not hasattr(self, "_flip_counter"):
            self._flip_counter = 0
        can_flip = self._flip_counter == 0

        if can_flip and near_limit:
            steer = steer_alt
            speed = -speed
            self._flip_counter = flip_cooldown_steps
        elif self._flip_counter > 0:
            self._flip_counter -= 1

        steer = float(np.clip(steer, -STEER_LIMIT, STEER_LIMIT))
        return steer, speed
#_maybe_flip 方法用于处理舵轮转角的翻转逻辑。当舵轮转角接近其限制时，函数会尝试将转角翻转 180 度，并相应地反转速度，以避免舵轮长时间处于极限位置。函数还实现了一个冷却机制，确保在翻转后的一段时间内不会再次发生翻转，以防止频繁切换导致的不稳定行为。
    def step(self, action: np.ndarray) -> np.ndarray:
        action = clip(action, self.action_space.low, self.action_space.high)
        px, py, yaw = self.state
        if not hasattr(self, "_step_count"):
            self._step_count = 0
        self._step_count += 1

        for i, name in enumerate(WHEEL_ORDER):
            d_steer = float(action[2 * i])
            d_speed = float(action[2 * i + 1])

            steer = self._steer[name] + d_steer * self.dt
            speed = self._speed[name] + d_speed * self.dt
            speed = float(np.clip(speed, -self.v_wheel_max, self.v_wheel_max))

            steer, speed = self._maybe_flip(steer, speed, self._prev_steer[name])

            self._steer[name] = steer
            self._speed[name] = speed
            self._prev_steer[name] = steer
#step 方法根据输入的动作更新机器人的状态。它首先将动作限制在定义的动作空间内，然后根据当前状态和动作计算每个舵轮的转角和速度。函数还调用 _maybe_flip 来处理舵轮转角的翻转逻辑。最后，函数使用最小二乘法求解机器人在世界坐标系下的速度和角速度，并应用物理约束来确保运动的可行性。最终，函数更新机器人的位置和朝向，并返回新的状态。
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
#使用最小二乘法求解机器人在世界坐标系下的速度 (vx_body, vy_body) 和角速度 w_body。矩阵 A 包含了每个舵轮的转角信息，而向量 b 包含了每个舵轮的速度。通过求解线性方程 A * [vx_body, vy_body, w_body] = b，可以得到机器人在自身坐标系下的速度和角速度。
        vx_body, vy_body, w_body = np.linalg.lstsq(A, b, rcond=None)[0]#通过最小二乘法求解线性方程组，得到机器人在自身坐标系下的速度 (vx_body, vy_body) 和角速度 w_body。
        v_body_limit = 2.0 * self.v_wheel_max#机器人在自身坐标系下的速度限制，通常是车轮最大速度的两倍，因为机器人可能同时使用多个车轮来产生更大的速度。
        vx_body = float(np.clip(vx_body, -v_body_limit, v_body_limit))
        vy_body = float(np.clip(vy_body, -v_body_limit, v_body_limit))
        w_body = float(np.clip(w_body, -self.w_max, self.w_max))
#将机器人在自身坐标系下的速度转换到世界坐标系下。由于机器人可能在运动过程中发生旋转，因此需要考虑当前的朝向 yaw 来计算世界坐标系下的速度分量。函数首先计算一个中间角度 yaw_mid，假设机器人在当前时间步内以平均角速度 w_body 旋转，然后使用这个角度来计算世界坐标系下的速度 vx_world 和 vy_world。
        yaw_mid = yaw + 0.5 * w_body * self.dt
        cos_yaw = np.cos(yaw_mid)
        sin_yaw = np.sin(yaw_mid)
        vx_world = vx_body * cos_yaw - vy_body * sin_yaw
        vy_world = vx_body * sin_yaw + vy_body * cos_yaw
        vel_new = np.array([vx_world, vy_world], dtype=np.float32)
        speed = float(np.hypot(vel_new[0], vel_new[1]))
        if speed > self.v_max and speed > 1e-9:
            vel_new = vel_new * (self.v_max / speed)
        delta_v = vel_new - self._vel_world
        delta_norm = float(np.hypot(delta_v[0], delta_v[1]))
        a_limit = self.a_max * self.dt
        if delta_norm > a_limit and delta_norm > 1e-9:
            delta_v = delta_v * (a_limit / delta_norm)
        vel_new = self._vel_world + delta_v
        self._vel_world = vel_new
        self._w_body = w_body
#根据计算得到的世界坐标系下的速度和角速度，更新机器人的位置和朝向。函数使用欧拉积分方法来更新位置和朝向，其中位置的更新考虑了当前的速度，而朝向的更新考虑了当前的角速度。最后，函数将新的状态返回。
        px_next = px + self._vel_world[0] * self.dt
        py_next = py + self._vel_world[1] * self.dt
        yaw_next = wrap_to_pi(yaw + self._w_body * self.dt)
        self.state = np.array([px_next, py_next, yaw_next], dtype=np.float32)
        return self.state
#step 方法的核心功能是根据输入的动作更新机器人的状态。它首先将动作限制在定义的动作空间内，然后根据当前状态和动作计算每个舵轮的转角和速度。函数还调用 _maybe_flip 来处理舵轮转角的翻转逻辑。接着，函数使用最小二乘法求解机器人在世界坐标系下的速度和角速度，并应用物理约束来确保运动的可行性。最终，函数更新机器人的位置和朝向，并返回新的状态。

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
#BallRefContext类实现了一个上下文环境，用于提供小球的参考状态。它包含一个 reset 方法来初始化参考状态，一个 step 方法来更新参考状态，以及一个 get_zero_state 方法来获取一个全零的参考状态。参考状态包含小球的位置 (bx, by) 和速度 (bdx, bdy)，这些信息可以被 MPC 规划器和环境交互使用，以指导机器人跟随小球的运动。

class MPC_RL_ToRwheelsimEnv(Env):#MPC_RL_ToRwheelsimEnv类实现了一个基于模型预测控制（MPC）和强化学习（RL）的环境，用于训练机器人跟随在圆角矩形轨迹上运动的小球。环境定义了状态空间、动作空间、奖励函数和终止条件，并通过与 ToRwheelsimRobot 和 BallRefContext 的交互来模拟机器人和小球的运动。环境还包含一些调试功能，以便在训练过程中监控机器人的性能。
    termination_penalty = 50.0#当环境达到终止条件时，给予的惩罚奖励。   
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }#环境的元数据，定义了支持的渲染模式。在这个环境中，支持 "human" 模式用于直接显示环境，以及 "rgb_array" 模式用于返回环境的图像数据。

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
        slip_weight: float = 0.2,
        slip_weight_start: Optional[float] = None,
        slip_weight_end: Optional[float] = None,
        slip_weight_warm_steps: int = 0,
        mpc_horizon: int = 10,#MPC 规划器的预测步长，表示在每次规划时考虑未来多少个时间步的状态和动作。
        mpc_dv_max: float = 0.15,
        mpc_dw_max: float = 0.3,
        mpc_k_pos: float = 1.2,
        mpc_k_yaw: float = 2.0,
        debug_stats: bool = False,
        debug_interval: int = 100,
        w_slip_start: float = 0.0,
        w_slip_end: float = 1.2,
        w_slip_warm_steps: int = 100000,
        w_slip_max: float = 0.5,
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
            slip_weight_start=slip_weight_start if slip_weight_start is not None else slip_weight,
            slip_weight_end=slip_weight_end if slip_weight_end is not None else slip_weight,
            slip_weight_warm_steps=slip_weight_warm_steps,
        )
        self.context = BallRefContext(dt=dt, t_total=t_total)
        self.mpc_planner = ChassisMPCPlanner(
            dt=dt,
            t_total=t_total,
            horizon=mpc_horizon,
            v_max=v_max,
            w_max=w_max,
            dv_max=mpc_dv_max,
            dw_max=mpc_dw_max,
            k_pos=mpc_k_pos,
            k_yaw=mpc_k_yaw,
        )
        self.dt = dt
        self.t_total = t_total
        self.follow_dist = follow_dist
        self.max_episode_steps = int(episode_steps)
        self.v_max = v_max
        self.w_max = w_max

        self.state_dim = 3#状态维度，包含位置 (x, y) 和朝向 (yaw)
        obs_dim = 24#24个观察维度分别为第一部分：当前对第0步参考的误差（6维）第二部分：当前对第1步参考的误差（6维）第三部分：四个轮子的状态特征（12维）
        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * obs_dim, dtype=np.float32),
            high=np.array([np.inf] * obs_dim, dtype=np.float32),
            dtype=np.float32,
        )#定义环境的观测空间为一个连续空间，包含 obs_dim 个维度，每个维度的取值范围为 [-inf, inf]。这个空间用于表示环境的状态信息，包括机器人当前状态、小球参考状态以及 MPC 规划器输出的未来轨迹信息。
        self.action_space = self.robot.action_space#定义环境的动作空间与机器人定义的动作空间相同，包含四个舵轮的转角速度和加速度，每个舵轮对应两个维度。转角速度的范围由 steer_rate_max 定义，加速度的范围由 wheel_acc_max 定义。总共有 8 个维度，分别对应四个舵轮的转角速度和加速度。
        self.side = -1.0#侧向偏移方向，-1.0 表示在小球的左侧跟随，1.0 表示在小球的右侧跟随。这个参数用于计算机器人应该跟随小球的目标位置和朝向。

        self.init_dist = 1.0
        self._prev_w_body = 0.0
        self._debug_stats = debug_stats
        self._debug_interval = int(debug_interval)
        self._debug_step = 0
        self._w_slip_start = float(w_slip_start)
        self._w_slip_end = float(w_slip_end)
        self._w_slip_warm_steps = int(w_slip_warm_steps)
        self._w_slip_max = float(w_slip_max)
        self._mpc_ref_traj = np.zeros((mpc_horizon, 6), dtype=np.float32)
        self._env_step_count = 0
        self._mpc_cached_step = -1
        self.seed()
#MPC_RL_ToRwheelsimEnv类的构造函数初始化了环境的各个组件，包括机器人、上下文和 MPC 规划器。它还定义了状态空间、动作空间、奖励函数和终止条件，并设置了一些调试参数。构造函数接受多个参数来配置环境的行为，例如时间步长、总时间、跟随距离、最大速度和加速度等。这些参数可以根据需要进行调整，以创建适合特定训练任务的环境。
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        init_state: Optional[Sequence[float]] = None,
        ref_time: Optional[float] = None,
    ) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self.seed(seed)
#配置环境的
        if ref_time is None:
            period = _rounded_rect_perimeter() / BALL_SPEED
            ref_time = period * self.np_random.uniform(0.0, 1.0)
        context_state = self.context.reset(ref_time=ref_time)
#设置随机参考起点
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
#随机初始化车体位置，确保在小球周围一定距离内，并且朝向小球
        self._state = State(
            robot_state=self.robot.reset(np.array(init_state, dtype=np.float32)),
            context_state=context_state,
        )
        self._prev_w_body = 0.0
        self._debug_step = 0
        self.mpc_planner.reset()
        self._env_step_count = 0
        self._mpc_cached_step = -1
        self._update_mpc_ref_traj()
        return self._get_obs(), self._get_info()
#reset 方法重置环境的状态，包括随机初始化机器人的位置和朝向，重置上下文状态，并重置 MPC 规划器。它还重置了步数计数器和 MPC 缓存标志，以确保在新的 episode 中正确更新 MPC 参考轨迹。最后，函数返回初始观察和信息字典。
    def _update_mpc_ref_traj(self) -> None:
        # Step-cached MPC: plan once per env step to keep obs/reward consistent.
        if self._mpc_cached_step == self._env_step_count:
            return
        self._mpc_ref_traj = self.mpc_planner.plan(
            self.robot.state,
            self.context.ref_time,
            self.follow_dist,
            self.side,
        )
        self._mpc_cached_step = self._env_step_count
#_update_mpc_ref_traj 方法用于更新 MPC 参考轨迹。为了提高效率，函数实现了一个基于环境步数的缓存机制，确保在同一环境步内只计算一次 MPC 轨迹。当环境步数发生变化时，函数调用 MPC 规划器的 plan 方法来生成新的参考轨迹，并将其存储在 _mpc_ref_traj 中。同时，函数更新 _mpc_cached_step 以记录当前环境步数，以便在下一次调用时判断是否需要重新计算 MPC 轨迹。
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        self._env_step_count += 1
        self._state = self._get_next_state(action)
        self._update_mpc_ref_traj()
        reward = self._get_reward(action)
        terminated = self._get_terminated()
        if terminated:
            reward -= self.termination_penalty
        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, info
#step 方法是环境的核心交互函数。它首先增加环境步数计数器，然后根据输入的动作计算下一个状态，并更新 MPC 参考轨迹。接着，函数计算当前步骤的奖励，并判断是否满足终止条件。如果环境达到终止条件，函数会在奖励中扣除一个预定义的惩罚值。最后，函数获取当前的观察和信息字典，并将它们与奖励和终止标志一起返回。
    def _get_obs(self) -> np.ndarray:
        px, py, yaw = self.robot.state
        vx, vy = self.robot.vel_world
        omega = self.robot.w_body

        ref0 = self._mpc_ref_traj[0]
        x_ref0, y_ref0, yaw_ref0, vx_ref0, vy_ref0, omega_ref0 = ref0
        rel_pos0 = np.array([px - x_ref0, py - y_ref0], dtype=np.float32)
        yaw_err0 = np.array([wrap_to_pi(yaw - yaw_ref0)], dtype=np.float32)
        vel_err0 = np.array([vx - vx_ref0, vy - vy_ref0, omega - omega_ref0], dtype=np.float32)

        ref1 = self._mpc_ref_traj[1] if self._mpc_ref_traj.shape[0] > 1 else ref0
        x_ref1, y_ref1, yaw_ref1, vx_ref1, vy_ref1, omega_ref1 = ref1
        rel_pos1 = np.array([px - x_ref1, py - y_ref1], dtype=np.float32)
        yaw_err1 = np.array([wrap_to_pi(yaw - yaw_ref1)], dtype=np.float32)
        vel_err1 = np.array([vx - vx_ref1, vy - vy_ref1, omega - omega_ref1], dtype=np.float32)
#观察向量包含了机器人当前状态与 MPC 参考轨迹前两步的误差信息。对于每一步参考状态，观察向量包括机器人位置与参考位置的相对位置、机器人朝向与参考朝向的误差，以及机器人速度与参考速度的误差。这些信息可以帮助强化学习算法理解当前状态与目标状态之间的关系，从而学习如何调整动作以更好地跟随小球的运动。
        wheel_features = []
        for name in WHEEL_ORDER:
            steer = self.robot._steer[name]
            speed = self.robot._speed[name]
            wheel_features.extend(
                [np.cos(steer), np.sin(steer), speed / max(self.robot.v_wheel_max, 1e-6)]
            )
        wheel_features = np.array(wheel_features, dtype=np.float32)#观察向量的最后部分包含了每个舵轮的特征信息，包括转角的余弦和正弦值以及归一化的速度。这些特征可以帮助强化学习算法理解每个舵轮的当前状态，从而更好地调整动作以实现期望的运动行为。
        return np.concatenate((rel_pos0, yaw_err0, vel_err0, rel_pos1, yaw_err1, vel_err1, wheel_features), axis=0)#最终，_get_obs 方法将所有这些信息拼接成一个单一的观察向量返回。这个观察向量包含了机器人当前状态与 MPC 参考轨迹前两步的误差信息，以及每个舵轮的特征信息，为强化学习算法提供了丰富的输入数据，以便学习如何控制机器人跟随小球的运动。
#_get_obs 方法构建了环境的观察向量。它首先提取机器人的当前状态，包括位置、朝向、速度和角速度。然后，它计算机器人相对于 MPC 参考轨迹前两步的相对位置、朝向误差和速度误差。最后，函数还提取了每个舵轮的特征，包括转角的余弦和正弦值以及归一化的速度，并将所有这些信息拼接成一个单一的观察向量返回。
    def _get_reward(self, action: np.ndarray) -> float:
        px, py, yaw = self.robot.state
        vx, vy = self.robot.vel_world
        omega = self.robot.w_body
        x_ref, y_ref, yaw_ref, vx_ref, vy_ref, omega_ref = self._mpc_ref_traj[0]
#奖励函数计算了多个方面的误差，包括位置误差、朝向误差、速度误差和角速度误差。位置误差通过机器人当前位置与 MPC 参考位置之间的距离来计算，并归一化为一个相对误差。朝向误差通过机器人当前朝向与 MPC 参考朝向之间的差值来计算，并归一化为一个相对误差。速度误差通过机器人当前速度与 MPC 参考速度之间的距离来计算，并归一化为一个相对误差。角速度误差通过机器人当前角速度与 MPC 参考角速度之间的差值来计算，并归一化为一个相对误差。这些误差被加权组合在一起，形成了最终的奖励值。此外，奖励函数还包括了控制成本、方向惩罚、浪费惩罚和滑移惩罚，以鼓励机器人采取更有效和稳定的动作。
        pos_err = float(np.hypot(px - x_ref, py - y_ref) / max(self.follow_dist, 1e-6))#位置误差通过机器人当前位置与 MPC 参考位置之间的距离来计算，并归一化为一个相对误差。
        yaw_err = abs(wrap_to_pi(yaw - yaw_ref)) / np.pi#朝向误差通过机器人当前朝向与 MPC 参考朝向之间的差值来计算，并归一化为一个相对误差。
        vel_err = float(np.hypot(vx - vx_ref, vy - vy_ref) / max(self.v_max, 1e-6))#速度误差通过机器人当前速度与 MPC 参考速度之间的距离来计算，并归一化为一个相对误差。
        omega_err = abs(omega - omega_ref) / max(self.w_max, 1e-6)# 角速度误差通过机器人当前角速度与 MPC 参考角速度之间的差值来计算，并归一化为一个相对误差。

        action = clip(action, self.action_space.low, self.action_space.high)
        ctrl_cost = 0.0
        for i in range(4):
            d_steer = action[2 * i] / max(self.robot.steer_rate_max, 1e-6)
            d_speed = action[2 * i + 1] / max(self.robot.wheel_acc_max, 1e-6)
            ctrl_cost += float(d_steer**2 + d_speed**2)
#对每个轮子的转角变化率和轮速变化率做平方惩罚，鼓励机器人采取更平滑的动作，避免过于激烈的控制输入。
        wheel_vecs = []
        for name in WHEEL_ORDER:
            steer = self.robot._steer[name]
            speed = self.robot._speed[name] / max(self.robot.v_wheel_max, 1e-6)
            wheel_vecs.append(speed * np.array([np.cos(steer), np.sin(steer)], dtype=np.float32))
        wheel_vecs = np.array(wheel_vecs, dtype=np.float32)
        v_mean = np.mean(wheel_vecs, axis=0)
        dir_penalty = float(np.sum(np.linalg.norm(wheel_vecs - v_mean, axis=1) ** 2))
        waste_penalty = float(
            np.sum(np.linalg.norm(wheel_vecs, axis=1) ** 2) - 4.0 * np.linalg.norm(v_mean) ** 2
        )
#计算每个轮子的速度向量，并通过计算它们与平均速度向量之间的差异来计算方向惩罚。这个惩罚鼓励所有轮子朝着相似的方向运动，从而提高机器人的运动效率。浪费惩罚通过计算所有轮子速度的平方和与平均速度平方的差异来计算，鼓励机器人更有效地利用其轮子产生运动。
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        v_body_x = vx * cos_yaw + vy * sin_yaw
        v_body_y = -vx * sin_yaw + vy * cos_yaw
        slip_penalty = 0.0
        slip_sq_list = []
        for name in WHEEL_ORDER:
            x_i, y_i = WHEEL_POS[name]
            steer = self.robot._steer[name]
            wheel_vx = v_body_x - self.robot.w_body * y_i
            wheel_vy = v_body_y + self.robot.w_body * x_i
            slip = (-np.sin(steer) * wheel_vx + np.cos(steer) * wheel_vy)
            slip_norm = slip / max(self.robot.v_wheel_max, 1e-6)
            slip_sq = float(slip_norm ** 2)
            slip_penalty += slip_sq
            slip_sq_list.append(slip_sq)
#计算每个轮子的滑移量，并将其归一化为一个相对值。滑移惩罚通过计算所有轮子滑移的平方和来计算，鼓励机器人减少轮子与地面之间的滑移，从而提高运动效率和稳定性。
        yaw_rate = float(self.robot.w_body)
        yaw_accel = (yaw_rate - self._prev_w_body) / max(self.dt, 1e-6)
        self._prev_w_body = yaw_rate
#横摆角加速度惩罚通过计算当前横摆角速度与前一时间步的差值来计算，并归一化为一个相对值。这个惩罚鼓励机器人保持更平稳的旋转，避免过于剧烈的横摆变化。
        w_pos = 6.0
        w_vel = 1.0
        w_yaw = 2.0
        w_omega = 0.5
        w_ctrl = 0.01
        w_dir = 0.2
        w_waste = 0.1
        #滑移惩罚的权重根据预设的 warm-up 步数逐渐增加，从 w_slip_start 线性增加到 w_slip_end，直到达到 warm-up 步数后保持在 w_slip_end。这个机制允许训练初期对滑移惩罚的影响较小，以便模型更快地学习基本的跟随行为，然后逐渐增加滑移惩罚
        if self._w_slip_warm_steps <= 0:
            w_slip = self._w_slip_end
        else:
            frac = min(1.0, self._debug_step / self._w_slip_warm_steps)
            w_slip = self._w_slip_start + (self._w_slip_end - self._w_slip_start) * frac
        w_yaw_accel = 0.01
        max_slip_penalty = max(slip_sq_list) if slip_sq_list else 0.0
        reward = (
            -w_pos * pos_err
            -w_vel * vel_err
            -w_yaw * yaw_err
            -w_omega * (omega_err ** 2)
            -w_yaw_accel * (yaw_accel ** 2)
            -w_ctrl * ctrl_cost
            -w_dir * dir_penalty
            -w_waste * waste_penalty
            -w_slip * slip_penalty
            -self._w_slip_max * max_slip_penalty
        )#最终的奖励值是所有这些误差和惩罚的加权组合。通过调整各个权重参数，可以控制不同类型的误差和惩罚对总奖励的影响，从而引导强化学习算法学习到更有效的跟随行为。
        if self._debug_stats:
            self._debug_step += 1
            if self._debug_step % max(self._debug_interval, 1) == 0:
                mean_action = float(np.mean(np.abs(action)))
                mean_speed = float(np.mean([abs(self.robot._speed[n]) for n in WHEEL_ORDER]))
                v_norm = float(np.hypot(vx, vy))
                print(
                    f"[debug] step={self._debug_step} "
                    f"mean|action|={mean_action:.4f} mean|wheel_speed|={mean_speed:.4f} "
                    f"|v|={v_norm:.4f}"
                )
        return reward
#_get_reward 方法计算了环境的奖励值。它首先计算了多个方面的误差，包括位置误差、朝向误差、速度误差和角速度误差。然后，它计算了控制成本、方向惩罚、浪费惩罚和滑移惩罚。最后，函数将所有这些误差和惩罚加权组合在一起，形成了最终的奖励值。此外，函数还包含了一些调试统计信息，用于监控训练过程中的动作大小、轮速和整体速度等指标。
    def _get_terminated(self) -> bool:
        px, py = self.robot.state[:2]
        bx, by = self.context.state.reference[:2]
        dist_to_ball = np.hypot(px - bx, py - by)
        return dist_to_ball > 5.0
#终止条件，距离目标太远就失败
    def render(self, mode="human"):
        import matplotlib.pyplot as plt
        import matplotlib.patches as pc

        plt.ion()
        fig = plt.figure(num=0, figsize=(6.4, 6.4))
        plt.clf()
        px, py, yaw = self.robot.state
        bx, by, bdx, bdy = self.context.state.reference
        rx, ry, _ = compute_follow_target(bx, by, bdx, bdy, self.follow_dist, self.side)

        ax = plt.axes(xlim=(px - 4, px + 4), ylim=(py - 4, py + 4))
        ax.set_aspect("equal")

        veh_length = 0.48
        veh_width = 0.35
        x_offset = veh_length / 2.0 * np.cos(yaw) - veh_width / 2.0 * np.sin(yaw)
        y_offset = veh_length / 2.0 * np.sin(yaw) + veh_width / 2.0 * np.cos(yaw)
        ax.add_patch(pc.Rectangle(
            (px - x_offset, py - y_offset),
            veh_length,
            veh_width,
            angle=np.rad2deg(yaw),
            facecolor="w",
            edgecolor="r",
            zorder=2,
        ))
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

        ax.set_title("Swerve Tracking (MPC-RL Env)")
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
#可视化环境状态，包括机器人、轮子、小球和目标位置。函数使用 Matplotlib 来绘制环境的当前状态，并支持两种渲染模式：human 模式直接显示图形界面，rgb_array 模式返回图像数据以供进一步处理或保存。

def env_creator(**kwargs):
    return MPC_RL_ToRwheelsimEnv(**kwargs)


# Training curriculum:
# 1. straight line
# 2. circular path
# 3. rounded rectangle
