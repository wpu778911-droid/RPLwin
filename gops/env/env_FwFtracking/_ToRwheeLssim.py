#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np

# ✅ 尝试使用交互后端；如果失败自动回退到 Agg（保存文件）
import matplotlib
try:
    # 首先尝试 TkAgg（需要系统 tkinter）
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    test_fig = plt.figure()
    plt.close(test_fig)
    interactive_mode = True
except Exception as e:
    # 回退到 Agg，后续会保存文件而非显示
    print(f"[警告] TkAgg 后端失败 ({type(e).__name__}，可能是 tkinter 缺失)，改用 Agg 后端并保存文件")
    matplotlib.use("Agg")
    interactive_mode = False

import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Polygon
import scipy.sparse as sp


try:
    import osqp
except ImportError as e:
    raise ImportError("需要安装 OSQP：pip install osqp") from e


# =========================================================
# 1) 参数区
# =========================================================
DT = 0.05

# 小球匀速与仿真时长配置
BALL_SPEED = 0.1  # m/s
NUM_LAPS = 2      # 矩形走几圈；想一直跑就调大

# 底盘约束
VX_MAX = 0.3
VY_MAX = 0.3
A_MAX  = 1.0
W_MAX  = 1.5

# 舵轮转角限位（±170°）
STEER_LIMIT = np.deg2rad(170.0)

# ✅ 方案一：舵轮角速度限制（关键参数）
# 真实电机不允许瞬间跳变：例如 120 deg/s
STEER_RATE_MAX = np.deg2rad(120.0)   # rad/s
STEER_DMAX = STEER_RATE_MAX * DT     # 每步最大角度变化

# 轮子参数（只用于可视化“滚动”辐条）
WHEEL_RADIUS = 0.06

# 你的 URDF 底盘几何
L = 0.48
W = 0.35
WHEEL_POS = {
    "FR": (+0.24, -0.175),
    "RR": (-0.24, -0.175),
    "RL": (-0.24, +0.175),
    "FL": (+0.24, +0.175),
}
WHEEL_LEN = 0.10
WHEEL_WID = 0.04

# MPC horizon
N = 20

# 小球矩形轨迹参数（半长、半宽），总尺寸 = 5m x 3m
RECT_W = 2.5
RECT_H = 1.5
CORNER_R = 0.3

def _rounded_rect_perimeter():
    straight_x = 2.0 * (RECT_W - CORNER_R)
    straight_y = 2.0 * (RECT_H - CORNER_R)
    arc = 0.5 * np.pi * CORNER_R
    return 2.0 * (straight_x + straight_y) + 4.0 * arc

RECT_PERIM = _rounded_rect_perimeter()
LAP_TIME = RECT_PERIM / BALL_SPEED
SIM_TIME = LAP_TIME * NUM_LAPS

# Runtime overrides (filled by CLI)
FOLLOW_DIST = 0.3
SIDE_FIXED = None
REF_TIME = 0.0
OUTPUT_DIR = None
STEPS_OVERRIDE = None

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ref-time", type=float, default=0.0)
    parser.add_argument("--follow-dist", type=float, default=FOLLOW_DIST)
    parser.add_argument("--side", type=float, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--num-laps", type=int, default=NUM_LAPS)
    parser.add_argument("--no-render", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    _args = _parse_args()
    FOLLOW_DIST = float(_args.follow_dist)
    SIDE_FIXED = None if _args.side is None else float(_args.side)
    REF_TIME = float(_args.ref_time)
    OUTPUT_DIR = _args.output_dir
    STEPS_OVERRIDE = _args.steps
    if _args.num_laps is not None:
        NUM_LAPS = int(_args.num_laps)
        SIM_TIME = LAP_TIME * NUM_LAPS
    if _args.seed is not None:
        np.random.seed(int(_args.seed))
    if _args.no_render:
        interactive_mode = False

# MPC 权重（可以调）
# 状态: [px, py, vx, vy, yaw]
Q_pos = 40.0
Q_vel = 5.0
Q_yaw = 10.0
Qf_pos = 80.0
Qf_vel = 10.0
Qf_yaw = 20.0

R_ax = 0.2
R_ay = 0.2
R_w  = 0.3

# (可选) 平滑项：惩罚 Δu
DU_ax = 0.05
DU_ay = 0.05
DU_w  = 0.05


# =========================================================
# 2) 工具函数
# =========================================================
def wrap_to_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def rot2d(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])

def clip(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


# =========================================================
# 3) 小球 8 字轨迹（含速度，便于 MPC 追踪）
# Gerono lemniscate:
#   x = A sin(wt)
#   y = A sin(wt) cos(wt) = 0.5 A sin(2wt)
# =========================================================
def _rounded_rect_pos_vel(s):
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

def ball_figure8(t):
    t = t + REF_TIME
    s = (t % LAP_TIME) * BALL_SPEED
    x, y, _, _ = _rounded_rect_pos_vel(s)
    return x, y

def ball_figure8_vel(t):
    t = t + REF_TIME
    s = (t % LAP_TIME) * BALL_SPEED
    _, _, dx, dy = _rounded_rect_pos_vel(s)
    return dx, dy


# =========================================================
# 4) 四舵轮逆运动学：车体系 (vx, vy, w) -> 每轮 (steer, speed)
# =========================================================
def swerve_ik(vx, vy, w):
    out = {}
    for name, (x_i, y_i) in WHEEL_POS.items():
        vix = vx - w * y_i
        viy = vy + w * x_i
        steer = np.arctan2(viy, vix)
        speed = np.hypot(vix, viy)
        out[name] = (steer, speed)
    return out

def steer_limit_with_flip(steer_des, speed_des):
    """±170° 限位 + 翻转策略（steer+pi, speed取反）尽量避免撞限位。"""
    steer_des = wrap_to_pi(steer_des)

    def over_cost(theta):
        if -STEER_LIMIT <= theta <= STEER_LIMIT:
            return 0.0
        if theta < -STEER_LIMIT:
            return (-STEER_LIMIT - theta)
        return (theta - STEER_LIMIT)

    # 原策略：比较不合法区间代价，选择更小代价的解（可能翻转）
    t1, s1 = steer_des, speed_des
    c1 = over_cost(t1)

    t2, s2 = wrap_to_pi(steer_des + np.pi), -speed_des
    c2 = over_cost(t2)

    # 引入一个翻转门槛（hysteresis）：只有当翻转后的代价显著更小
    # 时才选择翻转，以避免在边界附近频繁切换导致机器人运动反转。
    FLIP_MARGIN = 1e-2  # 可调（弧度或代价尺度），可改大到 0.1 以更强的抑制
    if (c2 + FLIP_MARGIN < c1) or (abs(c2 - c1) < 1e-9 and abs(t2) < abs(t1)):
        t, s = t2, s2
    else:
        t, s = t1, s1

    t = clip(t, -STEER_LIMIT, STEER_LIMIT)
    return t, s


# =========================================================
# ✅ 方案一核心：舵轮角“连续化”（无突变）
# - 在两个等价解 (θ, s) 和 (θ+pi, -s) 中选择“离上一时刻更近”的
# - 再加转向角速度限制 |Δθ| <= STEER_DMAX
# =========================================================
def limit_penalty(theta):
    """越过 ±STEER_LIMIT 的惩罚（越过越大），用于在接近限位时更倾向可行解。"""
    if -STEER_LIMIT <= theta <= STEER_LIMIT:
        return 0.0
    if theta < -STEER_LIMIT:
        return (-STEER_LIMIT - theta)
    return (theta - STEER_LIMIT)

def choose_and_rate_limit_steer(theta_raw, speed_raw, theta_prev, lam_limit=0.3):
    """
    输入：
      theta_raw = atan2 得到的期望角（未处理）
      speed_raw = 轮速（非负）
      theta_prev= 上一步实际下发角
    输出：
      theta_cmd, speed_cmd （已选择等价解 + 限位 + 角速度限制）
    """
    # 两个等价候选
    th1, sp1 = wrap_to_pi(theta_raw), speed_raw
    th2, sp2 = wrap_to_pi(theta_raw + np.pi), -speed_raw

    # 把候选先做限位裁剪（与实际一致）
    th1c = clip(th1, -STEER_LIMIT, STEER_LIMIT)
    th2c = clip(th2, -STEER_LIMIT, STEER_LIMIT)

    # 连续性代价（越接近上一时刻越好） + 限位惩罚（越界越不好）
    d1 = abs(wrap_to_pi(th1c - theta_prev)) + lam_limit * limit_penalty(th1)
    d2 = abs(wrap_to_pi(th2c - theta_prev)) + lam_limit * limit_penalty(th2)

    if d2 < d1:
        th_des, sp_des = th2c, sp2
    else:
        th_des, sp_des = th1c, sp1

    # 角速度限制：只允许走“最短角差”的一小步
    dth = wrap_to_pi(th_des - theta_prev)
    dth = clip(dth, -STEER_DMAX, +STEER_DMAX)
    th_cmd = wrap_to_pi(theta_prev + dth)

    # 再保证不超限（双保险）
    th_cmd = clip(th_cmd, -STEER_LIMIT, STEER_LIMIT)
    return th_cmd, sp_des


# =========================================================
# 5) MPC 模型：世界系线性模型
# 状态 x = [px, py, vx, vy, yaw]
# 控制 u = [ax, ay, w]
# =========================================================
def build_dynamics(dt):
    nx, nu = 5, 3
    A = np.eye(nx)
    A[0, 2] = dt
    A[1, 3] = dt
    A[4, 4] = 1.0

    B = np.zeros((nx, nu))
    B[0, 0] = 0.5 * dt**2
    B[1, 1] = 0.5 * dt**2
    B[2, 0] = dt
    B[3, 1] = dt
    B[4, 2] = dt
    return A, B

A_d, B_d = build_dynamics(DT)
nx, nu = 5, 3


# =========================================================
# 6) 构造 MPC QP（OSQP，稀疏）
# =========================================================
def mpc_setup_osqp(N):
    nX = N * nx
    nU = N * nu
    nvar = nX + nU

    # ---- cost H ----
    Q = np.diag([Q_pos, Q_pos, Q_vel, Q_vel, Q_yaw])
    Qf = np.diag([Qf_pos, Qf_pos, Qf_vel, Qf_vel, Qf_yaw])
    R = np.diag([R_ax, R_ay, R_w])
    Du = np.diag([DU_ax, DU_ay, DU_w])

    # block-diag for states
    Q_blocks = [Q]*(N-1) + [Qf]
    P_x = sp.block_diag(Q_blocks, format="csc")
    P_u = sp.block_diag([R]*N, format="csc")
    P = sp.block_diag([P_x, P_u], format="csc")

    # add Δu penalty
    if (DU_ax > 0) or (DU_ay > 0) or (DU_w > 0):
        rows = []
        cols = []
        vals = []
        for k in range(N):
            for j in range(nu):
                idx = k*nu + j
                rows.append(idx); cols.append(idx); vals.append(1.0)
                if k > 0:
                    rows.append(idx); cols.append((k-1)*nu + j); vals.append(-1.0)
        D = sp.csc_matrix((vals, (rows, cols)), shape=(nU, nU))
        P_du = D.T @ sp.kron(sp.eye(N, format="csc"), Du) @ D
        P = P + sp.block_diag([sp.csc_matrix((nX, nX)), P_du], format="csc")

    q = np.zeros(nvar)

    # ---- constraints ----
    Aeq_rows = N * nx
    Aeq = sp.lil_matrix((Aeq_rows, nvar))

    def x_index(k):  # k=1..N
        return (k-1)*nx

    def u_index(k):  # k=0..N-1
        return nX + k*nu

    # k=0: x1 - B u0 = A x0
    Aeq[0:nx, x_index(1):x_index(1)+nx] = sp.eye(nx)
    Aeq[0:nx, u_index(0):u_index(0)+nu] = -B_d

    # k=1..N-1
    for k in range(1, N):
        row = k*nx
        Aeq[row:row+nx, x_index(k):x_index(k)+nx] = -A_d
        Aeq[row:row+nx, x_index(k+1):x_index(k+1)+nx] = sp.eye(nx)
        Aeq[row:row+nx, u_index(k):u_index(k)+nu] = -B_d

    Aeq = Aeq.tocsc()

    I = sp.eye(nvar, format="csc")
    A_cons = sp.vstack([Aeq, I], format="csc")

    l = np.zeros(Aeq_rows + nvar)
    u = np.zeros(Aeq_rows + nvar)

    l[:Aeq_rows] = 0.0
    u[:Aeq_rows] = 0.0

    big = 1e9
    l[Aeq_rows:] = -big
    u[Aeq_rows:] = +big

    # bounds on predicted vx, vy
    for k in range(1, N+1):
        base = x_index(k)
        l[Aeq_rows + base + 2] = -VX_MAX
        u[Aeq_rows + base + 2] = +VX_MAX
        l[Aeq_rows + base + 3] = -VY_MAX
        u[Aeq_rows + base + 3] = +VY_MAX

    # bounds on inputs
    for k in range(N):
        base = u_index(k)
        l[Aeq_rows + base + 0] = -A_MAX
        u[Aeq_rows + base + 0] = +A_MAX
        l[Aeq_rows + base + 1] = -A_MAX
        u[Aeq_rows + base + 1] = +A_MAX
        l[Aeq_rows + base + 2] = -W_MAX
        u[Aeq_rows + base + 2] = +W_MAX

    solver = osqp.OSQP()
    solver.setup(P=P, q=q, A=A_cons, l=l, u=u, verbose=False, polish=True, warm_start=True)
    return solver, P, A_cons, l, u, Aeq_rows

solver, P, A_cons, l_base, u_base, Aeq_rows = mpc_setup_osqp(N)


# =========================================================
# 7) 单步 MPC 求解
# =========================================================
def solve_mpc_one_step(x0, t0):
    xref = np.zeros((N, nx))

    # 初始化：上一步的 yaw（作为 yaw_ref 的连续化基准）
    yaw_prev = x0[4]

    for k in range(1, N+1):
        tk = t0 + k*DT
        bx, by = ball_figure8(tk)
        bdx, bdy = ball_figure8_vel(tk)

        # 我们希望机器人保持与小球 1m 距离，且车头始终朝向小球
        # 所以目标机器人位置是小球位置沿小球速度方向后退 1m
        vel_norm = np.hypot(bdx, bdy)
        if vel_norm > 1e-6:
            ux, uy = bdx / vel_norm, bdy / vel_norm
        else:
            # 若小球速度很小，使用上一个小球位置指向当前的方向
            bx_prev, by_prev = ball_figure8(max(tk - DT, 0.0))
            vx_approx = bx - bx_prev
            vy_approx = by - by_prev
            vnorm2 = np.hypot(vx_approx, vy_approx)
            if vnorm2 > 1e-6:
                ux, uy = vx_approx / vnorm2, vy_approx / vnorm2
            else:
                ux, uy = 1.0, 0.0

        # desired robot position: offset from ball along the perpendicular (scan around the ball)
        follow_dist = FOLLOW_DIST
        # perpendicular to velocity: (-uy, ux)
        perp = np.array([-uy, ux])
        pnorm = np.hypot(perp[0], perp[1])
        if pnorm < 1e-6:
            perp_unit = np.array([0.0, 1.0])
        else:
            perp_unit = perp / pnorm

        if SIDE_FIXED is not None:
            side = SIDE_FIXED
        else:
            # choose side so that robot stays on the outer side of the rectangle
            # we prefer the candidate (ball +/- perp_unit*follow_dist) that has
            # larger distance to the rectangle center (origin)
            ball_pos = np.array([bx, by])
            cand1 = ball_pos + perp_unit * follow_dist  # side = +1
            cand2 = ball_pos - perp_unit * follow_dist  # side = -1
            # rectangle center is at origin (0,0)
            d1 = np.hypot(cand1[0], cand1[1])
            d2 = np.hypot(cand2[0], cand2[1])
            if abs(d1 - d2) > 1e-6:
                side = 1.0 if d1 > d2 else -1.0
            else:
                # fallback to initial side relative to ball
                robot_init_rel = np.array([x0[0], x0[1]]) - ball_pos
                side_dot = np.dot(perp_unit, robot_init_rel)
                side = 1.0 if side_dot >= 0 else -1.0

        rx = bx + perp_unit[0] * follow_dist * side
        ry = by + perp_unit[1] * follow_dist * side

        # desired yaw: robot heading should point to the ball
        yaw_ref = np.arctan2(by - ry, bx - rx)
        # 连续化 yaw
        dyaw = wrap_to_pi(yaw_ref - yaw_prev)
        yaw_ref = yaw_prev + dyaw
        yaw_prev = yaw_ref

        # set reference state: position = rx,ry; velocity ~ ball velocity
        xref[k-1, :] = np.array([rx, ry, bdx, bdy, yaw_ref])

    Q = np.diag([Q_pos, Q_pos, Q_vel, Q_vel, Q_yaw])
    Qf = np.diag([Qf_pos, Qf_pos, Qf_vel, Qf_vel, Qf_yaw])

    q = np.zeros(N*nx + N*nu)
    for k in range(1, N+1):
        Qk = Q if k < N else Qf
        q[(k-1)*nx:k*nx] = -Qk @ xref[k-1]

    solver.update(q=q)

    l = l_base.copy()
    u = u_base.copy()

    rhs0 = A_d @ x0
    l[:nx] = rhs0
    u[:nx] = rhs0
    l[nx:Aeq_rows] = 0.0
    u[nx:Aeq_rows] = 0.0

    solver.update(l=l, u=u)

    res = solver.solve()
    if res.info.status_val not in (1, 2):
        return np.zeros(3), xref

    z = res.x
    u0 = z[N*nx : N*nx + nu]
    return u0, xref


# =========================================================
# 8) 主仿真：先跑完记录，再做动画 & 作图
# =========================================================
steps = int(SIM_TIME / DT) if STEPS_OVERRIDE is None else int(STEPS_OVERRIDE)

# 初始小球位置与速度
bx0, by0 = ball_figure8(0.0)
bdx0, bdy0 = ball_figure8_vel(0.0)

# 初始机器人位置：以小球初始位置为中心，距离固定为 1m，方向以左下(-135°)为主、微随机扰动
dist0 = 1.0
base_angle = -3.0 * np.pi / 4.0  # -135 degrees (left-bottom)
angle = base_angle + np.random.uniform(-0.2, 0.2)
robot_px0 = bx0 + dist0 * np.cos(angle)
robot_py0 = by0 + dist0 * np.sin(angle)
robot_yaw0 = np.arctan2(by0 - robot_py0, bx0 - robot_px0)
x_state = np.array([robot_px0, robot_py0, 0.0, 0.0, robot_yaw0])

t = 0.0

hist_t = []
hist_robot = []
hist_ball = []
hist_u = []
hist_err = []

hist_wheel_steer = {k: [] for k in WHEEL_POS.keys()}
hist_wheel_speed = {k: [] for k in WHEEL_POS.keys()}
hist_wheel_phi = {k: 0.0 for k in WHEEL_POS.keys()}

# ✅ 方案一：记录“实际下发的舵轮角”（连续的）
steer_prev = {k: 0.0 for k in WHEEL_POS.keys()}

for i in range(steps):
    bx, by = ball_figure8(t)

    u0, _ = solve_mpc_one_step(x_state, t)
    ax_cmd, ay_cmd, w_cmd = u0

    px, py, vx, vy, yaw = x_state
    px_next = px + vx*DT + 0.5*ax_cmd*DT**2
    py_next = py + vy*DT + 0.5*ay_cmd*DT**2
    vx_next = vx + ax_cmd*DT
    vy_next = vy + ay_cmd*DT
    yaw_next = wrap_to_pi(yaw + w_cmd*DT)

    x_state = np.array([px_next, py_next, vx_next, vy_next, yaw_next])

    err = np.hypot(bx - px_next, by - py_next)

    # 车体系速度用于舵轮 IK
    Rw = rot2d(yaw_next)
    v_body = Rw.T @ np.array([vx_next, vy_next])
    vx_b, vy_b = v_body[0], v_body[1]

    wheel_cmd = swerve_ik(vx_b, vy_b, w_cmd)

    # ✅ 方案一：每个轮子用 “等价解选择(连续性优先) + 角速度限制”
    for name in WHEEL_POS.keys():
        theta_raw, speed_raw = wheel_cmd[name]  # theta_raw=atan2, speed_raw>=0
        # 先用你原来的限位+flip做一个基础候选（不丢你原逻辑）
        # 但最终选用 choose_and_rate_limit_steer 来保证连续
        theta0, speed0 = steer_limit_with_flip(theta_raw, speed_raw)

        theta_cmd, speed_cmd = choose_and_rate_limit_steer(
            theta0, abs(speed0), steer_prev[name], lam_limit=0.3
        )
        # speed0 的符号（正/负）要保留：如果原本 flip 后是负速度，就沿用它
        if speed0 < 0:
            speed_cmd = -abs(speed_cmd)

        steer_prev[name] = theta_cmd

        hist_wheel_steer[name].append(theta_cmd)
        hist_wheel_speed[name].append(speed_cmd)

    hist_t.append(t)
    hist_robot.append(x_state.copy())
    hist_ball.append((bx, by))
    hist_u.append([ax_cmd, ay_cmd, w_cmd])
    hist_err.append(err)

    t += DT

hist_t = np.array(hist_t)
hist_robot = np.array(hist_robot)
hist_ball = np.array(hist_ball)
hist_u = np.array(hist_u)
hist_err = np.array(hist_err)

vx_w = hist_robot[:, 2]
vy_w = hist_robot[:, 3]
speed_w = np.hypot(vx_w, vy_w)

ax_w = hist_u[:, 0]
ay_w = hist_u[:, 1]
acc_w = np.hypot(ax_w, ay_w)

yaw_w = hist_robot[:, 4]

# Calculate ball trajectory curvature
# Ball position derivatives
ball_x = hist_ball[:, 0]
ball_y = hist_ball[:, 1]

# First derivative (velocity)
ball_vx = np.gradient(ball_x, hist_t)
ball_vy = np.gradient(ball_y, hist_t)
ball_speed = np.hypot(ball_vx, ball_vy)

# Second derivative (acceleration)
ball_ax = np.gradient(ball_vx, hist_t)
ball_ay = np.gradient(ball_vy, hist_t)

# Curvature κ = |x'*y'' - y'*x''| / (x'^2 + y'^2)^(3/2)
numerator = np.abs(ball_vx * ball_ay - ball_vy * ball_ax)
denominator = np.power(ball_vx**2 + ball_vy**2, 1.5) + 1e-8  # avoid division by zero
ball_curvature = numerator / denominator


# =========================================================
# 9) 动画可视化（用记录回放）
# =========================================================
fig, ax = plt.subplots(figsize=(12, 12))
ax.set_aspect("equal")
# larger view so robot start and motion are visible
ax.set_xlim(-6.0, 6.0)
ax.set_ylim(-6.0, 6.0)
ax.set_title("QP-MPC Swerve Tracking a Ball (Animation) — Scan Follow (1m offset)")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.grid(True)

path_samples = np.linspace(0.0, LAP_TIME, 400, endpoint=True)
rect_path = np.array([ball_figure8(t) for t in path_samples], dtype=np.float32)
line_ball_ref, = ax.plot(rect_path[:, 0], rect_path[:, 1], "r-", lw=1.5, alpha=0.5, label="ball ref path")

line_robot, = ax.plot([], [], "b--", lw=1, label="robot path")
line_ball,  = ax.plot([], [], "r:",  lw=1, label="ball path")

ball_patch = Circle((0, 0), radius=0.05, color="red")
ax.add_patch(ball_patch)

chassis_poly = Polygon(np.zeros((4, 2)), closed=True, fill=False, edgecolor="k", lw=2)
ax.add_patch(chassis_poly)

heading_line, = ax.plot([], [], "k-", lw=3, label="heading")
vel_line,     = ax.plot([], [], "g-", lw=3, label="velocity")

wheel_patches = {}
wheel_spokes = {}
for name in WHEEL_POS.keys():
    # 轮子矩形：使用边框+填充以确保可见
    poly = Polygon(np.zeros((4, 2)), closed=True, fill=True, 
                   facecolor="lightgray", edgecolor="darkgray", lw=1.5)
    wheel_patches[name] = poly
    ax.add_patch(poly)
    # 轮子辐条（转动指示）
    spk, = ax.plot([], [], color="orange", lw=2)
    wheel_spokes[name] = spk

ax.legend(loc="upper right")

robot_path_x, robot_path_y = [], []
ball_path_x, ball_path_y = [], []
# time label (updated each frame)
time_text = ax.text(0.02, 0.96, '', transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

def update_anim(i):
    px, py, vx, vy, yaw = hist_robot[i]
    bx, by = hist_ball[i]
    ax_cmd, ay_cmd, w_cmd = hist_u[i]

    robot_path_x.append(px); robot_path_y.append(py)
    ball_path_x.append(bx);  ball_path_y.append(by)
    line_robot.set_data(robot_path_x, robot_path_y)
    line_ball.set_data(ball_path_x, ball_path_y)

    ball_patch.center = (bx, by)

    Rw = rot2d(yaw)
    chassis = np.array([
        [ L/2,  W/2],
        [ L/2, -W/2],
        [-L/2, -W/2],
        [-L/2,  W/2],
    ])
    chassis_world = (Rw @ chassis.T).T + np.array([px, py])
    chassis_poly.set_xy(chassis_world)

    head = np.array([0.25*np.cos(yaw), 0.25*np.sin(yaw)])
    heading_line.set_data([px, px+head[0]], [py, py+head[1]])

    v_world = np.array([vx, vy])
    vel_line.set_data([px, px+0.6*v_world[0]], [py, py+0.6*v_world[1]])

    for name, (wx, wy) in WHEEL_POS.items():
        steer = hist_wheel_steer[name][i]
        speed = hist_wheel_speed[name][i]

        center = Rw @ np.array([wx, wy]) + np.array([px, py])

        wheel_shape = np.array([
            [ WHEEL_LEN/2,  WHEEL_WID/2],
            [ WHEEL_LEN/2, -WHEEL_WID/2],
            [-WHEEL_LEN/2, -WHEEL_WID/2],
            [-WHEEL_LEN/2,  WHEEL_WID/2],
        ])

        wheel_R = Rw @ rot2d(steer)
        wheel_world = (wheel_R @ wheel_shape.T).T + center
        wheel_patches[name].set_xy(wheel_world)

        # rolling spoke
        hist_wheel_phi[name] = wrap_to_pi(hist_wheel_phi[name] + (speed / max(WHEEL_RADIUS, 1e-6)) * DT)
        phi = hist_wheel_phi[name]
        spoke_len = WHEEL_WID * 0.9
        spoke_local = rot2d(phi) @ np.array([0.0, spoke_len/2])
        p1 = center + wheel_R @ (-spoke_local)
        p2 = center + wheel_R @ ( spoke_local)
        wheel_spokes[name].set_data([p1[0], p2[0]], [p1[1], p2[1]])

    # update time label (after processing all wheels)
    try:
        t_now = hist_t[i]
        time_text.set_text(f"t={t_now:.2f} s")
    except Exception:
        time_text.set_text('')

    return [line_robot, line_ball, ball_patch, chassis_poly, heading_line, vel_line] + \
           list(wheel_patches.values()) + list(wheel_spokes.values()) + [time_text]

ani = FuncAnimation(fig, update_anim, frames=len(hist_t), interval=DT*1000, blit=False)

# prepare the run output directory: result/第N次 (auto-increment)
result_base = Path(OUTPUT_DIR) if OUTPUT_DIR else Path("result")
result_base.mkdir(exist_ok=True)

# Find max run number
run_dirs = [d for d in result_base.glob("第*") if d.is_dir()]
max_run = 0
for d in run_dirs:
    try:
        num = int(d.name[1:-1])  # extract number from "第N次"
        max_run = max(max_run, num)
    except:
        pass

run_num = max_run + 1
run_dir = result_base / f"第{run_num}次"
run_dir.mkdir(exist_ok=True, parents=True)
print(f"Output folder: {run_dir}")

# Keep animation reference and show interactively if supported
anim_fig_num = fig.number
_ani = ani
if interactive_mode:
    plt.show()
else:
    print("[信息] Agg 后端模式：将保存动画和数据图到文件")



# =========================================================
# 10) 输出曲线：速度 / 加速度 / 误差 / 4轮转向角
# =========================================================
plt.figure(figsize=(10, 4))
plt.plot(hist_t, vx_w, label="vx_world")
plt.plot(hist_t, vy_w, label="vy_world")
plt.plot(hist_t, speed_w, label="speed_world")
plt.axhline(VX_MAX, linestyle="--")
plt.axhline(-VX_MAX, linestyle="--")
plt.title("Chassis Velocity (World)")
plt.xlabel("t [s]")
plt.ylabel("m/s")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 4))
plt.plot(hist_t, ax_w, label="ax_world")
plt.plot(hist_t, ay_w, label="ay_world")
plt.plot(hist_t, acc_w, label="acc_world")
plt.axhline(A_MAX, linestyle="--")
plt.axhline(-A_MAX, linestyle="--")
plt.title("Chassis Acceleration (World) = MPC control")
plt.xlabel("t [s]")
plt.ylabel("m/s^2")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 4))
plt.plot(hist_t, hist_err, label="position error")
plt.title("Tracking Error (distance)")
plt.xlabel("t [s]")
plt.ylabel("m")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 6))
for name in WHEEL_POS.keys():
    plt.plot(hist_t, np.array(hist_wheel_steer[name])*180/np.pi, label=f"{name} steer")
plt.axhline(170, linestyle="--")
plt.axhline(-170, linestyle="--")
plt.title("Wheel Steering Angles (deg) with ±170° limit")
plt.xlabel("t [s]")
plt.ylabel("deg")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(10, 4))
plt.plot(hist_t, ball_curvature, label="Ball trajectory curvature", color="purple", linewidth=2)
plt.title("Ball Trajectory Instantaneous Curvature")
plt.xlabel("t [s]")
plt.ylabel("Curvature (1/m)")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save data plots to the run-specific directory
fignums = [n for n in plt.get_fignums() if n != anim_fig_num]
fignums.sort()

desired_names = [
    '01_chassis_velocity.png',
    '02_chassis_acceleration.png',
    '03_tracking_error.png',
    '04_wheel_steer.png',
    '05_ball_curvature.png',
]

saved_files = []
for idx, num in enumerate(fignums):
    fig_obj = plt.figure(num)
    if idx < len(desired_names):
        fname = desired_names[idx]
    else:
        fname = f'figure_{idx+1:02d}.png'
    outpath = run_dir / fname
    fig_obj.savefig(str(outpath), bbox_inches='tight')
    saved_files.append(outpath.name)
    print(f'Saved figure to {outpath}')

print(f'Saved data figures to {run_dir}')

# Also save animation (try MP4 first, fallback to GIF)
try:
    out_mp4 = run_dir / 'swerve_animation.mp4'
    ani.save(str(out_mp4), fps=max(1, int(1.0/DT)))
    saved_files.insert(0, out_mp4.name)
    print(f'Saved animation to {out_mp4}')
except Exception as e:
    try:
        out_gif = run_dir / 'swerve_animation.gif'
        ani.save(str(out_gif), writer='pillow', fps=max(1, int(1.0/DT)))
        saved_files.insert(0, out_gif.name)
        print(f'Saved animation to {out_gif}')
    except Exception as e2:
        print(f'Failed to save animation: {e}, {e2}')

# Generate HTML index for this run
html_path = run_dir / 'index.html'
with open(html_path, 'w', encoding='utf-8') as f:
    f.write('<!doctype html>\n<html><head><meta charset="utf-8"><title>Run Results</title></head><body>\n')
    f.write(f'<h1>Simulation Run #{run_num}</h1>\n')
    f.write('<table border="1" cellpadding="6">\n')
    f.write('<tr><th>#</th><th>File</th><th>Preview</th></tr>\n')
    for i, name in enumerate(saved_files, start=1):
        f.write(f'<tr><td>{i}</td><td>{name}</td><td>')
        lower = name.lower()
        if lower.endswith('.png') or lower.endswith('.gif'):
            f.write(f'<img src="{name}" style="max-height:150px">')
        elif lower.endswith('.mp4'):
            f.write(f'<video src="{name}" controls style="max-height:150px"></video>')
        else:
            f.write('N/A')
        f.write('</td></tr>\n')
    f.write('</table>\n')
    f.write('</body></html>\n')

print(f'Wrote run summary to {html_path}')

# Also generate a master index file listing all runs in the result directory
master_html = result_base / 'index.html'
with open(master_html, 'w', encoding='utf-8') as f:
    f.write('<!doctype html>\n<html><head><meta charset="utf-8"><title>All Runs</title></head><body>\n')
    f.write('<h1>All Simulation Runs</h1>\n')
    f.write('<ul>\n')
    # Extract all run directories and sort numerically
    run_dirs = []
    for d in result_base.glob("第*"):
        if d.is_dir():
            try:
                num = int(d.name[1:-1])
                run_dirs.append((num, d))
            except:
                pass
    run_dirs.sort(key=lambda x: x[0])
    for num, run_d in run_dirs:
        idx_file = run_d / 'index.html'
        if idx_file.exists():
            f.write(f'<li><a href="{run_d.name}/index.html">{run_d.name}</a></li>\n')
    f.write('</ul>\n')
    f.write('</body></html>\n')

print(f'Wrote master index to {master_html}')
