from gops.env.env_FwFtracking.env_RPL_FWTsim import ToRwheelsimSeqScanEnv


def env_creator(**kwargs):
    # Receding-horizon actor chunking:
    # actor outputs 3 wheel-action commands, env executes only the first one.
    kwargs.setdefault("action_horizon", 3)

    # Make the two unexecuted actions useful: expose the previous tail plan to
    # the next observation and reward the next output for staying consistent.
    kwargs.setdefault("include_prev_plan_in_obs", True)
    kwargs.setdefault("prev_plan_obs_steps", 2)
    kwargs.setdefault("plan_shift_weight", 0.012)

    # Same intent as the Mode1/yaw cleanup: keep the straight-line solution
    # close to the MPC-like 90-degree wheel alignment and reduce yaw error.
    kwargs.setdefault("yaw_huber_weight", 8.0)
    kwargs.setdefault("yaw_cos_weight", 4.5)
    kwargs.setdefault("yaw_progress_weight", 4.5)

    kwargs.setdefault("mode1_proto_peak_weight", 3.9)
    kwargs.setdefault("mode1_target_pen_weight", 1.55)
    kwargs.setdefault("mode1_spread_pen_weight", 0.90)
    kwargs.setdefault("mode1_speed_pen_weight", 0.30)

    kwargs.setdefault("mode1_safe_target_weight", 0.32)
    kwargs.setdefault("mode1_safe_speed_weight", 0.12)
    kwargs.setdefault("mode1_safe_steer_weight", 0.14)

    # Keep Mode2 as a weak hint, not the main driver for this experiment.
    kwargs.setdefault("mode2_proto_scale", 0.55)

    return ToRwheelsimSeqScanEnv(**kwargs)
