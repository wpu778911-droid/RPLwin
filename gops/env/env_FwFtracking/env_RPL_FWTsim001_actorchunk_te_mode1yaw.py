from gops.env.env_FwFtracking.env_RPL_FWTsim import ToRwheelsimSeqScanEnv


def env_creator(**kwargs):
    # Temporal-ensembled receding-horizon actor chunking:
    # actor outputs 3 actions, env executes one closed-loop blended action.
    kwargs.setdefault("action_horizon", 3)

    # Let the actor observe its previous unexecuted plan tail.
    kwargs.setdefault("include_prev_plan_in_obs", True)
    kwargs.setdefault("prev_plan_obs_steps", 2)

    # Penalize contradicting the previous tail plan and also blend old/current
    # predictions for the actually executed action.
    kwargs.setdefault("plan_shift_weight", 0.010)
    kwargs.setdefault("temporal_ensemble_alpha", 0.35)
    kwargs.setdefault("temporal_ensemble_decay", 0.50)

    # Same Mode1/yaw focus as the previous actor-chunking version.
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

    kwargs.setdefault("mode2_proto_scale", 0.55)

    return ToRwheelsimSeqScanEnv(**kwargs)
