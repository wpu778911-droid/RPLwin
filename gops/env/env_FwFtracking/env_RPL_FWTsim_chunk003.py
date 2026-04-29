from gops.env.env_FwFtracking.env_RPL_FWTsim_chunk import ToRwheelsimChunkEnv


def env_creator(**kwargs):
    kwargs = dict(kwargs)
    kwargs.setdefault("action_horizon", 3)
    kwargs.setdefault("chunk_execute_length", 3)
    return ToRwheelsimChunkEnv(**kwargs)
