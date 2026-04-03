from gops.env.env_FwFtracking.env_RPL_FWTsim import ToRwheelsimSeqScanEnv


def env_creator(**kwargs):
    return ToRwheelsimSeqScanEnv(**kwargs)
