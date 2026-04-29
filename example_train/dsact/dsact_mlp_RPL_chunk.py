#!/usr/bin/env python
#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Description: example for dsac-t + RPL FWT sim (action chunking env) + mlp + offserial

import argparse
import json
import os
import shutil
from datetime import datetime

import numpy as np

from gops.create_pkg.create_alg import create_alg
from gops.create_pkg.create_buffer import create_buffer
from gops.create_pkg.create_env import create_env
from gops.create_pkg.create_evaluator import create_evaluator
from gops.create_pkg.create_sampler import create_sampler
from gops.create_pkg.create_trainer import create_trainer
from gops.utils.init_args import init_args
from gops.utils.plot_evaluation import plot_all
from gops.utils.tensorboard_setup import start_tensorboard, save_tb_to_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ################################################
    # Key Parameters for users
    parser.add_argument("--env_id", type=str, default="env_RPL_FWTsim_chunk003", help="id of environment")
    parser.add_argument("--algorithm", type=str, default="DSACT", help="RL algorithm")
    parser.add_argument("--enable_cuda", default=False, help="Enable CUDA")
    parser.add_argument("--seed", default=12345, help="Global seed")

    ################################################
    # 1. Parameters for environment
    parser.add_argument("--reward_scale", type=float, default=1.0, help="reward scale factor")
    parser.add_argument("--is_render", type=bool, default=False, help="Draw environment animation")
    parser.add_argument("--is_adversary", type=bool, default=False, help="Adversary training")

    ################################################
    # 2.1 Parameters of value approximate function
    parser.add_argument(
        "--value_func_name",
        type=str,
        default="ActionValueDistri",
        help="Options: StateValue/ActionValue/ActionValueDis/ActionValueDistri",
    )
    parser.add_argument("--value_func_type", type=str, default="MLP", help="Options: MLP/CNN/CNN_SHARED/RNN/POLY/GAUSS")
    parser.add_argument("--value_hidden_sizes", type=list, default=[256, 256, 256])
    parser.add_argument(
        "--value_hidden_activation", type=str, default="gelu", help="Options: relu/gelu/elu/selu/sigmoid/tanh"
    )
    parser.add_argument("--value_output_activation", type=str, default="linear", help="Options: linear/tanh")

    ################################################
    # 2.2 Parameters of policy approximate function
    parser.add_argument(
        "--policy_func_name",
        type=str,
        default="StochaPolicy",
        help="Options: None/DetermPolicy/FiniteHorizonPolicy/StochaPolicy",
    )
    parser.add_argument(
        "--policy_func_type", type=str, default="MLP", help="Options: MLP/CNN/CNN_SHARED/RNN/POLY/GAUSS"
    )
    parser.add_argument(
        "--policy_act_distribution",
        type=str,
        default="TanhGaussDistribution",
        help="Options: default/TanhGaussDistribution/GaussDistribution",
    )
    parser.add_argument("--policy_hidden_sizes", type=list, default=[256, 256, 256])
    parser.add_argument(
        "--policy_hidden_activation", type=str, default="gelu", help="Options: relu/gelu/elu/selu/sigmoid/tanh"
    )
    parser.add_argument("--policy_output_activation", type=str, default="linear", help="Options: linear/tanh")
    parser.add_argument("--policy_min_log_std", type=int, default=-20)
    parser.add_argument("--policy_max_log_std", type=int, default=0.5)

    ################################################
    # 3. Parameters for RL algorithm
    parser.add_argument("--value_learning_rate", type=float, default=0.0001)
    parser.add_argument("--policy_learning_rate", type=float, default=0.0001)
    parser.add_argument("--alpha_learning_rate", type=float, default=0.0003)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--auto_alpha", type=bool, default=True)
    parser.add_argument("--delay_update", type=int, default=2)

    ################################################
    # 4. Parameters for trainer
    parser.add_argument(
        "--trainer",
        type=str,
        default="off_serial_trainer",
        help="Options: on_serial_trainer, on_sync_trainer, off_serial_trainer, off_async_trainer",
    )
    parser.add_argument("--max_iteration", type=int, default=1500000)
    parser.add_argument("--ini_network_dir", type=str, default=None)

    ################################################
    # 4.1. Parameters for off_serial_trainer
    parser.add_argument(
        "--buffer_name", type=str, default="replay_buffer", help="Options: replay_buffer/prioritized_replay_buffer"
    )
    parser.add_argument("--buffer_warm_size", type=int, default=10000)
    parser.add_argument("--buffer_max_size", type=int, default=2 * 500000)
    parser.add_argument("--replay_batch_size", type=int, default=256)
    parser.add_argument("--sample_interval", type=int, default=1)

    ################################################
    # 5. Parameters for sampler
    parser.add_argument("--sampler_name", type=str, default="off_sampler", help="Options: on_sampler/off_sampler")
    parser.add_argument("--sample_batch_size", type=int, default=20)
    parser.add_argument("--noise_params", type=dict, default=None)

    ################################################
    # 6. Parameters for evaluator
    parser.add_argument("--evaluator_name", type=str, default="evaluator")
    parser.add_argument("--num_eval_episode", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=2500)
    parser.add_argument("--eval_save", type=str, default=False, help="save evaluation data")

    ################################################
    # 7. Data savings
    parser.add_argument("--save_folder", type=str, default="results/RPL_FWTsim_chunk003")
    parser.add_argument("--apprfunc_save_interval", type=int, default=50000)
    parser.add_argument("--log_save_interval", type=int, default=10000)
    parser.add_argument("--save_env_snapshot", type=bool, default=True, help="Save env code snapshot")

    args = vars(parser.parse_args())
    env = create_env(**args)
    args = init_args(env, **args)

    if args.get("save_env_snapshot", True):
        stamp = datetime.now().strftime("%y%m%d-%H%M%S")
        snapshot_dir = os.path.join(args["save_folder"], "env_snapshots", stamp)
        os.makedirs(snapshot_dir, exist_ok=True)
        env_id = args.get("env_id", "")
        env_map = {
            "env_RPL_FWTsim001": "gops/env/env_FwFtracking/env_RPL_FWTsim.py",
            "env_RPL_FWTsim_chunk": "gops/env/env_FwFtracking/env_RPL_FWTsim_chunk.py",
            "env_RPL_FWTsim_chunk001": "gops/env/env_FwFtracking/env_RPL_FWTsim_chunk001.py",
            "env_RPL_FWTsim_chunk003": "gops/env/env_FwFtracking/env_RPL_FWTsim_chunk003.py",
            "env_RPL_FWTsim_trackfirst": "gops/env/env_FwFtracking/env_RPL_FWTsim_trackfirst.py",
            "env_RPL_FWTsim_trackfirst001": "gops/env/env_FwFtracking/env_RPL_FWTsim_trackfirst001.py",
            "env_ToRwheelsim": "gops/env/env_FwFtracking/env_ToRwheelsim.py",
            "MPC_RL_env_ToRwheelsim": "gops/env/env_FwFtracking/MPC_RL_env_ToRwheelsim.py",
        }
        env_path = env_map.get(env_id)
        if env_path and os.path.isfile(env_path):
            shutil.copy2(env_path, os.path.join(snapshot_dir, os.path.basename(env_path)))

        def _json_default(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        with open(os.path.join(snapshot_dir, "train_args.json"), "w", encoding="utf-8") as f:
            json.dump(args, f, indent=2, ensure_ascii=False, default=_json_default)

    start_tensorboard(args["save_folder"])
    alg = create_alg(**args)
    sampler = create_sampler(**args)
    buffer = create_buffer(**args)
    evaluator = create_evaluator(**args)
    trainer = create_trainer(alg, sampler, buffer, evaluator, **args)

    trainer.train()
    print("Training is finished!")

    plot_all(args["save_folder"])
    save_tb_to_csv(args["save_folder"])
    print("Plot & Save are finished!")
