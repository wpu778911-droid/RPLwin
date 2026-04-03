#!/usr/bin/env python

import argparse

from gops.create_pkg.create_alg import create_alg
from gops.create_pkg.create_buffer import create_buffer
from gops.create_pkg.create_env import create_env
from gops.create_pkg.create_evaluator import create_evaluator
from gops.create_pkg.create_sampler import create_sampler
from gops.create_pkg.create_trainer import create_trainer
from gops.utils.init_args import init_args
from gops.utils.plot_evaluation import plot_all
from gops.utils.tensorboard_setup import save_tb_to_csv, start_tensorboard


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env_id", type=str, default="env_paThi_sim2", help="id of environment")
    parser.add_argument("--algorithm", type=str, default="PPO", help="RL algorithm")
    parser.add_argument("--enable_cuda", default=False, help="Enable CUDA")
    parser.add_argument("--seed", type=int, default=12345, help="Global seed")

    parser.add_argument("--is_render", type=bool, default=False, help="Draw environment animation")
    parser.add_argument("--is_adversary", type=bool, default=False, help="Adversary training")
    parser.add_argument("--is_constrained", type=bool, default=False, help="Constrained environment")

    parser.add_argument("--mode", type=str, default="rl", help="rl/rule_preview/nearest_angle")
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--t_total", type=float, default=45.0)
    parser.add_argument("--episode_steps", type=int, default=450)
    parser.add_argument("--traj_type", type=str, default="train_generator")
    parser.add_argument("--random_traj_on_reset", type=bool, default=False)
    parser.add_argument("--random_ref_time", type=bool, default=True)
    parser.add_argument("--n_preview", type=int, default=5)
    parser.add_argument("--v_wheel_max", type=float, default=0.35)
    parser.add_argument("--w_max", type=float, default=1.5)
    parser.add_argument("--v_max", type=float, default=0.35)
    parser.add_argument("--a_max", type=float, default=0.35)
    parser.add_argument("--preview_cost_scale", type=float, default=0.2)
    parser.add_argument("--danger_margin_deg", type=float, default=12.0)
    parser.add_argument("--kp_pos_x", type=float, default=1.2)
    parser.add_argument("--kp_pos_y", type=float, default=1.6)
    parser.add_argument("--kd_vel_x", type=float, default=0.6)
    parser.add_argument("--kd_vel_y", type=float, default=0.6)
    parser.add_argument("--kp_yaw", type=float, default=2.2)
    parser.add_argument("--init_pos_noise", type=float, default=0.15)
    parser.add_argument("--init_yaw_noise_deg", type=float, default=12.0)
    parser.add_argument("--init_steer_noise_deg", type=float, default=20.0)

    parser.add_argument("--value_func_name", type=str, default="StateValue")
    parser.add_argument("--value_func_type", type=str, default="MLP")
    parser.add_argument("--value_hidden_sizes", type=list, default=[256, 256, 256])
    parser.add_argument("--value_hidden_activation", type=str, default="tanh")
    parser.add_argument("--value_output_activation", type=str, default="linear")

    parser.add_argument("--policy_func_name", type=str, default="StochaPolicyDis")
    parser.add_argument("--policy_func_type", type=str, default="MLP")
    parser.add_argument("--policy_act_distribution", type=str, default="default")
    parser.add_argument("--policy_hidden_sizes", type=list, default=[256, 256, 256])
    parser.add_argument("--policy_hidden_activation", type=str, default="tanh")
    parser.add_argument("--policy_output_activation", type=str, default="linear")

    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--loss_coefficient_kl", type=float, default=0.0)
    parser.add_argument("--loss_coefficient_value", type=float, default=0.5)
    parser.add_argument("--loss_coefficient_entropy", type=float, default=0.01)
    parser.add_argument("--advantage_norm", type=bool, default=True)
    parser.add_argument("--loss_value_clip", type=bool, default=False)
    parser.add_argument("--loss_value_norm", type=bool, default=False)
    parser.add_argument("--schedule_adam", type=str, default="None")
    parser.add_argument("--schedule_clip", type=str, default="None")

    parser.add_argument("--trainer", type=str, default="on_serial_trainer")
    parser.add_argument("--max_iteration", type=int, default=1000)
    parser.add_argument("--ini_network_dir", type=str, default=None)
    parser.add_argument("--num_repeat", type=int, default=4)
    parser.add_argument("--num_mini_batch", type=int, default=4)
    parser.add_argument("--mini_batch_size", type=int, default=64)
    parser.add_argument("--num_epoch", type=int, default=16)

    parser.add_argument("--sampler_name", type=str, default="on_sampler")
    parser.add_argument("--sample_batch_size", type=int, default=256)
    parser.add_argument("--noise_params", type=dict, default={"epsilon": 0.0})

    parser.add_argument("--buffer_name", type=str, default="replay_buffer")
    parser.add_argument("--buffer_warm_size", type=int, default=1024)
    parser.add_argument("--buffer_max_size", type=int, default=200000)

    parser.add_argument("--evaluator_name", type=str, default="evaluator")
    parser.add_argument("--num_eval_episode", type=int, default=2)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--eval_save", type=str, default=False)

    parser.add_argument("--save_folder", type=str, default="results/PPO_paThi_sim2")
    parser.add_argument("--apprfunc_save_interval", type=int, default=100)
    parser.add_argument("--log_save_interval", type=int, default=10)

    args = vars(parser.parse_args())
    assert args["num_mini_batch"] * args["mini_batch_size"] == args["sample_batch_size"], "sample_batch_size error"

    env = create_env(**args)
    args = init_args(env, **args)

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
