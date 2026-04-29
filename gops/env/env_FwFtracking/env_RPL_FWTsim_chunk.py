from typing import Dict, List, Optional, Tuple

import numpy as np
from gym import spaces

from gops.env.env_FwFtracking.env_RPL_FWTsim import (
    ToRwheelsimSeqScanEnv,
    WHEEL_ORDER,
    wrap_to_pi,
)
from gops.env.env_gen_ocp.pyth_base import State


class ToRwheelsimChunkEnv(ToRwheelsimSeqScanEnv):
    """Execute the full action chunk sequentially before returning control."""

    def __init__(
        self,
        *args,
        action_horizon: int = 2,
        chunk_execute_length: Optional[int] = None,
        **kwargs,
    ):
        # Reuse the stable base env implementation, then override the action horizon
        # and action space for chunk execution.
        super().__init__(*args, action_horizon=3, **kwargs)
        self.gamma = float(kwargs.get("gamma", 0.99))
        self.action_horizon = int(action_horizon)
        if self.action_horizon < 1:
            raise ValueError("action_horizon must be >= 1.")

        low_step = np.array(
            [-self.robot.steer_rate_max, -self.robot.wheel_acc_max] * 4, dtype=np.float32
        )
        high_step = np.array(
            [self.robot.steer_rate_max, self.robot.wheel_acc_max] * 4, dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.tile(low_step, self.action_horizon),
            high=np.tile(high_step, self.action_horizon),
            dtype=np.float32,
        )

        self.chunk_execute_length = (
            self.action_horizon if chunk_execute_length is None else int(chunk_execute_length)
        )
        if self.chunk_execute_length < 1 or self.chunk_execute_length > self.action_horizon:
            raise ValueError("chunk_execute_length must be in [1, action_horizon].")

    @property
    def additional_info(self) -> Dict[str, Dict]:
        return {
            "discount_steps": {"shape": (), "dtype": np.float32},
            "chunk_executed_steps": {"shape": (), "dtype": np.float32},
        }

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        info = dict(info)
        info["discount_steps"] = np.float32(1.0)
        info["chunk_executed_steps"] = np.float32(0.0)
        return obs, info

    def _remaining_action_seq(self, action_seq: np.ndarray, start_idx: int) -> np.ndarray:
        tail = action_seq[start_idx:]
        if tail.shape[0] >= self.action_horizon:
            return tail[: self.action_horizon]
        pad = np.repeat(tail[-1:, :], self.action_horizon - tail.shape[0], axis=0)
        return np.concatenate([tail, pad], axis=0)

    def _chunk_internal_terms(
        self,
        executed_actions: List[np.ndarray],
        executed_steers: List[np.ndarray],
        step_terms: List[Dict[str, float]],
    ) -> Dict[str, float]:
        if len(executed_actions) <= 1:
            return {
                "chunk_action_delta_pen": 0.0,
                "chunk_steer_delta_pen": 0.0,
                "chunk_mode_continuity_pen": 0.0,
                "chunk_internal_penalty": 0.0,
            }

        act_norm = np.array(
            [
                self.robot.steer_rate_max,
                self.robot.wheel_acc_max,
                self.robot.steer_rate_max,
                self.robot.wheel_acc_max,
                self.robot.steer_rate_max,
                self.robot.wheel_acc_max,
                self.robot.steer_rate_max,
                self.robot.wheel_acc_max,
            ],
            dtype=np.float32,
        )
        steer_angle_scale = max(np.deg2rad(15.0), 1e-6)

        action_delta_pen = 0.0
        steer_delta_pen = 0.0
        mode_continuity_pen = 0.0

        for idx in range(1, len(executed_actions)):
            prev_action = executed_actions[idx - 1]
            curr_action = executed_actions[idx]
            da = (curr_action - prev_action) / np.maximum(act_norm, 1e-6)
            action_delta_pen += float(np.mean(da**2))

            prev_steer = executed_steers[idx - 1]
            curr_steer = executed_steers[idx]
            dsteer = np.array(
                [wrap_to_pi(float(c - p)) for p, c in zip(prev_steer, curr_steer)],
                dtype=np.float32,
            )
            steer_delta_pen += float(np.mean((dsteer / steer_angle_scale) ** 2))

            prev_mode_raw = float(step_terms[idx - 1].get("mode_raw", 0.0))
            curr_mode_raw = float(step_terms[idx].get("mode_raw", 0.0))
            prev_alpha_1 = float(step_terms[idx - 1].get("mode_alpha_1", 0.0))
            curr_alpha_1 = float(step_terms[idx].get("mode_alpha_1", 0.0))
            prev_alpha_2 = float(step_terms[idx - 1].get("mode_alpha_2", 0.0))
            curr_alpha_2 = float(step_terms[idx].get("mode_alpha_2", 0.0))
            active_weight = 0.5 * (
                max(prev_alpha_1, prev_alpha_2) + max(curr_alpha_1, curr_alpha_2)
            )
            mode_continuity_pen += float(active_weight * (curr_mode_raw - prev_mode_raw) ** 2)

        denom = float(len(executed_actions) - 1)
        action_delta_pen /= denom
        steer_delta_pen /= denom
        mode_continuity_pen /= denom
        internal_penalty = (
            0.020 * action_delta_pen
            + 0.018 * steer_delta_pen
            + 0.060 * mode_continuity_pen
        )
        return {
            "chunk_action_delta_pen": float(action_delta_pen),
            "chunk_steer_delta_pen": float(steer_delta_pen),
            "chunk_mode_continuity_pen": float(mode_continuity_pen),
            "chunk_internal_penalty": float(internal_penalty),
        }

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        action_seq, _ = self._split_action(action)
        total_reward = 0.0
        terminated = False
        term_sums: Dict[str, float] = {}
        final_terms = {}
        executed_steps = 0
        executed_actions: List[np.ndarray] = []
        executed_steers: List[np.ndarray] = []
        step_terms: List[Dict[str, float]] = []

        for chunk_idx in range(self.chunk_execute_length):
            action_exec = action_seq[chunk_idx].copy()
            remaining_action_seq = self._remaining_action_seq(action_seq, chunk_idx)

            robot_state_next = self.robot.step(action_exec)
            context_state_next = self.context.step()
            self._state = State(robot_state=robot_state_next, context_state=context_state_next)

            self._elapsed_steps += 1
            reward, terms = self._compute_reward(
                remaining_action_seq, action_exec, update_cache=True
            )
            terminated = self._get_terminated()
            if terminated:
                reward -= self.termination_penalty

            total_reward += (self.gamma ** chunk_idx) * float(reward)
            final_terms = terms
            executed_steps += 1
            executed_actions.append(action_exec.copy())
            executed_steers.append(
                np.array([self.robot._steer[n] for n in WHEEL_ORDER], dtype=np.float32)
            )
            step_terms.append(dict(terms))
            for key, value in terms.items():
                term_sums[key] = term_sums.get(key, 0.0) + float(value)

            if terminated:
                break

        chunk_internal_terms = self._chunk_internal_terms(
            executed_actions, executed_steers, step_terms
        )
        total_reward -= float(chunk_internal_terms["chunk_internal_penalty"])

        if executed_steps > 0:
            mean_terms = {key: value / executed_steps for key, value in term_sums.items()}
        else:
            mean_terms = dict(final_terms)
        mean_terms.update(chunk_internal_terms)

        self._debug_reward_terms = mean_terms
        obs = self._get_obs()
        info = self._get_info()
        info.update(mean_terms)
        info["flip_total"] = int(self._flip_total)
        info["elapsed_steps"] = int(self._elapsed_steps)
        info["discount_steps"] = np.float32(executed_steps if executed_steps > 0 else 1.0)
        info["chunk_reward_discount"] = float(self.gamma)
        info["chunk_execute_length"] = int(self.chunk_execute_length)
        info["chunk_executed_steps"] = int(executed_steps)
        return obs, float(total_reward), terminated, info

    def _get_reward(self, action: np.ndarray) -> float:
        action_seq, _ = self._split_action(action)
        total_reward = 0.0
        executed_actions: List[np.ndarray] = []
        executed_steers: List[np.ndarray] = []
        step_terms: List[Dict[str, float]] = []

        for chunk_idx in range(self.chunk_execute_length):
            action_exec = action_seq[chunk_idx].copy()
            remaining_action_seq = self._remaining_action_seq(action_seq, chunk_idx)
            reward, terms = self._compute_reward(
                remaining_action_seq, action_exec, update_cache=False
            )
            total_reward += (self.gamma ** chunk_idx) * float(reward)
            executed_actions.append(action_exec.copy())
            executed_steers.append(
                np.array([self.robot._steer[n] for n in WHEEL_ORDER], dtype=np.float32)
            )
            step_terms.append(dict(terms))

        chunk_internal_terms = self._chunk_internal_terms(
            executed_actions, executed_steers, step_terms
        )
        total_reward -= float(chunk_internal_terms["chunk_internal_penalty"])

        return float(total_reward)


def env_creator(**kwargs):
    return ToRwheelsimChunkEnv(**kwargs)
