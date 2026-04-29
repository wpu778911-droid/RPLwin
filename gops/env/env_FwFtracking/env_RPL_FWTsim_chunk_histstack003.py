from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np
from gym import spaces

from gops.env.env_FwFtracking.env_RPL_FWTsim_chunk import ToRwheelsimChunkEnv


class ToRwheelsimChunkHistStackEnv(ToRwheelsimChunkEnv):
    """3-step action chunking with a flat stacked observation history for MLP policies."""

    def __init__(self, *args, history_length: int = 4, **kwargs):
        kwargs = dict(kwargs)
        kwargs.setdefault("action_horizon", 3)
        kwargs.setdefault("chunk_execute_length", 3)
        super().__init__(*args, **kwargs)

        self.history_length = int(history_length)
        if self.history_length < 1:
            raise ValueError("history_length must be >= 1.")

        base_obs_dim = int(self.observation_space.shape[0])
        self._base_obs_dim = base_obs_dim
        self._obs_history: Deque[np.ndarray] = deque(maxlen=self.history_length)
        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * (base_obs_dim * self.history_length), dtype=np.float32),
            high=np.array([np.inf] * (base_obs_dim * self.history_length), dtype=np.float32),
            dtype=np.float32,
        )

    def _get_stacked_obs(self) -> np.ndarray:
        if len(self._obs_history) != self.history_length:
            raise RuntimeError("Observation history is not initialized.")
        return np.concatenate(list(self._obs_history), axis=0).astype(np.float32)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        init_state=None,
        ref_time: Optional[float] = None,
    ) -> Tuple[np.ndarray, dict]:
        obs, info = super().reset(seed=seed, options=options, init_state=init_state, ref_time=ref_time)
        self._obs_history.clear()
        for _ in range(self.history_length):
            self._obs_history.append(np.asarray(obs, dtype=np.float32).copy())
        return self._get_stacked_obs(), info

    def step(self, action: np.ndarray):
        obs, reward, done, info = super().step(action)
        self._obs_history.append(np.asarray(obs, dtype=np.float32).copy())
        info["history_length"] = int(self.history_length)
        return self._get_stacked_obs(), reward, done, info


def env_creator(**kwargs):
    return ToRwheelsimChunkHistStackEnv(**kwargs)
