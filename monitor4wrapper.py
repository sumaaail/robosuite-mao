import time
from types import Union

import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn


class Monitor4wrapper(Monitor):
    def __init__(self, wrappered_env, logdir):
        super(Monitor4wrapper, self).__init__(env=wrappered_env,filename=logdir)
    def reset(self, **kwargs) -> GymObs:
        self.rewards = []
        return self.env.reset()
    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)
        if done:
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)
            ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time()-self.t_start, 6)}
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_returns.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(ep_info)
            info['episode'] = ep_info
        self.total_steps += 1
        return observation, reward, done, info

