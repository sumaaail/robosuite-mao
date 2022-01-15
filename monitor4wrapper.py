import csv
import os
import time
from typing import Union, Tuple

import numpy as np
from stable_baselines3.common.monitor import Monitor, ResultsWriter
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
EXTRA_MONITOR = 'extra_info.csv'

class Monitor4wrapper(Monitor):
    def __init__(self,
                 wrappered_env,
                 logdir,
                 extra_print_key: Tuple[str, ...] = (),
                 ):
        super(Monitor4wrapper, self).__init__(env=wrappered_env, filename=logdir)
        # extra information log
        self.extra_print_key = extra_print_key
        self.extra_log_handler = open(os.path.join(logdir, EXTRA_MONITOR), "wt")
        self.extra_logger = csv.DictWriter(self.extra_log_handler, fieldnames=extra_print_key)

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
            #     for other extra info to add
            # for key in self.extra_print_key:

            self.episode_returns.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(ep_info)
            if self.extra_print_key:
                self.extra_logger.writerow({"action_space": action})
                self.extra_log_handler.flush()
            info['episode'] = ep_info
        self.total_steps += 1
        return observation, reward, done, info

