import os

import gym
import numpy as np

from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

class SaveOnBestEpisodeRewardCallback(BaseCallback):
    '''
    param: check_freq
    param: log_dir
    param: verbose: verbosity level
    '''
    def __init__(self, check_freq:int, log_dir:str, verbose:int=1):
        super(SaveOnBestEpisodeRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model.zip')
        self.best_mean_reward = -np.inf

    def _init_callback(self)->None:
        # create folder if necessary
        # if self.save_path is not None:
        #     os.makedirs(self.save_path, exist_ok=True)
        pass

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            mean_reward = np.mean(y[-100:])
            if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward : {} - Last mean reward per episode: {}".format(self.best_mean_reward, mean_reward))

                # a new best model, save it
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                    self.model.save(self.save_path)
        return True