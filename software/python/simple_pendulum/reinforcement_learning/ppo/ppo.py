import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

from simple_pendulum.model.pendulum_plant import PendulumPlant
from simple_pendulum.simulation.simulation import Simulator
from simple_pendulum.simulation.gym_environment import SimplePendulumEnv

class ppo_trainer():
    """
    Class to train a policy for pendulum swingup with the PPO method.
    """
    def __int__(self,log_dir='ppo_training'):
        """
        Class to train a policy for pendulum swingup with the PPO method.
        Parameters
        ----------
        log_dir: string, default="ppo_training"
            path to directory where results and log data will be stored.
        """
        self.log_dir = log_dir

    def init_pendulum(self, mass=0.57288, length=0.5, inertia=None,
                      damping=0.15, coulomb_friction=0.0, gravity=9.81,
                      torque_limit=2.0):
        pass

    def init_environment(self,
                         dt=0.01,
                         integrator="runge_kutta",
                         max_steps=1000,
                         reward_type="soft_binary_with_repellor",
                         state_representation=2,
                         validation_limit=-150,
                         target=[np.pi, 0.0],
                         state_target_epsilon=[1e-2, 1e-2],
                         random_init="everywhere"):
        pass

    def init_agent(self,
                   learning_rate=0.0003,
                   warm_start=False,
                   warm_start_path="",
                   verbose=1):
        pass

    def train(self,
              training_timesteps=1e6,
              reward_threshold=1000.0,
              eval_frequency=10000,
              n_eval_episodes=20,
              verbose=1):
        pass

