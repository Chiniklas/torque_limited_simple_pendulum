import os
import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

from simple_pendulum.model.pendulum_plant import PendulumPlant
from simple_pendulum.simulation.simulation import Simulator
from software.python.simple_pendulum.simulation.gym_environment import SimplePendulumEnv

class td3_trainer():
    """
    Class to train a policy for pendulum swingup with the PPO method.
    """

    def __init__(self,
                 log_dir="td3_training"):
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
        self.pen_mass = mass
        self.pen_length = length
        if inertia is None:
            inertia = mass * length ** 2
        self.pen_inertia = inertia
        self.pen_damping = damping
        self.pen_cfric = coulomb_friction
        self.pen_gravity = gravity
        self.pen_torque_limit = torque_limit

        self.pendulum = PendulumPlant(mass=self.pen_mass,
                                      length=self.pen_length,
                                      damping=self.pen_damping,
                                      gravity=self.pen_gravity,
                                      coulomb_fric=self.pen_cfric,
                                      inertia=self.pen_inertia,
                                      torque_limit=self.pen_torque_limit)

        self.simulator = Simulator(plant=self.pendulum)

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

        self.env = SimplePendulumEnv(simulator=self.simulator,
                                     max_steps=max_steps,
                                     reward_type=reward_type,
                                     dt=dt,
                                     integrator=integrator,
                                     state_representation=state_representation,
                                     validation_limit=validation_limit,
                                     scale_action=True,
                                     random_init=random_init)

        # setup evaluation environment
        self.eval_env = SimplePendulumEnv(
            simulator=self.simulator,
            max_steps=max_steps,
            reward_type=reward_type,
            dt=dt,
            integrator=integrator,
            state_representation=state_representation,
            validation_limit=validation_limit,
            scale_action=True,
            random_init="False")

    def init_agent(self,
                   learning_rate=0.0003,
                   warm_start=False,
                   warm_start_path="",
                   verbose=1):
        tensorboard_log = os.path.join(self.log_dir, "tb_logs")
        self.agent = TD3(MlpPolicy,
                         self.env,
                         verbose=verbose,
                         tensorboard_log=tensorboard_log,
                         learning_rate=learning_rate)
        if warm_start:
            self.agent.set_parameters(load_path_or_dict=warm_start_path)
    def train(self,
              training_timesteps=1e6,
              reward_threshold=1000.0,
              eval_frequency=10000,
              n_eval_episodes=20,
              verbose=1):

        # define training callbacks
        callback_on_best = StopTrainingOnRewardThreshold(
            reward_threshold=reward_threshold,
            verbose=verbose)

        log_path = os.path.join(self.log_dir, 'best_model')

        eval_callback = EvalCallback(
            self.eval_env,
            callback_on_new_best=callback_on_best,
            best_model_save_path=log_path,
            log_path=log_path,
            eval_freq=eval_frequency,
            verbose=verbose,
            n_eval_episodes=n_eval_episodes)
        # train
        self.agent.learn(total_timesteps=int(training_timesteps),
                         callback=eval_callback)