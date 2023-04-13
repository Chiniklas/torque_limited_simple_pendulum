"""
PPO Controller
==============
"""

# Other imports
import numpy as np
from stable_baselines3 import PPO

# local imports
from simple_pendulum.controllers.abstract_controller import AbstractController


class PPOController(AbstractController):
    def __init__(self,
                 model_path,
                 torque_limit,
                 use_symmetry=True,
                 state_representation=2):

        self.model = PPO.load(model_path)
        self.torque_limit = float(torque_limit)
        self.use_symmetry = bool(use_symmetry)
        self.state_representation = state_representation

        if state_representation == 2:
            # state is [th, th, vel]
            self.low = np.array([-6 * 2 * np.pi, -20])
            self.high = np.array([6 * 2 * np.pi, 20])
        elif state_representation == 3:
            # state is [cos(th), sin(th), vel]
            self.low = np.array([-1., -1., -8.])
            self.high = np.array([1., 1., 8.])

    def get_control_output(self, meas_pos, meas_vel, meas_tau=0, meas_time=0):
        pos = float(np.squeeze(meas_pos))
        vel = float(np.squeeze(meas_vel))

        # map meas pos to [-np.pi, np.pi]
        meas_pos_mod = np.mod(pos + np.pi, 2 * np.pi) - np.pi
        # observation = np.squeeze(np.array([meas_pos_mod, vel]))
        observation = self.get_observation([meas_pos_mod, vel])

        if self.use_symmetry:
            observation[0] *= np.sign(meas_pos_mod)
            observation[1] *= np.sign(meas_pos_mod)
            des_tau, _states = self.model.predict(observation)
            des_tau *= np.sign(meas_pos_mod)
        else:
            des_tau, _states = self.model.predict(observation)
        des_tau *= self.torque_limit

        # since this is a pure torque controller,
        # set pos_des and vel_des to None
        des_pos = None
        des_vel = None

        if np.abs(meas_pos) < 1e-4 and np.abs(meas_vel) < 1e-4:
            des_tau = self.torque_limit

        return des_pos, des_vel, float(des_tau)

    def get_observation(self, state):
        st = np.copy(state)
        st[1] = np.clip(st[1], self.low[-1], self.high[-1])
        if self.state_representation == 2:
            observation = np.array([obs for obs in st], dtype=np.float32)
        elif self.state_representation == 3:
            observation = np.array([np.cos(st[0]),
                                    np.sin(st[0]),
                                    st[1]],
                                   dtype=np.float32)

        return observation
