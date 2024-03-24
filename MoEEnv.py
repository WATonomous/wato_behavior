# This file is to edit the MetaDriveEnv to render during step
from metadrive.metadrive.envs import MetaDriveEnv
from metadrive.metadrive.envs import TopDownMetaDrive

from mpc import MPCController

from gymnasium.spaces import Discrete

import numpy as np

import math


class MoEEnv(MetaDriveEnv):
    # Override init to modify config
    def __init__(self, config, window=False):
        # Increase speed reward and driving reward from 0.1 to 0.5
        config["speed_reward"] = 0.5
        config["driving_reward"] = 0.5
        self.window = window
        # config["norm_pixel"] = False
        super().__init__(config)

        self.prev_action = np.array([0, 0]).reshape(-1, 1)

    # Override action space to output a discrete value
    # this is the index of the model the router is choosing
    @property
    def action_space(self):
        return Discrete(4) # For now assume 4 models


    # Add render to the environment step
    def step(self, action):
        # Get the chosen model
        model_ind = action

        # Get the vehicle's current lane
        lane = self.vehicle.lane

        # Get the vehicle's current position relative to lane
        long_pos = self.vehicle.lane.local_coordinates(self.vehicle.position)[0]

        # Calculate the angle between the vehicle's heading and the direction of the lane
        angle_error = self.vehicle.heading_theta - lane.heading_theta_at(long_pos + 0.1)

        # Steer the vehicle to minimize the angle error
        steering = -angle_error

        # Set throttle to maintain some speed
        velocity_error = math.sqrt(self.vehicle.velocity[0]**2 + self.vehicle.velocity[1]**2) \
            - 10
        throttle = -velocity_error*0.05
        # print(throttle)

        obs, reward, terminated, truncated, info = super().step([steering, throttle])

        out = self.render(mode="topdown",
            scaling=None,
            film_size=(500, 500),
            screen_size=(2000, 500),
            # target_vehicle_heading_up=True,
            camera_position=(0,0),
            screen_record=False,
            window=self.window,
            text={"episode_step": self.engine.episode_step,
                    "mode": "Trigger"})

        return obs, reward, terminated, truncated, info

        if model_ind == 0:
            # The first model is simple keep lane
            raise NotImplementedError
        elif model_ind == 1:
            # change left lane
            raise NotImplementedError
        elif model_ind == 2:
            raise NotImplementedError

    # Override the reward function
    def reward_function(self, vehicle_id):
        reward, step_info = super().reward_function(vehicle_id)
        return reward, step_info