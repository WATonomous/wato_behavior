# This file is to edit the MetaDriveEnv to render during step
from metadrive.envs import MetaDriveEnv
from metadrive.envs import TopDownMetaDrive

from mpc import MPCController

from gymnasium.spaces import Dict, Box

import numpy as np

import math


class BehaviourEnv(MetaDriveEnv):
    # Override init to modify config
    def __init__(self, config, window=False):
        # Increase speed reward and driving reward from 0.1 to 0.5
        config["speed_reward"] = 0.5
        config["driving_reward"] = 0.5
        self.window = window
        # config["norm_pixel"] = False
        super().__init__(config)

        self.prev_action = np.array([0, 0]).reshape(-1, 1)

    # Override action space to output 4 numbers
    @property
    def action_space(self):
        # spaces_dict = {
        #     'x_pos': Box(low=0.0, high=10.0),
        #     'y_pos': Box(low=-5.0, high=-5.0),
        #     'theta': Box(low=-np.pi/3.0, high=np.pi/3.0),
        #     'vel': Box(low=0.0, high=10.0)
        # }
        return Box(low=-1.0, high=1.0, shape=(1,3))


    # Add render to the environment step
    def step(self, action):
        # action[1] = max(0.1, action[1])

        # print([
        #     (action[0][0] + 1.0) * 5.0, 
        #     action[0][1] * 5.0,
        #     action[0][2] * np.pi/3.0,
        #     (action[0][3] + 1.0) * 5.0
        #     ])

        # Take the action (target position, angle and velocity) and give to mpc
        mpc_controller = MPCController()
        # mpc_controller.set_objective(
        #     (action[0][0] + 1.0),
        #     action[0][1],
        #     action[0][2] * np.pi/3.0,
        #     (action[0][3] + 1.0) * 5.0
        #     )
        target_x, target_y = mpc_controller.set_objective(
            action[0][0] * np.pi/3.0,
            action[0][1] * np.pi/3.0,
            (action[0][2] + 1.0)
            )
        mpc = mpc_controller.mpc
        cur_vel = self.agent.velocity
        x0 = np.array([0, 0, 0, math.sqrt(cur_vel[0]**2 + cur_vel[1]**2)]).reshape(-1,1)
        u0 = self.prev_action
        mpc.x0 = x0
        mpc.u0 = u0
        mpc.set_initial_guess()

        init_x = self.agent.position[0]
        init_y = self.agent.position[1]

        # Compute actual control values
        u0 = mpc.make_step(x0)

        control_action = np.array([u0[1][0], u0[0][0]])

        

        MAX_STEPS = 70
        i = 0
        while x0[0][0] < target_x and i < MAX_STEPS:
            # print(control_action)
            obs, reward, terminated, truncated, info = super().step(control_action)
            x0[0] = self.agent.position[0] - init_x
            x0[1] = self.agent.position[1] - init_y
            x0[2] = math.atan(x0[1]/x0[0])
            x0[3] = math.sqrt(self.agent.velocity[0]**2 + self.agent.velocity[1]**2)
            u0 = mpc.make_step(x0)
            control_action = np.array([u0[1][0], u0[0][0]])
            if terminated or truncated:
                break
            i += 1

            # out = self.render(mode="topdown", 
            # scaling=2, 
            # camera_position=(100, 0), 
            # screen_size=(500, 500),
            # screen_record=False,
            # window=self.window,
            # text={"episode_step": self.engine.episode_step,
            #         "mode": "Trigger"})
            
        self.prev_action = np.copy(u0)


        return obs, reward, terminated, truncated, info

    # Override the reward function
    def reward_function(self, vehicle_id):
        reward, step_info = super().reward_function(vehicle_id)

        # Keep the same reward as vanilla implementation but add scaled time factor
        # if self.config["horizon"] is not None:
        #     reward -= 2.0 * self.episode_lengths[vehicle_id] / self.config["horizon"]
        # step_info["step_reward"] = reward


        return reward, step_info



