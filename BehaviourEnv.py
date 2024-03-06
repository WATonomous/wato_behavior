# This file is to edit the MetaDriveEnv to render during step
from TemporalMap import TemporalMap
from metadrive.metadrive.envs import MetaDriveEnv
# from metadrive.metadrive.envs import TopDownMetaDrive

# from metadrive.metadrive.obs.top_down_obs_multi_channel import TopDownMultiChannel
# from metadrive.metadrive.obs.state_obs import StateObservation

from mpc import MPCController

from gymnasium.spaces import Dict, Box

import numpy as np

import math


class BehaviourEnv(MetaDriveEnv):
    # Override init to modify config
    def __init__(self, new_config, window=False):
        config = super(BehaviourEnv, self).default_config()
        config.update(new_config)
        
        # Increase speed reward and driving reward from 0.1 to 0.5
        config["speed_reward"] = 0.5
        config["driving_reward"] = 0.5
        config["traffic_density"] = 0.2
        # config["use_render"] = True
        self.window = window
        # print(config)

        config["norm_pixel"] = False
        super().__init__(config)

        # print(self.observation_space)
        self.prev_action = np.array([0, 0]).reshape(-1, 1)

    # Override action space to output 4 numbers
    @property
    def action_space(self):
        return Box(low=-1.0, high=1.0, shape=(1,3))
    
    def get_single_observation(self):
        return TemporalMap(
            self.config["vehicle_config"]
        )


    # Override environment step
    def step(self, action):
        # Take the action (target position (polar coords), angle and velocity) and give to mpc
        mpc_controller = MPCController()
        target_x, target_y = mpc_controller.set_objective(
            action[0][0] * np.pi/3.0,
            action[0][1] * np.pi/3.0,
            (action[0][2] + 1.0) * 30.0
            )
        mpc = mpc_controller.mpc
        cur_vel = self.agent.velocity
        # Current state [x_pos, y_pos, theta, vel]
        # MPC is run relative to each time step which is why the pos values are 0
        x0 = np.array([0, 0, 0, math.sqrt(cur_vel[0]**2 + cur_vel[1]**2)]).reshape(-1,1)
        # Use previous action as current action
        u0 = self.prev_action
        mpc.x0 = x0
        mpc.u0 = u0
        mpc.set_initial_guess()

        init_x = self.agent.position[0]
        init_y = self.agent.position[1]

        # Compute actual control values
        u0 = mpc.make_step(x0)

        control_action = np.array([u0[1][0], u0[0][0]])

        MAX_STEPS = 10
        i = 0
        while x0[0][0] < (target_x/2.0) and i < MAX_STEPS:
            # print(self.agent.velocity[0]**2 + self.agent.velocity[1]**2)
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



