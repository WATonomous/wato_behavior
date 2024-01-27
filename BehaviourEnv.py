# This file is to edit the MetaDriveEnv to render during step
from metadrive.metadrive.envs import TopDownMetaDrive
from metadrive.metadrive.envs import MetaDriveEnv

class BehaviourEnv(MetaDriveEnv):
    # Override init to modify config
    def __init__(self, config, window=False):
        # Increase speed reward and driving reward from 0.1 to 0.5
        config["speed_reward"] = 1.0
        config["driving_reward"] = 1.0
        self.window = window
        # config["norm_pixel"] = False
        super().__init__(config)

    # Add render to the environment step
    def step(self, action):
        action[1] = max(0.1, action[1])
        ret = super().step(action)
        out = self.render(mode="topdown", 
                scaling=2, 
                camera_position=(100, 0), 
                screen_size=(500, 500),
                screen_record=False,
                window=self.window,
                text={"episode_step": self.engine.episode_step,
                        "mode": "Trigger"})
        return ret

    # Override the reward function
    def reward_function(self, vehicle_id):
        reward, step_info = super().reward_function(vehicle_id)

        # Keep the same reward as vanilla implementation but add scaled time factor
        # if self.config["horizon"] is not None:
        #     reward -= 2.0 * self.episode_lengths[vehicle_id] / self.config["horizon"]
        # step_info["step_reward"] = reward

        return reward, step_info

