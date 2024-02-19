from metadrive.metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.metadrive.obs.top_down_obs_multi_channel import TopDownMultiChannel

import numpy as np
from gymnasium.spaces import Box

import pygame
from metadrive.metadrive.obs.top_down_obs_impl import ObjectGraphics

from metadrive.metadrive.constants import DEFAULT_AGENT
from metadrive.metadrive.obs.top_down_obs_impl import COLOR_BLACK

class TemporalMap(TopDownMultiChannel):
    def __init__(
        self, 
        vehicle_config, 
        onscreen: bool = False, 
        resolution=(128, 128),
        max_distance=20
    ):
        super().__init__(
            vehicle_config, 
            onscreen=onscreen,
            clip_rgb=True,
            resolution=resolution, 
            max_distance=max_distance,
            frame_stack=1
        )
    
    @property
    def observation_space(self):
        return Box(low=0, high=1.0, shape=(1, 128, 128))
    
    def draw_scene(self):
        # Set the active area that can be modify to accelerate
        assert len(self.engine.agents) == 1, "Don't support multi-agent top-down observation yet!"
        vehicle = self.engine.agents[DEFAULT_AGENT]
        pos = self.canvas_runtime.pos2pix(*vehicle.position)
        clip_size = (int(self.obs_window.get_size()[0] * 1.1), int(self.obs_window.get_size()[0] * 1.1))
        self.canvas_runtime.set_clip((pos[0] - clip_size[0] / 2, pos[1] - clip_size[1] / 2, clip_size[0], clip_size[1]))
        self.canvas_runtime.fill(COLOR_BLACK)
        self.canvas_runtime.blit(self.canvas_background, (0, 0))

        # Draw vehicles
        # TODO PZH: I hate computing these in pygame-related code!!!
        ego_heading = vehicle.heading_theta
        ego_heading = ego_heading if abs(ego_heading) > 2 * np.pi / 180 else 0

        # ObjectGraphics.display(
        #     object=vehicle, surface=self.canvas_runtime, heading=ego_heading, color=ObjectGraphics.GREEN
        # )
        for v in self.engine.traffic_manager.vehicles:
            if v is vehicle:
                continue
            h = v.heading_theta
            h = h if abs(h) > 2 * np.pi / 180 else 0
            ObjectGraphics.display(object=v, surface=self.canvas_runtime, heading=h, color=ObjectGraphics.BLUE)
            arrow_length = np.linalg.norm(v.velocity)*3 # You can adjust this value
            arrow_width = 3
            heading = v.heading_theta
            heading = heading if abs(heading) > 2 * np.pi / 180 else 0
            position = [*self.canvas_runtime.pos2pix(v.position[0], v.position[1])]
            w = self.canvas_runtime.pix(v.WIDTH)
            h = self.canvas_runtime.pix(v.LENGTH)
            # As the following rotate code is for left-handed coordinates,
            # so we plus -1 before the heading to adapt it to right-handed coordinates
            angle = -np.rad2deg(heading)

            arrow_start = pygame.math.Vector2(h / 2, 0).rotate(angle) + position  # Front of the box
            arrow_end = arrow_start + pygame.math.Vector2(np.cos(heading), -np.sin(heading)) * arrow_length
            end_line1 = pygame.math.Vector2(0, w / 4).rotate(angle) + arrow_end
            end_line2 = pygame.math.Vector2(0, -w / 4).rotate(angle) + arrow_end

            pygame.draw.lines(self.canvas_runtime, (110, 110, 110), False, [arrow_start, arrow_end], width=arrow_width)
            pygame.draw.lines(self.canvas_runtime, (110, 110, 110), False, [end_line1, end_line2], width=arrow_width)

        # Prepare a runtime canvas for rotation
        return self.obs_window.render(canvas_dict=dict(
                road_network=self.canvas_road_network,
                traffic_flow=self.canvas_runtime,
                target_vehicle=self.canvas_ego,
                # navigation=self.canvas_navigation,
            ), position=pos, heading=vehicle.heading_theta)

    def observe(self, vehicle: BaseVehicle):
    
        obs = super().observe(vehicle)

        # Mirror occupancy grid horizontally (makes more sense)
        obs_new = np.clip(obs[..., 0] - np.clip(obs[..., 2], 0, 0.5019608), 0, 1)
        return np.array([np.transpose(obs_new)])

        


