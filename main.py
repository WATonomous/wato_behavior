# Import our redefined environment
from BehaviourEnv import BehaviourEnv

# Import the RL algorithms we will use
from stable_baselines3 import SAC

# For vectorizing the environment
from stable_baselines3.common.vec_env import SubprocVecEnv

from MoEEnv import MoEEnv

def make_env(rank, num_env):
    """
    Utility function for multiprocessed env.

    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init():
        env = BehaviourEnv(dict(
            traffic_mode="trigger",
            num_scenarios=num_env,
            start_seed=rank
            ), window=True) 
        env.reset(seed=rank)
        return env

    return _init

# For getting number of cpus
from multiprocessing import cpu_count

# Import torch to define policy arch
import torch

# Training Function
def train(learning_rate, timesteps, policy='MlpPolicy', explore_exploit_coeff='auto'):
    # Create the parallelized vectorized environment
    # Number of process CANNOT exceed number of cpus
    env = SubprocVecEnv([make_env(rank=i, num_env=NUM_CPU) for i in range(NUM_CPU)])

    # define custom network architecture
    policy_network_arch = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(
            pi=[128, 64, 32],
            qf=[128, 64, 32]
        ),
        normalize_images=False
    ) 

    # Create Model 
    model = SAC(
        policy=policy, 
        env=env, 
        learning_rate=learning_rate, 
        policy_kwargs=policy_network_arch, 
        tensorboard_log='./behaviour_board', 
        buffer_size=10000,
        ent_coef=explore_exploit_coeff,
        target_entropy=0.4
        )
    
    # Load existing model
    # model = SAC.load(MODEL_NAME)
    # model.set_env(env)
    # model.target_entropy = 0.2

    model.learn(total_timesteps=timesteps)
    
    env.close()
    return model

# Function for testing saved model
def test(model_name, timesteps=10000):
    model = SAC.load(model_name)
    env = BehaviourEnv(dict(
        traffic_mode="trigger",
        num_scenarios=100
        ), window=True) 
    obs, info = env.reset()
    for i in range(timesteps):
        action, _ = model.predict(observation=obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
             obs, info = env.reset()
        # draw_multi_channels_top_down_observation(np.transpose(obs))
    # env.top_down_renderer.generate_gif()
    env.close()

from PIL import Image
from metadrive.metadrive.examples.top_down_metadrive import draw_multi_channels_top_down_observation

import numpy as np
import random

# Test the environment by just having the car go straight
def test_env():
    env = MoEEnv(dict(
        traffic_mode="respawn",
        num_scenarios=100
        ), window=True) 
    obs, info = env.reset()
    for i in range(10000):
        action = [[0, 0, -0.6]]
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
        # draw_multi_channels_top_down_observation(obs)
    # env.top_down_renderer.generate_gif()
    env.close()

# Define constants
NUM_CPU = cpu_count() - 3
LEARNING_RATE=0.0002
TRAINING_TIMESTEPS=100000
MODEL_NAME='sac_model'


if __name__ == '__main__':
    # model = train(
    #     policy='CnnPolicy', 
    #     learning_rate=LEARNING_RATE, 
    #     timesteps=TRAINING_TIMESTEPS)
    # model.save(MODEL_NAME)
    # print("done training")
    # test(model_name=MODEL_NAME)
    test_env()


