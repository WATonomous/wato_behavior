# Import our redefined environment
from BehaviourEnv import BehaviourEnv

# Import the RL algorithms we will use
from stable_baselines3 import SAC

# For vectorizing the environment
from stable_baselines3.common.vec_env import SubprocVecEnv
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
            start_seed=rank,
            ), window=False) 
        env.reset(seed=rank)
        return env

    return _init

# For getting number of cpus
from multiprocessing import cpu_count

# Import PyTorch for custom policy networks
import torch

# Training Function
def train(learning_rate, timesteps, policy='MlpPolicy', policy_kwargs=None, model_name=None):
    if policy_kwargs == None:
        raise ValueError("Policy Network Architecture Not Specified")

    # Create the parallelized vectorized environment
    # Number of process CANNOT exceed number of cpus
    env = SubprocVecEnv([make_env(rank=i, num_env=NUM_CPU) for i in range(NUM_CPU)]) 
    
    # Create Model 
    if model_name == None:
        model = SAC(policy=policy, env=env, learning_rate=learning_rate, 
                    policy_kwargs=policy_kwargs, 
                    buffer_size=1000)
                    # tensorboard_log='./behaviour')
    else:
        model = SAC.load(model_name)


    model.learn(total_timesteps=timesteps)
    
    env.close()
    return model

def test(model_name, timesteps=10000):
    model = SAC.load(model_name)
    env = BehaviourEnv(dict(
        traffic_mode="trigger",
        num_scenarios=900,
        start_seed=600
        ), window=True) 
    obs, info = env.reset()
    for i in range(timesteps):
        action, _ = model.predict(observation=obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
             obs, info = env.reset()
    env.close()

def getSavedModel(model_name):
    return SAC.load(model_name)


# Define constants
# NUM_CPU defines the number of concurrent instances to run when training 
# (should not exceed the number of cpus)
NUM_CPU = cpu_count() - 2
# NUM_CPU=1
LEARNING_RATE=0.0003
TRAINING_TIMESTEPS=50000
MODEL_NAME='sac_model'

if __name__ == '__main__':
    # Currently the actor and critic networks are 3 32 neuron layers
    policy_network_arch = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(
            pi=[32, 32, 32],
            qf=[32, 32, 32]
        ),
        normalize_images=False
    )
    model = train(
        policy='MlpPolicy',
        learning_rate=LEARNING_RATE,
        timesteps=TRAINING_TIMESTEPS, 
        policy_kwargs=policy_network_arch)
    model.save(MODEL_NAME)

    # test(model_name=MODEL_NAME)