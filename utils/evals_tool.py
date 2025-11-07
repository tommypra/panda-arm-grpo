from typing import Callable

import torch
import gymnasium as gym
import numpy as np
import random
import time


def run_simulation(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cuda"),
    capture_video: bool = True,
    gamma: float = 0.99,
    seed: int = 1,
    render_mode:str = 'rgb_array'
) -> np.float32:    
    
    
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, capture_video, run_name, render_mode=render_mode, gamma=gamma)])
    agent = Model(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()
    
    obs, _ = envs.reset()
    episodic_return = []
    for ep in range(eval_episodes):
        ep_reward = 0
        step = 0
        obs, _ = envs.reset()
        while True:
            step += 1
            actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
            obs, reward, termination, truncation, info = envs.step(actions.cpu().numpy())
            ep_reward += reward
            time.sleep(0.2)
            if termination or truncation:
                break
        print(f"Episode {ep+1}, Reward: {ep_reward}, Steps: {step}")
        episodic_return.append(ep_reward)
    envs.close()
    print(f"Average episodic return: {np.mean(episodic_return)}")
    
    return episodic_return