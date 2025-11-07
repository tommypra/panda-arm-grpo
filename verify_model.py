from typing import Callable

import gymnasium as gym
import torch
import numpy as np
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
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, capture_video, run_name, gamma, render_mode='human')])
    agent = Model(envs).to(device)
    old_state_dict = agent.state_dict()
    loaded_state_dict = torch.load(model_path)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()
    obs, _ = envs.reset()
    
    while True:
        # actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
        actions, _, _, = agent.get_action(torch.Tensor(obs).to(device))
        next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
        obs = next_obs
        print(infos)
        time.sleep(0.1)


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = False,
    gamma: float = 0.99,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, capture_video, run_name, gamma)])
    print(envs.single_action_space)
    agent = Model(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    agent.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        actions, _, _, _ = agent.get_action(torch.Tensor(obs).to(device))
        next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        elif "episode" in infos:
            print(f"eval_episode={len(episodic_returns)}, episodic_return={infos['episode']['r']}")
            episodic_returns += [infos["episode"]["r"]]
        obs = next_obs

    return episodic_returns


if __name__ == "__main__":
    from huggingface_hub import hf_hub_download
    
    # from ppo_continuous import Agent, make_env
    from grpo_synchronous import Agent_GRPO, make_env
    import random
    
    model_path = 'runs/FrankaReachDense-v0__xaGRPO_42__42__1742334749/xaGRPO_42_grpo_sync.pth'
    
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    run_simulation(
        model_path,
        make_env,
        "FrankaReachDense-v0",
        eval_episodes=10,
        run_name=f"eval",
        Model=Agent_GRPO,
        # Model=Agent,
        device="cpu",
        capture_video=False,
    )