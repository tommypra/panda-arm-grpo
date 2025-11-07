# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
import logging
import copy
from dataclasses import dataclass

import gymnasium as gym
import panda_mujoco_gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    # exp_name: str = os.path.basename(__file__)[: -len(".py")]
    exp_name: str = f"xaGRPO_42"
    """the name of this experiment"""
    seed: int = 42
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "totokRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""
    enable_logging: bool = False
    """flag to enable/disable detailed logging"""

    # Algorithm specific arguments
    env_id: str = "FrankaReachDense-v0"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 2e-5
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    macro_steps: int = 4
    """the number of steps for sampling the environment"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    penalty_coef: float = 10.0
    """the penalty coefficient of KL Divergence"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, gamma, render_mode="rgb_array"):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode=render_mode, max_episode_steps=100)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, render_mode=render_mode, max_episode_steps=100)
        env = gym.wrappers.FilterObservation(env, filter_keys=['observation', 'desired_goal'])
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
    
class Agent_GRPO(nn.Module):
    def __init__(self, envs):
        super().__init__()
        
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))
        
    def get_action(self, x):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)
    
    def get_actions(self, x):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        action_samples = probs.sample()
        return action_samples, probs.log_prob(action_samples).sum(1)
    
    def get_logprob(self, x, action):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
    
        probs = Normal(action_mean, action_std)
        return probs.log_prob(action).sum(2)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    # Logging
    logger = logging.getLogger("continuous_PPO")
    logger.setLevel(logging.DEBUG if args.enable_logging else logging.WARNING)
    
    file_handler = logging.FileHandler("log/training_log.txt")
    file_handler.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    logger.info("=======================")
    logger.info("Traning begin!")
    logger.info("=======================")

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.AsyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.macro_steps)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    
    agent = Agent_GRPO(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.macro_steps) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.macro_steps) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.macro_steps)).to(device)
    rewards = torch.zeros((args.num_steps, args.macro_steps)).to(device)
    advantages = torch.zeros_like(rewards).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / (1.25 * args.num_iterations)
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
            
        best_env_id = 0
        all_length = []
        all_reward = []
        all_success = []
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = torch.tile(next_obs[best_env_id], (args.macro_steps, 1))

            # Sample T actions with macro steps according to the current policy 
            with torch.no_grad():
                actions_g, logprobs_g = agent.get_actions(obs[step])
                
            states = envs.call('get_state')
            current_states = states[best_env_id]
            for env_id in range(envs.num_envs):
                envs.call('set_state', current_states)
            
            if (iteration % 10 == 0):
                logger.info(f"====== Step: {step}")
                logger.info(f"== Current obs: {next_obs}")
            
            next_obs, reward, terminations, truncations, infos = envs.step(actions_g.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            mean_reward_group, std_reward_group = reward.mean(0), reward.std(0)
            
            actions[step] = actions_g
            logprobs[step] = logprobs_g
            rewards[step] = torch.Tensor(reward).to(device)
            advantages[step] = (rewards[step] - mean_reward_group) / (std_reward_group + 1e-8) / args.macro_steps

            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            
            if (iteration % 10 == 0):
                logger.info(f"== Next obs: {next_obs}")
                logger.info(f"== Reward: {reward}")
                logger.info(f"== Termination: {terminations} or Truncation: {truncations}")
                logger.info(f"== Info: {infos}")
            
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
            elif "episode" in infos:
                reward_info = infos['episode']['r'][best_env_id]
                length_info = infos['episode']['l'][best_env_id]
                success_info = 1 if length_info < 100 else 0
                all_reward.append(reward_info)
                all_length.append(length_info)
                all_success.append(success_info)

                if reward_info != 0:
                    print(f"global_step={global_step}, episodic_return={reward_info.item()}")
                    writer.add_scalar(f"charts/episodic_return", reward_info.item(), global_step)
                    writer.add_scalar(f"charts/episodic_length", length_info.item(), global_step)
                
        # Calculate advantages
        writer.add_scalar("rollout/ep_rew_mean", np.mean(all_reward), global_step)
        writer.add_scalar("rollout/ep_len_mean", np.mean(all_length), global_step)
        writer.add_scalar("rollout/success_rate", np.mean(all_success), global_step)
        
        b_obs = obs
        b_actions = actions
        b_logprobs = logprobs
        b_advantages = advantages
        
        clipfracs = []
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                
                newlogprob = agent.get_logprob(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                
                ratio = logratio.exp()
                mb_advantages = b_advantages[mb_inds]
                
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean(dim=0).mean()
                
                loss = pg_loss + args.penalty_coef * approx_kl
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                
        if iteration % 50 == 0:
            model_path = f"runs/{run_name}/{args.exp_name}_grpo_sync_{iteration}.pth"
            torch.save(agent.state_dict(), model_path)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
    
    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}_grpo_sync.pth"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()