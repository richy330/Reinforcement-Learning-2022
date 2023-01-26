#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 23:28:47 2023

@author: richard
"""

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
import argparse
import copy
import os
import random
import time
from distutils.util import strtobool
from collections import deque


import gym
from gym.spaces import Box
from gym.wrappers import FrameStack
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
# from stable_baselines3.common.atari_wrappers import (
#     ClipRewardEnv,
#     EpisodicLifeEnv,
#     FireResetEnv,
#     MaxAndSkipEnv,
#     NoopResetEnv,
# )
# from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter



obs_size = 84
env_rendering = False
save_network_frequency = 50000
def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument("--upload-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to upload the saved model to huggingface")
    parser.add_argument("--hf-entity", type=str, default="",
        help="the user or org name of the model repository from the Hugging Face Hub")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=100_000_000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=10_000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.,
        help="the target network update rate")
    parser.add_argument("--target-network-frequency", type=int, default=1000,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=32,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.01,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.10,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=4,
        help="the frequency of training")
    args = parser.parse_args()
    # fmt: on
    return args


# def make_env(env_id, seed, idx, capture_video, run_name):
#     def thunk():
#         env = gym.make(env_id)
#         env = gym.wrappers.RecordEpisodeStatistics(env)
#         if capture_video:
#             if idx == 0:
#                 env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
#         env = NoopResetEnv(env, noop_max=30)
#         env = MaxAndSkipEnv(env, skip=4)
#         env = EpisodicLifeEnv(env)
#         if "FIRE" in env.unwrapped.get_action_meanings():
#             env = FireResetEnv(env)
#         env = ClipRewardEnv(env)
#         env = gym.wrappers.ResizeObservation(env, (84, 84))
#         env = gym.wrappers.GrayScaleObservation(env)
#         env = gym.wrappers.FrameStack(env, 4)
#         env.seed(seed)
#         env.action_space.seed(seed)
#         env.observation_space.seed(seed)
#         return env

#     return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        return self.network(x / 255.0)



class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info



class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        transform = torchvision.transforms.Grayscale()
        observation = transform(observation)
        return observation



class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = (shape, shape) if isinstance(shape, int) else tuple(shape)
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(self.shape),
                                                     torchvision.transforms.Normalize(0, 255)])
        return transforms(observation).squeeze(0)


class ExperienceReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def add(self, state, next_state, action, reward, done):
        state = state.__array__()
        next_state = next_state.__array__()
        self.memory.append((state, next_state, action, reward, done))

    def sample(self, batch_size):
        # DONE: uniformly sample batches of Tensors for: state, next_state, action, reward, done
        # ...
        # samples = random.sample(self.memory, batch_size)
        # state = torch.tensor(np.array([s[0] for s in samples]), dtype=torch.float32)
        # next_state = torch.tensor(np.array([s[1] for s in samples]), dtype=torch.float32)
        # action = torch.tensor(np.array([s[2] for s in samples]), dtype=torch.float32)
        # reward = torch.tensor(np.array([s[3] for s in samples]), dtype=torch.float32)
        # done = torch.tensor(np.array([s[4] for s in samples]), dtype=torch.float32)
        
        
        
        samples = random.sample(self.memory, batch_size)
        state = torch.tensor(np.array([s[0] for s in samples]), dtype=torch.float32)
        next_state = torch.tensor(np.array([s[1] for s in samples]), dtype=torch.float32)

        action_np = np.fromiter((s[2] for s in samples), float)
        action = torch.tensor(action_np, dtype=torch.float32)
        reward = torch.tensor(np.array([s[3] for s in samples]), dtype=torch.float32)
        done = torch.tensor(np.array([s[4] for s in samples]), dtype=torch.float32)
        return state, next_state, action, reward, done


def lazyframe_to_tensor(x):
    return torch.tensor(x.__array__()).float()


def print_metrics(it: int, metrics: dict, is_training: bool, window=100, it_per_hour=0):
    reward_mean = np.mean(metrics['reward'][-window:])
    loss_mean = np.mean(metrics['loss'][-window:])
    mode = "train" if is_training else "test"
    print(f"Episode {it:4d} | {mode:5s} | Episodes/h {it_per_hour:.2f}| reward {reward_mean:5.5f} | loss {loss_mean:5.5f}")


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def save_checkpoint(curr_step: int, eps: float, train_metrics: dict, checkpoint_directory):
    save_dict = {'curr_step': curr_step, 
                 'train_metrics': train_metrics, 
                 'eps': eps,
                 'online_dqn': q_network.state_dict(), 
                 'target_dqn': target_network.state_dict()}
    torch.save(save_dict, checkpoint_directory)
    
    
    
if __name__ == "__main__":
    args = parse_args()
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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    
    # Create and preprocess the Space Invaders environment
    if env_rendering:
        envs = gym.make("ALE/SpaceInvaders-v5", full_action_space=False, render_mode="human")
    else:
        envs = gym.make("ALE/SpaceInvaders-v5", full_action_space=False)

    envs = SkipFrame(envs, skip=4)
    envs = GrayScaleObservation(envs)
    envs = ResizeObservation(envs, shape=84)
    envs = FrameStack(envs, num_stack=4)
    image_stack, h, w = envs.observation_space.shape
    num_actions = envs.action_space.n
    print('Number of stacked frames: ', image_stack)
    print('Resized observation space dimensionality: ', h, w)
    print('Number of available actions by the agent: ', num_actions)
    
    
    #envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    #assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(num_actions).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(num_actions).to(device)
    target_network.load_state_dict(q_network.state_dict())

    # rb = ReplayBuffer(
    #     args.buffer_size,
    #     envs.single_observation_space,
    #     envs.single_action_space,
    #     device,
    #     optimize_memory_usage=True,
    #     handle_timeout_termination=True,
    # )
    rb = ExperienceReplayMemory(args.buffer_size)
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    episode_number = 0
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array(envs.action_space.sample())
        else:
            obs_tensor = lazyframe_to_tensor(obs).unsqueeze(0)
            q_values = q_network(torch.Tensor(obs_tensor).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()[0]
            #print(actions)

        # TRY NOT TO MODIFY: execute the game and log data.
        
        next_obs, rewards, dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        # for info in infos:
        #     if "episode" in info.keys():
        #         print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
        #         writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
        #         writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
        #         writer.add_scalar("charts/epsilon", epsilon, global_step)
        #         break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = copy.deepcopy(next_obs)

        if dones:
            episode_number += 1
            #print(f'episode {episode_number} finished')
            #real_next_obs[idx] = infos[idx]["terminal_observation"]
        rb.add(obs, real_next_obs, actions, rewards, dones)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                state, next_state, action, reward, dones = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(next_state).max(dim=1)
                    td_target = reward.flatten() + args.gamma * target_max * (1 - dones.flatten())
                old_val = q_network(state).gather(1, action.long().unsqueeze(1)).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    print(f"loss: {loss.item()}")
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                print('syncing networks')
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )
            

            if episode_number % save_network_frequency == 0: 
                checkpoint_directory = f'./standard_model_episode{episode_number}.pth.tar'
                save_checkpoint(global_step, epsilon, dict(), checkpoint_directory)


    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.dqn_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=0.05,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
