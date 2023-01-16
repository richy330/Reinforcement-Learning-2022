#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 14:35:26 2023

@author: richard
"""

import time
import sys, os
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn.functional as F
import random
from gym.spaces import Box
from collections import deque
import copy
from gym.wrappers import FrameStack


rng = np.random.default_rng()



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

    def store(self, state, next_state, action, reward, done):
        state = state.__array__()
        next_state = next_state.__array__()
        self.memory.append((state, next_state, action, reward, done))

    def sample(self, batch_size):
        # TODO: uniformly sample batches of Tensors for: state, next_state, action, reward, done
        # ...
        state, next_state, action, reward, done = [], [], [], [], []
        sample_indizes = rng.choice(len(self), size=batch_size, replace=True)
        for index in sample_indizes:
            one_state, one_next_state, one_action, one_reward, one_done = self.memory[index]
            state.append(one_state)
            next_state.append(one_next_state)
            action.append(one_action)
            reward.append(one_reward)
            done.append(one_done)
        return (
            torch.tensor(state), 
            torch.tensor(next_state), 
            torch.tensor(action), 
            torch.tensor(reward), 
            torch.tensor(done))



class DeepQNet(torch.nn.Module):
    def __init__(self, h, w, image_stack, num_actions):
        super(DeepQNet, self).__init__()
        # TODO: create a convolutional neural network
        # taken from torch-demo
        # Rich: find out how to properly use the conctructor-arguments here
        
        # Grayscale image has one channel only, but we send 4 images per sample
        n_input_channes = image_stack
        self.conv1 = torch.nn.Conv2d(in_channels=n_input_channes, out_channels=5, kernel_size=(3, 3))
        self.conv2 = torch.nn.Conv2d(in_channels=5, out_channels=10, kernel_size=(3, 3))
        self.conv3 = torch.nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3))
        self.pool = torch.nn.MaxPool2d(2, 2)
        # fc: Full Connection
        # self.fc1 = torch.nn.Linear(20 * 2 * 2, 40)
        # TODO: find out how to come up with 1280 here
        self.fc1 = torch.nn.Linear(1280, 40)
        self.fc2 = torch.nn.Linear(40, num_actions)

    def forward(self, x):
        # TODO: forward pass from the neural network
        # taken from torch-demo
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)     # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def convert(x):
    return torch.tensor(x.__array__()).float()
   
def policy(state, is_training):
    global eps
    state = convert(state).unsqueeze(0).to(device)
    # TODO: Implement an epsilon-greedy policy
    # Rich: Decide if we use the online- or target network for finding the greedy action
    if is_training and (rng.random() <= eps):
        return env.action_space.sample()
    else:
        return online_dqn(state).argmax()


def compute_loss(state, action, reward, next_state, done):
    state = convert(state).to(device)
    next_state = convert(next_state).to(device)
    action = action.to(device)
    reward = reward.to(device)
    done = done.to(device)
    
    # TODO: Compute the DQN (or DDQN) loss based on the criterion
    # Rich: state, action, etc are already torch tensors of sampled batches
    
    Q_online = torch.take(online_dqn(state), action.long())
    Q_target_max, _ = target_dqn(state).max(dim=1)
    Q_target = reward + gamma * Q_target_max * (1 - done.long())
    return criterion(Q_online, Q_target)


def run_episode(curr_step: int, buffer: ExperienceReplayMemory, is_training: bool):
    global eps
    global target_dqn
    episode_reward, episode_loss = 0, 0.
    state = env.reset()
    
    # Rich: i think this is the max episode length
    for t in range(max_train_frames):
        action = policy(state, is_training)
        curr_step += 1
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward

        if is_training:
            buffer.store(state, next_state, action, reward, done)
            
            
            if curr_step == burn_in_phase:
                print('Burn in phase finished')
            if curr_step > burn_in_phase:
                state_batch, next_state_batch, action_batch, reward_batch, done_batch = buffer.sample(batch_size)

                if curr_step % sync_target == 0:
                    # TODO: Periodically update your target_dqn at each sync_target frames
                    # ...
                    print(f'Syncing Networks (current step: {curr_step})')
                    target_dqn.load_state_dict(online_dqn.state_dict())

                    
                loss = compute_loss(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                episode_loss += loss.item()
        else:
            with torch.no_grad():
                episode_loss += compute_loss(state, action, reward, next_state, done).item()

        state = next_state

        if done:
            break

    return dict(reward=episode_reward, loss=episode_loss / t), curr_step
    
def update_metrics(metrics: dict, episode: dict):
    for k, v in episode.items():
        metrics[k].append(v)


def print_metrics(it: int, metrics: dict, is_training: bool, window=100, it_per_hour=None):
    reward_mean = np.mean(metrics['reward'][-window:])
    loss_mean = np.mean(metrics['loss'][-window:])
    mode = "train" if is_training else "test"
    print(f"Episode {it:4d} | {mode:5s} | Episodes/h {it_per_hour:.2f}| reward {reward_mean:5.5f} | loss {loss_mean:5.5f}")


def save_checkpoint(curr_step: int, eps: float, train_metrics: dict):
    save_dict = {'curr_step': curr_step, 
                 'train_metrics': train_metrics, 
                 'eps': eps,
                 'online_dqn': online_dqn.state_dict(), 
                 'target_dqn': target_dqn.state_dict()}
    torch.save(save_dict, './your_saved_model.pth.tar')

    

env_rendering = False    # Set to False while training your model on Colab
testing_mode = False
test_model_directory = './your_saved_model.pth.tar'

# Create and preprocess the Space Invaders environment
if env_rendering:
    env = gym.make("ALE/SpaceInvaders-v5", full_action_space=False, render_mode="human")
else:
    env = gym.make("ALE/SpaceInvaders-v5", full_action_space=False)
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)
image_stack, h, w = env.observation_space.shape
num_actions = env.action_space.n
print('Number of stacked frames: ', image_stack)
print('Resized observation space dimensionality: ', h, w)
print('Number of available actions by the agent: ', num_actions)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

seed = 61
env.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Hyperparameters (to be modified)
batch_size = 32
alpha = 0.00025
gamma = 0.95
eps, eps_decay, min_eps = 1.0, 0.999, 0.05
buffer = ExperienceReplayMemory(5_000)
burn_in_phase = 20_000
sync_target = 30_000
max_train_frames = 10_000
max_train_episodes = 100_000
max_test_episodes = 1
curr_step = 0
print_metric_period = 1
save_network_period = 50


online_dqn = DeepQNet(h, w, image_stack, num_actions)
target_dqn = copy.deepcopy(online_dqn)
online_dqn.to(device)
target_dqn.to(device)
for param in online_dqn.parameters():
    param.requires_grad = False

# TODO: create the appropriate MSE criterion and Adam optimizer
# Rich: decide if online or target parameters to be passed to Adam
optimizer = torch.optim.Adam(target_dqn.parameters())
criterion = torch.nn.MSELoss()


if testing_mode:
    # TODO: Load your saved online_dqn model for evaluation
    # ...
    test_metrics = dict(reward=[], loss=[])
    for it in range(max_test_episodes):
        episode_metrics, curr_step = run_episode(curr_step, buffer, is_training=False)
        update_metrics(test_metrics, episode_metrics)
        print_metrics(it + 1, test_metrics, is_training=False)
else:
    train_metrics = dict(reward=[], loss=[])
    t0 = time.time()
    for it in range(max_train_episodes):
        episode_metrics, curr_step = run_episode(curr_step, buffer, is_training=True)
        update_metrics(train_metrics, episode_metrics)
        t1 = time.time()
        it_per_hour = 3600/(t1 - t0)
        if curr_step > burn_in_phase and eps > min_eps:
            eps *= eps_decay
        if it % print_metric_period == 0:
            print_metrics(it, train_metrics, is_training=True, it_per_hour=it_per_hour)
        if it % save_network_period == 0:
            save_checkpoint(curr_step, eps, train_metrics)
        t0 = time.time()