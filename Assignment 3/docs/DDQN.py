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

# import cProfile, pstats
# from pstats import SortKey

# profiler = cProfile.Profile()
# profiler.enable()

rng = np.random.default_rng()
eps_init = 1.0



# Hyperparameters (to be modified)
batch_size = 32
obs_size = 84
alpha = 0.00025
gamma = 0.95
eps, eps_decay, min_eps = eps_init, 0.999, min(eps_init, 0.05)
experience_replay_size = 10_000
burn_in_phase = 2_000
sync_target = 10_000
max_train_frames = 10_000
max_train_episodes = 2_000
max_test_episodes = 1
curr_step = 0
print_metric_period = 1
save_network_period = 20

env_rendering = True   # Set to False while training your model on Colab
testing_mode = True # if True, also give the checkpoint directory to load!

load_pretrained_model = False    #Set to True to load a state
initial_episode_number = 20   #Number of episode to load
checkpoint_directory = f'./standard_model_eps_init1.0_episode{initial_episode_number}.pth.tar'



run_as_ddqn = False #Decides DDQN(TRUE) or DQN(FALSE) will be used for calculating the loss
# if run_as_ddqn:
#     checkpoint_directory = f'./standard_model_eps_init{eps}_episode{initial_episode_number}_DDQN.pth.tar'
# else:
#     checkpoint_directory = f'./standard_model_eps_init{eps}_episode{initial_episode_number}.pth.tar'

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
        samples = random.sample(self.memory, batch_size)
        state = torch.tensor(np.array([s[0] for s in samples]), dtype=torch.float32)
        next_state = torch.tensor(np.array([s[1] for s in samples]), dtype=torch.float32)
        action = torch.tensor(np.array([s[2] for s in samples]), dtype=torch.float32)
        reward = torch.tensor(np.array([s[3] for s in samples]), dtype=torch.float32)
        done = torch.tensor(np.array([s[4] for s in samples]), dtype=torch.float32)

        # state = torch.empty(size=(batch_size, image_stack, obs_size, obs_size))
        # next_state = torch.empty(size=(batch_size, image_stack, obs_size, obs_size))
        # action = torch.empty(size=(32,))
        # reward = torch.empty(size=(32,))
        # done = torch.empty(size=(32,))
        # sample_indizes = rng.choice(len(self), size=batch_size, replace=True)
        # for i, index in enumerate(sample_indizes):
        #     one_state, one_next_state, one_action, one_reward, one_done = self.memory[index]
        #     state[i, :, :, :] = torch.from_numpy(one_state)
        #     next_state[i, :, :, :] = torch.from_numpy(one_next_state)
        #     action[i] = one_action
        #     reward[i] = one_reward
        #     done[i] = one_done
        return state, next_state, action, reward, done


class DeepQNet(torch.nn.Module):
    def __init__(self, h, w, image_stack, num_actions):
        super(DeepQNet, self).__init__()
        # TODO: create a convolutional neural network
        # taken from torch-demo
        # Rich: find out how to properly use the conctructor-arguments here
        
        # Grayscale image has one channel only, but we send 4 images per sample
        n_input_channes = image_stack
        self.conv1 = torch.nn.Conv2d(in_channels=n_input_channes, out_channels=16, kernel_size=(3, 3))
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3))
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.pool = torch.nn.MaxPool2d(2, 2)
        # fc: Full Connection
        self.fc1 = torch.nn.Linear(4096, 512)
        self.fc2 = torch.nn.Linear(512, num_actions)

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
    
    if run_as_ddqn == True:
        Q_online = torch.take(online_dqn(state), action.long())
        action_Q_online_max, _ = online_dqn(next_state).max(dim=1)
        Q_online_max = torch.take(target_dqn(next_state), action_Q_online_max.long())
        Q_target = reward + gamma * Q_online_max * (1 - done.long()) 
    else:
        Q_online = torch.take(online_dqn(state), action.long())
        Q_target_max, _ = target_dqn(next_state).max(dim=1)
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
                state_tensor = convert(state).unsqueeze(0)
                next_state_tensor = convert(next_state).unsqueeze(0)
                reward = torch.tensor(reward)
                done = torch.tensor(done)
                episode_loss += compute_loss(state_tensor, action, reward, next_state_tensor, done).item()

        state = next_state

        if done:
            break
    return dict(reward=episode_reward, loss=episode_loss / t), curr_step
    

def update_metrics(metrics: dict, episode: dict):
    for k, v in episode.items():
        metrics[k].append(v)


def print_metrics(it: int, metrics: dict, is_training: bool, window=100, it_per_hour=0):
    reward_mean = np.mean(metrics['reward'][-window:])
    loss_mean = np.mean(metrics['loss'][-window:])
    mode = "train" if is_training else "test"
    print(f"Episode {it:4d} | {mode:5s} | Episodes/h {it_per_hour:.2f}| reward {reward_mean:5.5f} | loss {loss_mean:5.5f}")


def save_checkpoint(curr_step: int, eps: float, train_metrics: dict, checkpoint_directory):
    save_dict = {'curr_step': curr_step, 
                 'train_metrics': train_metrics, 
                 'eps': eps,
                 'online_dqn': online_dqn.state_dict(), 
                 'target_dqn': target_dqn.state_dict()}
    torch.save(save_dict, checkpoint_directory)


def load_dqn(checkpoint_directory):
    checkpoint = torch.load(checkpoint_directory)
    online_state_dict = checkpoint['online_dqn']
    target_state_dict = checkpoint['target_dqn']
    curr_step = checkpoint['curr_step']
    train_metrics = checkpoint['train_metrics']
    eps = checkpoint['eps']
    
    online_dqn = DeepQNet(h, w, image_stack, num_actions)
    online_dqn.load_state_dict(online_state_dict)
    target_dqn = DeepQNet(h, w, image_stack, num_actions)
    target_dqn.load_state_dict(target_state_dict)
    
    return online_dqn, target_dqn, curr_step, train_metrics, eps


buffer = ExperienceReplayMemory(experience_replay_size)
# Create and preprocess the Space Invaders environment
if env_rendering:
    env = gym.make("ALE/SpaceInvaders-v5", full_action_space=False, render_mode="human")
else:
    env = gym.make("ALE/SpaceInvaders-v5", full_action_space=False)

env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=obs_size)
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

if load_pretrained_model or testing_mode:
    online_dqn, target_dqn, curr_step, train_metrics, eps = load_dqn(checkpoint_directory)
    burn_in_phase += curr_step
else:
    online_dqn = DeepQNet(h, w, image_stack, num_actions)
    target_dqn = copy.deepcopy(online_dqn)
    initial_episode_number = 0
online_dqn.to(device)
target_dqn.to(device)
        
for param in target_dqn.parameters():
    param.requires_grad = False
    
# TODO: create the appropriate MSE criterion and Adam optimizer

optimizer = torch.optim.Adam(online_dqn.parameters())
criterion = torch.nn.MSELoss()

if testing_mode:
    # TODO: Load your saved online_dqn model for evaluation
    # ...
    curr_step = 0
    test_metrics = dict(reward=[], loss=[])
    for it in range(max_test_episodes):
        episode_metrics, curr_step = run_episode(curr_step, buffer, is_training=False)
        update_metrics(test_metrics, episode_metrics)
        print_metrics(it + 1, test_metrics, is_training=False, window=1)
 
else:
    if load_pretrained_model == False:
        train_metrics = dict(reward=[], loss=[])
    t0 = time.time()
    for it in range(initial_episode_number+1, max_train_episodes):
        episode_metrics, curr_step = run_episode(curr_step, buffer, is_training=True)
        update_metrics(train_metrics, episode_metrics)
        t1 = time.time()
        it_per_hour = 3600/(t1 - t0)
        if curr_step > burn_in_phase and eps > min_eps:
            eps *= eps_decay
        if it % print_metric_period == 0:
            print_metrics(it, train_metrics, is_training=True, it_per_hour=it_per_hour, window=1)
        if it % save_network_period == 0:
            checkpoint_directory = f'./standard_model_eps_init{eps_init}_episode{it}.pth.tar'
            save_checkpoint(curr_step, eps, train_metrics, checkpoint_directory)
        t0 = time.time()     
