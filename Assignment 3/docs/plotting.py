#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 16:39:26 2023

@author: richard
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


max_episode_number_plot = 4000
smoothing_window_len = 200

checkpoint_directories = [
    './finished_checkpoints/ddqn_model_eps_init0.8_episode8000_alpha0.00025_huber_loss_tau0.2.pth.tar',
    './finished_checkpoints/3step_reward_ddqn_model_eps_init0.8_episode22500_alpha0.00025_huber_loss_tau0.2.pth.tar',
    './finished_checkpoints/6step_reward_ddqn_model_eps_init0.8_episode23000_alpha0.00025_huber_loss_tau0.2.pth.tar'
]


def smooth(signal, window_len):
    window = np.ones((window_len))/window_len
    smoothed = np.convolve(window, signal, mode='valid')
    return smoothed


def load_data(path):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    train_metrics = checkpoint['train_metrics']
    reward = train_metrics['reward']
    loss = train_metrics['loss']
    
    return (
        smooth(loss[:min(len(loss), max_episode_number_plot)], smoothing_window_len), 
        smooth(reward[:min(len(loss), max_episode_number_plot)], smoothing_window_len))





plt.figure()
for i, path in enumerate(checkpoint_directories):
    loss, reward = load_data(path)
    
    plt.plot(loss, label=f'loss {i}')
    plt.plot(reward, label=f'reward {i}')
    
    
plt.legend()