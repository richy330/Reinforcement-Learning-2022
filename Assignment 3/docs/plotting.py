#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 16:39:26 2023

@author: richard
"""

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import torch


#%% adjusting plot appearance
figsize = [12, 8]
fontsize = 18
dpi = 200

alphas = [1, 0.6, 0.3, 0.1]
color1, color2 = 'red', 'blue'

max_episode_number_plot = 8001
smoothing_window_len = 200
plot_name = 'test_plot'
legend_names = ['1 step', '3 step', '6 step']

checkpoint_directories = [
    './finished_checkpoints/ddqn_model_eps_init0.8_episode8000_alpha0.00025_huber_loss_tau0.2.pth.tar',
    './finished_checkpoints/3step_reward_ddqn_model_eps_init0.8_episode22500_alpha0.00025_huber_loss_tau0.2.pth.tar',
    './finished_checkpoints/6step_reward_ddqn_model_eps_init0.8_episode23000_alpha0.00025_huber_loss_tau0.2.pth.tar'
]






pylab.rcParams.update({
    'figure.figsize': figsize,
    'legend.fontsize': fontsize,
    'axes.labelsize': fontsize,
    'axes.titlesize': fontsize,
    'xtick.labelsize': fontsize,
    'ytick.labelsize': fontsize,
    'savefig.dpi': dpi,
    "font.family": "serif"
})

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





fig, ax1 = plt.subplots()
ax2 = ax1.twinx()



for i, path in enumerate(checkpoint_directories):
    loss, reward = load_data(path)
    alpha = alphas[i]
    label = legend_names[i]
    
    ax1.plot(reward, label=f'reward {label}', alpha=alpha, color=color1)
    ax2.plot(loss, label=f'loss {label}', alpha=alpha, color=color2)

    
fig.legend()
ax1.set_xlabel('Epochs')

ax1.set_ylabel('Reward', color=color1)
ax2.set_ylabel('Loss', color=color2)

ax1.tick_params(axis ='y', labelcolor = color1)
ax2.tick_params(axis ='y', labelcolor = color2)

plt.savefig(f'./{plot_name}.png')


   