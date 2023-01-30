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

plt.close('all')


#%% adjusting plot appearance
figsize = [12, 8]
fontsize = 18
dpi = 200


color1, color2 = 'red', 'blue'
alphas = [1, 0.8, 0.6, 0.4]



max_episode_number_plot = 8001
smoothing_window_len = 200




# legend_names = ['1-step', '3-step', '6-step']
# title = 'Reward curves for different n-step reward methods under DDQN'
# plot_filename = './plots/n_step_methods_with_loss.png'
# checkpoint_directories = [
#     './finished_checkpoints/ddqn_model_eps_init0.8_episode8000_alpha0.00025_huber_loss_tau0.2.pth.tar',
#     './finished_checkpoints/3step_reward_ddqn_model_eps_init0.8_episode22500_alpha0.00025_huber_loss_tau0.2.pth.tar',
#     './finished_checkpoints/6step_reward_ddqn_model_eps_init0.8_episode23000_alpha0.00025_huber_loss_tau0.2.pth.tar'
# ]




# legend_names = ['eps=1', 'eps=0.8', 'eps=0.5', 'eps=0.2']
# title = 'Reward curves for different epsilons under DQN'
# plot_filename = './plots/epsilons_std_mse_with_loss.png'
# checkpoint_directories = [
#     './finished_checkpoints/standard_model_eps_init1.0_episode10000_alpha0.00025.pth.tar',
#     './finished_checkpoints/standard_model_eps_init0.8_episode8001_alpha0.00025_mse_loss.pth.tar',
#     './finished_checkpoints/standard_model_eps_init0.5_episode4500_alpha0.00025_mse_loss.pth.tar',
#     './finished_checkpoints/standard_model_eps_init0.2_episode4500_alpha0.00025_mse_loss.pth.tar',
# ]




# legend_names = ['eps=1', 'eps=0.8', 'eps=0.5', 'eps=0.2']
# title = 'Reward curves for different epsilons under DDQN'
# plot_filename = './plots/epsilons_ddqn_mse_with_loss.png'
# checkpoint_directories = [
#     './finished_checkpoints/ddqn_model_eps_init1.0_episode8000_alpha0.00025.pth.tar',
#     './finished_checkpoints/ddqn_model_eps_init0.8_episode8000_alpha0.00025_mse_loss.pth.tar',
#     './finished_checkpoints/ddqn_model_eps_init0.5_episode8000_alpha0.00025_mse_loss.pth.tar',
#     './finished_checkpoints/ddqn_model_eps_init0.2_episode8000_alpha0.00025_mse_loss.pth.tar',
# ]


# legend_names = ['eps=1', 'eps=0.8', 'eps=0.5', 'eps=0.2']
# title = 'Reward curves for different epsilons under DDQN, Huber Loss'
# plot_filename = './plots/epsilons_ddqn_huber.png'
# checkpoint_directories = [
#     './finished_checkpoints/ddqn_model_eps_init1.0_episode4200_alpha0.00025_huber_loss.pth.tar',
#     './finished_checkpoints/ddqn_model_eps_init0.8_episode8000_alpha0.00025_huber_loss.pth.tar',
#     './finished_checkpoints/ddqn_model_eps_init0.5_episode8000_alpha0.00025_huber_loss_tau0.2.pth.tar',
#     './finished_checkpoints/ddqn_model_eps_init0.2_episode4200_alpha0.00025_huber_loss.pth.tar',
# ]


# legend_names = ['DQN-MSE', 'DDQN-MSE', 'DDQN-Huber']
# title = 'Reward curves under different DQN and DDQN configurations, eps=1'
# plot_filename = './plots/dqn_ddqn_huber_huber_mse_with_loss.png'
# checkpoint_directories = [
#     './finished_checkpoints/standard_model_eps_init1.0_episode10000_alpha0.00025.pth.tar',
#     './finished_checkpoints/ddqn_model_eps_init1.0_episode8000_alpha0.00025.pth.tar',
#     './finished_checkpoints/ddqn_model_eps_init1.0_episode4200_alpha0.00025_huber_loss.pth.tar',

# ]

# legend_names = ['Standard DDQN', 'Prioritized Sampling DDQN']
# title = 'Reward curves under standard DDQN and prioritized sampling DDQN, eps=1'
# plot_filename = './plots/prio_sampling.png'

# alphas = [0.4, 1, 0.3, 0.1]
# checkpoint_directories = [
#     './finished_checkpoints/ddqn_model_eps_init1_episode8000_alpha0.00025_huber_loss.pth.tar',
#     './finished_checkpoints/ddqn_model_eps_init1.0_episode8000_alpha0.00025_huber_loss_tau0.2_weighted_sampling.pth.tar'

# ]



legend_names = ['Standard Network', 'Complex Network']
title = 'Reward curves for Network structures'
plot_filename = './plots/network_cmp_with_loss.png'
alphas = [0.4, 1, 0.3, 0.1]
checkpoint_directories = [
    './finished_checkpoints/ddqn_model_eps_init0.8_episode8000_alpha0.00025_huber_loss.pth.tar',
    './finished_checkpoints/modded_network_ddqn_model_eps_init0.8_episode8000_alpha0.00025_huber_loss_tau0.2.pth.tar',
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
plt.title(title)
ax1.set_xlabel('Epochs')

ax1.tick_params(axis ='y', labelcolor = color1)
ax1.set_ylabel('Reward', color=color1)
ax1.set_ylim((0, None))

ax2.tick_params(axis ='y', labelcolor = color2)
ax2.set_ylabel('Loss', color=color2)
ax2.set_ylim((0, 5))

plt.savefig(plot_filename)


   