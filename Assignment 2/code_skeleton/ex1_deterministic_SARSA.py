import matplotlib.pyplot as plt
import numpy as np
from gym.envs.toy_text import CliffWalkingEnv
from cliffwalking_utils import plot_results
from utils import env_reset, env_step

rng = np.random.default_rng()

debugging = False


class DeterministicSARSA:
  def __init__(self, alpha, eps, gamma, alpha_decay, eps_decay, max_train_iterations, max_episode_length):
    self.alpha = alpha
    self.eps = eps
    self.gamma = gamma
    self.alpha_decay = alpha_decay
    self.eps_decay = eps_decay
    self.max_train_iterations = max_train_iterations
    self.max_episode_length = max_episode_length

    self.env = CliffWalkingEnv()
    self.num_actions = self.env.action_space.n
    self.num_observations = self.env.observation_space.n

    self.Q = np.zeros((self.num_observations, self.num_actions))



  def policy(self, state, is_training):
    # DONE: Implement an epsilon-greedy policy
    # - with probability eps return a random action
    # - otherwise find the action that maximizes Q
    if rng.uniform() > self.eps:
        action = np.argmax(self.Q[state])
    else:
        action = rng.integers(0, 3)
    return action



  def train_step(self, state, action, reward, next_state, next_action, done):
    # DONE: Implement the SARSA update.
    # - Q(s, a) = alpha * (reward + gamma * Q(s', a') - Q(s, a))
    # - Make sure that Q(s', a') = 0 if we reach a terminal state
    if done:
        next_Q = 0
    else:
        next_Q = self.Q[next_state, next_action]
    self.Q[state, action] += self.alpha * (reward + self.gamma * next_Q - self.Q[state, action])



  def run_episode(self, training, render=False):
    episode_reward = 0
    state = env_reset(self.env, render)
    action = self.policy(state, training)
    for t in range(self.max_episode_length):

      next_state, reward, done, _ = env_step(self.env, action, render)
      episode_reward += reward
      next_action = self.policy(next_state, training)
      if training:
        self.train_step(state, action, reward, next_state, next_action, done)
      state, action = next_state, next_action
      if done:
        break
    return episode_reward



  def train(self):
    self.train_reward = []
    for it in range(self.max_train_iterations):
      self.train_reward.append(self.run_episode(training=True))
      self.alpha *= self.alpha_decay
      self.eps *= self.eps_decay

  def test(self, render=False):
    self.test_reward = self.run_episode(training=False, render=render)




def tune_alpha_eps(gamma=0.9):
    # Done Create suitable parameter ranges (np.arange)
    alpha_range = np.arange(0, 1.01, step=0.1)
    eps_range = np.arange(0, 1.01, step=0.1)

    
    # DONE: Change `debugging` to `False` after finishing your implementation! Report the results averaged over 5 repetitions!

    if debugging:
      num_repetitions = 1
    else:
      num_repetitions = 5
    
    train_results = np.zeros((len(alpha_range), len(eps_range), num_repetitions))
    test_results = np.zeros((len(alpha_range), len(eps_range), num_repetitions))
    
    for i, alpha in enumerate(alpha_range):
      print(f'alpha = {alpha:0.2f} ', end='')
      for j, eps in enumerate(eps_range):
        for k in range(num_repetitions):
          alg = DeterministicSARSA(alpha=alpha, eps=eps, gamma=gamma, alpha_decay=1., eps_decay=1.,
                                   max_train_iterations=200, max_episode_length=200)
          alg.train()
          alg.test()
          train_results[i, j, k] = np.mean(alg.train_reward)
          test_results[i, j, k] = alg.test_reward
        print('.', end='')
      print()
    
    
    # DONE: Find and print the top-3 parameter combinations, that perform best during the test phase
    
    n_top_results = 3
    avg_test_results = np.mean(test_results, axis=2)
    assert avg_test_results.shape == (len(alpha_range), len(eps_range))
    top_test_results_indices = np.argsort(avg_test_results, axis=None)[-n_top_results:]
    for flat_index in top_test_results_indices:
        top_alpha_indx = flat_index // len(eps_range)
        top_eps_index = int(flat_index % len(eps_range))
        top_test_result = avg_test_results.flatten()[flat_index]
        print(f'top alpha: {alpha_range[top_alpha_indx]}, top epsilon: {eps_range[top_eps_index]}, top result: {top_test_result}')

    
    plot_results(alpha_range, eps_range, 'alpha', 'epsilon', train_results, test_results)
    plt.savefig(r'../plots/det_sarsa_alpha_eps.png')
    
    

def tune_alpha_gamma(eps=0.1):
    # DONE Create suitable parameter ranges (np.arange)
    alpha_range = np.arange(0, 1.01, step=0.1)
    gamma_range = np.arange(0, 1.01, step=0.1)
    
    # DONE: Change `debugging` to `False` after finishing your implementation! Report the results averaged over 5 repetitions!
    if debugging:
      num_repetitions = 1
    else:
      num_repetitions = 5
    
    train_results = np.zeros((len(alpha_range), len(gamma_range), num_repetitions))
    test_results = np.zeros((len(alpha_range), len(gamma_range), num_repetitions))
    
    for i, alpha in enumerate(alpha_range):
      print(f'alpha = {alpha:0.2f} ', end='')
      for j, gamma in enumerate(gamma_range):
        for k in range(num_repetitions):
          alg = DeterministicSARSA(alpha=alpha, eps=eps, gamma=gamma, alpha_decay=1., eps_decay=1.,
                                   max_train_iterations=200, max_episode_length=200)
          alg.train()
          alg.test()
          train_results[i, j, k] = np.mean(alg.train_reward)
          test_results[i, j, k] = alg.test_reward
        print('.', end='')
      print()

  # DONE: Find and print the top-3 parameter combinations, that perform best during the test phase
    n_top_results = 3
    avg_test_results = np.mean(test_results, axis=2)
    assert avg_test_results.shape == (len(alpha_range), len(gamma_range))
    #print(f'avg test result: {avg_test_results}')
    
    top_test_results_indices = np.argsort(avg_test_results, axis=None)[-n_top_results:]
    for flat_index in top_test_results_indices:
        top_alpha_indx = flat_index // len(gamma_range)
        top_gamma_index = int(flat_index % len(gamma_range))
        top_test_result = avg_test_results.flatten()[flat_index]
        print(f'top alpha: {alpha_range[top_alpha_indx]}, top gamma: {gamma_range[top_gamma_index]}, achieved test return: {top_test_result}')


    plot_results(alpha_range, gamma_range, 'alpha', 'gamma', train_results, test_results)
    plt.savefig(r'../plots/det_sarsa_alpha_gamma.png')


if __name__ == '__main__':
  tune_alpha_eps(gamma=0.9)
  tune_alpha_gamma(eps=0.1)
