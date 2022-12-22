import numpy as np


def env_step(env, action, render=False):
  step = env.step(action)
  if render:
    env.render()
  return step


def env_reset(env, render=False):
  reset = env.reset()
  if render:
    env.render()
  return reset


def update(reward_list, loss_list, reward, loss):
  reward_list.append(reward)
  loss_list.append(loss)


def update_metrics(metrics, episode):
  for k, v in episode.items():
    metrics[k].append(v)


def print_metrics(it, metrics, training, window=100):
  reward_mean = np.mean(metrics['reward'][-window:])
  loss_mean = np.mean(metrics['loss'][-window:])
  mode = "train" if training else "test"
  print(f"It {it:4d} | {mode:5s} | reward {reward_mean:5.1f} | loss {loss_mean:5.2f}")
