import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np


#%% adjusting plot appearance
plt.close('all')
figsize = (12, 12 / 1.618)
fontsize = 10
dpi = 200




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

def plot_results(range1, range2, range1_label, range2_label, train_results, test_results, suptitle='Deterministic SARSA'):
  fig, axes = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, sharex='all', sharey='all',
                           figsize=figsize)
  X, Y = np.meshgrid(range1, range2, indexing='ij')
  for ax, Z, title in zip(axes, [train_results, test_results], ['training phase', 'test phase']):
    ax.plot_surface(X, Y, Z.mean(-1), cmap='coolwarm')
    ax.set_title(title)
    ax.set_xlabel(range1_label)
    ax.set_ylabel(range2_label)
    ax.set_zlabel('reward')

  plt.suptitle(suptitle)
  plt.show()
