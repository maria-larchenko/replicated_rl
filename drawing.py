import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['figure.figsize'] = (8, 4)
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.grid.axis'] = 'both'
plt.rcParams['axes.grid.which'] = 'both'
plt.rcParams['grid.alpha'] = 1.0
plt.rcParams['grid.color'] = '#b0b0b0'
plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['grid.linewidth'] = 0.6


def plot_results(score, epsilon, title, info, filename, mean_window=100):
    running_average = np.convolve(score, np.ones(mean_window)/mean_window, mode='valid')

    fig, (ax0, ax1) = plt.subplots(1, 2)
    fig.suptitle(title)

    ax0.plot(score)
    ax0.plot(running_average)
    ax0.set(xlabel='Episode', ylabel='Score')

    ax1.plot(epsilon, color='r')
    ax1.set(xlabel='Episode', ylabel='Epsilon', yscale='log')
    ax1.text(0.9, 0.9, info, va='top', ha='right', transform=ax1.transAxes)

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
