from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt

# plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 10.0
plt.rcParams['ytick.labelsize'] = 'small'
plt.rcParams['xtick.labelsize'] = 'small'
plt.rcParams['figure.figsize'] = (8, 4)
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.grid.axis'] = 'both'
plt.rcParams['axes.grid.which'] = 'both'
plt.rcParams['grid.alpha'] = 1.0
plt.rcParams['grid.color'] = '#b0b0b0'
plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['grid.linewidth'] = 0.6


def plot_result(score, epsilon, title, info, filename, mean_window=50):
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


def plot_results(scores, epsilon, title, info, filename, mean_window=100, cmap='PuBu'):
    fig, (ax0, ax1) = plt.subplots(1, 2)
    fig.suptitle(title)
    colormap = plt.get_cmap(cmap)
    colorcycler = cycler(color=[colormap(k) for k in np.linspace(0.4, 1, len(scores))])

    ax0.set(xlabel='Episode', ylabel='Score')
    ax0.set_prop_cycle(colorcycler)
    for score in scores:
        ax0.plot(score, alpha=0.7, lw=1)

    for score in scores:
        running_average = np.convolve(score, np.ones(mean_window)/mean_window, mode='valid')
        ax0.plot(running_average, '#e24848', alpha=0.9, lw=1)

    ax1.plot(epsilon, color='tab:red')
    ax1.set(xlabel='Episode', ylabel='Epsilon', yscale='log')
    ax1.text(0.9, 0.9, info, va='top', ha='right', transform=ax1.transAxes)

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def plot_result_frames(scores, epsilon, title, info, filename,
                       mean_window=500, cmap='PuBu', lr=None, master_score=None, txt=''):
    fig, (ax0, ax1) = plt.subplots(1, 2)
    fig.suptitle(title)
    colormap = plt.get_cmap(cmap)
    colorcycler = cycler(color=[colormap(k) for k in np.linspace(0.4, 1, len(scores))])

    ax0.set(xlabel='Frame', ylabel='Score')
    ax0.set_prop_cycle(colorcycler)
    for score in scores:
        ax0.plot(score, alpha=0.7, lw=1)
    # for score in scores:
    if mean_window is not None:
        running_average = np.convolve(np.mean(scores, axis=0), np.ones(mean_window) / mean_window, mode='valid')
        ax0.plot(running_average, '#e24848', label=f'average, {mean_window} frames', alpha=0.9, lw=1)
    if master_score is not None:
        ax0.plot(master_score, color='tab:green', label='master', lw=1)
    ax0.legend(loc='lower right')

    ax1.set(xlabel='Frame')
    if epsilon is not None:
        ax1.plot(epsilon, color='tab:red', label='epsilon')
    if lr is not None:
        ax1.plot(lr, color='tab:blue', label='learning rate')
    ax1.text(0.9, 0.7, info, va='top', ha='right', transform=ax1.transAxes)
    ax1.legend(loc='upper right')

    fig.text(1.0, 0.0, txt, color="grey", fontsize=8, va='bottom', ha='right')
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
