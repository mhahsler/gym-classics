import matplotlib.pyplot as plt
import numpy as np

def simple_moving_average(data, window_size = 100):
    weights = np.ones(window_size) / window_size
    sma = np.convolve(data, weights, mode='valid')
    sma = np.concatenate((np.full((window_size)//2, np.nan),sma,np.full(len(data)-len(sma)-(window_size)//2, np.nan)))  # Pad the beginning with NaN for alignment
    return sma


def cum_avg(data):
    return np.cumsum(data) / np.arange(1, len(data) + 1)

def plot_returns(returns, y_label = "Episode Return", title = "", window_size = 100):
    """Plot episode returns with a moving average and cumulative average."""
    x = range(len(returns))
    plt.plot(x, returns, label="Episode")
    plt.plot(x, simple_moving_average(returns, window_size), label="Moving Average (100)")
    plt.plot(x, cum_avg(returns), label="Cumulative Average")

    plt.xlabel("Episode")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()
    
def plot_ep_lens(ep_lens, y_label = "Episode Length", title = "", window_size = 100):
    """Plot episode lengths with a moving average."""
    x = range(len(ep_lens))
    plt.plot(x, ep_lens, label="Episode Length")
    plt.plot(x, simple_moving_average(ep_lens, window_size), label="Moving Average (100)")

    plt.xlabel("Episode")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()
