import matplotlib.pyplot as plt
import numpy as np

def simple_moving_average(data, window_size = 100):
    weights = np.ones(window_size) / window_size
    sma = np.convolve(data, weights, mode='valid')
    sma = np.concatenate((np.full((window_size)//2, np.nan),sma,np.full(len(data)-len(sma)-(window_size)//2, np.nan)))  # Pad the beginning with NaN for alignment
    return sma


def cum_avg(data):
    return np.cumsum(data) / np.arange(1, len(data) + 1)

def plot_returns(returns, title = ""):
    x = range(len(returns))
    plt.plot(x, returns, label="Episode Returns")
    plt.plot(x, simple_moving_average(returns, 100), label="Moving Average (100)")
    plt.plot(x, cum_avg(returns), label="Cumulative Average Return")

    plt.xlabel("Episode")
    plt.ylabel("Episode Return")
    plt.title(title)
    plt.legend()
    plt.show()
