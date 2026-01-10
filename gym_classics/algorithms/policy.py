import numpy as np
from gym_classics.algorithms.DP import backup

def random_policy(env):
    return np.random.choice(env.actions(), size=len(env.states()))

def encode_policy(env, policy, type = "text"):
    return [env.encode_action(a, type = type) for a in policy]

# Note: argmax could be randomized
def greedy_policy(env, V, discount=1):
    """
    Calculate the greedy policy for a given value function.
    
    :param env: the environment
    :param V: the value function
    :param discount: discount factor
    """
    policy = np.zeros(len(env.states()))

    for s in env.states():
        Q_values = [backup(env, discount, V, s, a) for a in env.actions()]
        policy[s] = np.argmax(Q_values)

    return policy

