import numpy as np
import random
from gym_classics.envs.abstract.base_env import BaseEnv as GymClassicsBaseEnv
import gymnasium as gym

# np.argmax does not break ties randomly
def random_argmax(x, axis = None):
        if axis is None:
            return np.random.choice(np.where(x == np.max(x))[0])
        else:
            return np.apply_along_axis(random_argmax, axis, x)

def random_policy(env):
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert isinstance(env.observation_space, gym.spaces.Discrete)
    
    return np.random.choice(env.action_space.n, size=env.observation_space.n)

# only for gym-classics environments!
def encode_policy(env, policy, type = "text"):
    assert isinstance(env, GymClassicsBaseEnv)
    
    return [env.unwrapped.encode_action(a, type = type) for a in policy]


# this is a copy from DP to prevent circular imports.
def _backup(env, discount, V, state, action):
    V = np.array(V)
    
    next_states, rewards, terminals, probs = env.model(state, action)
    bootstraps = (1.0 - terminals) * V[next_states]
    return np.sum(probs * (rewards + discount * bootstraps))

def greedy_policy(env, V, discount=1):
    """
    Calculate the greedy policy for a given value function.
    
    :param env: the environment
    :param V: the value function as a 1-D numpy array
    :param discount: discount factor
    """
    
    assert isinstance(env, GymClassicsBaseEnv)
    assert 0.0 <= discount <= 1.0
    
    policy = np.zeros(env.observation_space.n, dtype=np.int64)

    env = env.unwrapped
    
    for s in env.states():
        Q_values = [_backup(env, discount, V, s, a) for a in range(env.action_space.n)]
        policy[s] = random_argmax(Q_values)

    return policy

def greedy_policy_Q(env, Q, discount=1):
    """
    Calculate the greedy policy for a given value function.
    
    :param env: the environment
    :param Q: the action value function as a state-by-action numpy array
    :param discount: discount factor
    """
    
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert isinstance(env.observation_space, gym.spaces.Discrete)
    assert 0.0 <= discount <= 1.0

    return random_argmax(Q, axis=1)

