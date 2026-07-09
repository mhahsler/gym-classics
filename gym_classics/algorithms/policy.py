"""This file implements different policy representations and functions for working with policies in gym-classics environments.
It includes functions for creating random policies, encoding policies for display, and computing greedy policies based
"""

import numpy as np
import random

import gymnasium as gym

from gym_classics.utils import random_argmax
from gym_classics.envs.abstract.base_env import BaseEnv as GymClassicsBaseEnv

def make_multidiscrete_policy(policy, env):
    """
    Converts a tabular policy vector to a multi-discrete tabular policy stored in a dictionary that can be used for sample.
    """
    assert isinstance(env.observation_space, gym.spaces.MultiDiscrete), "Requires an environment with a multi-discrete state spaces."
    
    if isinstance(policy, dict):
        return policy
        
    if isinstance(env, GymClassicsBaseEnv):
        return dict(zip(env.id2state(list(env.states())), [policy[s] for s in env.states()]))
    else:
        raise ValueError("Unsupported environment type")


def random_policy(env):
    """
    Create a random policy for the given environment. 
    The policy is represented as a numpy array where each entry corresponds to an action for a state. 
    """
    
    if isinstance(env.observation_space, gym.spaces.Discrete):
        return np.random.choice(env.action_space.n, size = env.observation_space.n)
    else:
        assert isinstance(env, GymClassicsBaseEnv), "Only gym-classics environments are supported for random policies with multi-discrete state spaces."
        return make_multidiscrete_policy(np.random.choice(env.action_space.n, size = len(env.states())), env)

# only for gym-classics environments!
def encode_policy(env, policy, type = "text"):
    """
    Encode a policy for display. The policy is represented as a numpy array where each entry corresponds to an action for a state.
    The function returns a list of action names corresponding to the actions in the policy.
    """
    
    assert isinstance(env, GymClassicsBaseEnv)
    
    return [env.unwrapped.id2action(a, type = type) for a in policy]


# this is a copy from DP to prevent circular imports.
def _backup(env, discount, V, s, a):
    V = np.array(V)
    
    next_states, rewards, terminals, probs = env.model(s, a)
    bootstraps = (1.0 - terminals) * V[next_states]
    return np.sum(probs * (rewards + discount * bootstraps))

def greedy_policy(env, V, discount=1):
    """
    Calculate the greedy policy for a given value function.
    
    :param env: the environment
    :param V: the value function as a 1-D numpy array
    :param discount: discount factor
    """
    assert isinstance(env, GymClassicsBaseEnv), "greedy_policy requires a gym-classics environment with discrete state space."
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert isinstance(env.observation_space, gym.spaces.Discrete)
    assert 0.0 <= discount <= 1.0
    
    policy = np.zeros(len(env.states()), dtype=np.int64)

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

def epsilon_greedy_action(policy, state = None, epsilon = 0):
    """
    Get an epsilon-greedy action for a given tabular policy.
    
    :param policy: the policy as a 1-D numpy array
    :param epsilon: the probability of taking a random action
    :param state: the current state
    """
    
    if epsilon>0 and np.random.rand() < epsilon:
        return np.random.choice(len(policy)) if state is None else np.random.choice(len(policy[state]))   
    
    if state is None:
        return random_argmax(policy)
    else:
        return random_argmax(policy[state])