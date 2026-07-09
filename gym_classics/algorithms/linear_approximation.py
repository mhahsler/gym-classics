"""This file implements linear function approximation algorithms for policy evaluation and control. This is not a tabular approach
and does not require discrete state spaces. The user needs to implement the state_features function to convert states to feature 
vectors.
"""

import numpy as np
from itertools import product
import warnings

import gymnasium as gym
from gym_classics.algorithms.policy import random_policy, random_argmax
from gym_classics.algorithms.schedules import Schedule, ConstantSchedule
from gym_classics.envs.abstract.base_env import BaseEnv as GymClassicsBaseEnv

from tqdm import tqdm

def state_features(s,env):
    """
    Converts a state to a state feature vector. It needs to be overwritten by the user to implement different feature representations. 
    This could be linear features, tile coding, radial basis functions, Fourier basis functions, or even a neural network.
    
    :param s: state
    :param env: environment instance   
    
    :return a state feature vector
    """
    raise NotImplementedError("state_features function needs to be implemented by the user. By default, it just concatenates a constant feature (for the intercept) with the state itself. This is equivalent to linear function approximation with a tabular representation.")
   

def active_weights(a, sf_len):
    """helper for q_hat()"""
    return [0] + list(range(a*sf_len+1, a*sf_len+sf_len+1))

def state_action_features(s,a,env):
    s = state_features(s,env)
    x = np.zeros(1+len(s)*env.action_space.n)
    x[active_weights(a, len(s)-1)] = s
    return x

def v_hat(s, w, env):
    """
    Estimate Value function
    
    :param s: state id
    :param w: weight vector
    :param env: environment instance
    
    :return the state value estimate
    """
    return np.dot(w, state_features(s, env))

def q_hat(s, a, w, env):
    """
    Estimate the action value function.
    
    :param s: state id
    :param a: action
    :param w: weight vector
    :param env: environment instance
    :return the state-action value estimate
    """    
    x = state_action_features(s, a, env)
    return np.dot(w, x)

def epsilon_greedy_action_w(env, w, state, epsilon = 0):
    """
    Get an epsilon-greedy action for a given policy.
    
    :param w: weight vector for the action-value function approximator
    :param env: environment instance
    :param state: the current state
    :param epsilon: the probability of taking a random action
    """
    
    if epsilon>0 and np.random.rand() < epsilon:
        return env.action_space.sample()
    
    return random_argmax([q_hat(state, a, w, env) for a in range(env.action_space.n)])


def MSVE(V, V_true, weight=None):
    """
    Calculate the (weighted) mean squared value error.
    
    :param V: value function to evaluate
    :param V_true: the value function to compare to
    :param weight: weight for each state. Typically the stationary state visit distribution.
    """
    if weight is None:
        weight = np.ones(len(V))
    
    return np.sum(weight * (V - V_true)**2)


def semi_gradient_TD0_estimation(env, policy, n, alpha, gamma, max_episode_length=1000, verbose =False):
    """
    Estimate the state-value function using the semi-gradient TD(0) algorithm.

    This function runs TD(0) learning with function approximation over multiple
    episodes generated from a given policy and environment. Updates are performed
    using the semi-gradient of the value function approximation.

    Parameters
    ----------
    env : Environment following the Gym interface from which episodes are sampled.
    policy : a deterministic policy as a vector.
    n : int
        Number of episodes to run for value estimation. Must be positive.
    alpha : float
        Step-size (learning rate) for TD updates. Must be in the interval (0, 1].
    gamma : float
        Discount factor for future rewards. Must be in the interval [0, 1].
    max_episode_length : int, optional
        Maximum number of time steps per episode (default is 1000).
    verbose : bool, optional
        If True, prints progress or diagnostic information during training
        (default is True).

    Returns
    -------
    w
        Returns the learned weight vector for the approximate value function.
    """
    assert gamma >= 0 and gamma <= 1, "Gamma must be in [0,1]"
    assert n > 0, "Number of episodes must be positive"
    assert max_episode_length > 0, "Max episode length must be positive"
    
    if isinstance(env.observation_space, gym.spaces.Discrete):
        warnings.warn("The environment has a discrete state space. Consider using a tabular method instead of function approximation.")
    
    if not isinstance(alpha, Schedule):
        alpha = ConstantSchedule(alpha)

    state, _ = env.reset()
    w = np.zeros(len(state_features(state, env)))  # Initialize weights (intercept + x and y)

    for episode in tqdm(range(n), desc="Semi-Gradient TD(0)", disable=verbose):
        state, _ = env.reset()
        done = False

        i = 0
        while not done and i < max_episode_length:
            action = policy[state]  # follow policy
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Semi-gradient TD(0) update
            # Note: v_hat(terminal, w) needs to be 0
            if terminated:
                w += alpha(episode) * (reward - v_hat(state, w, env)) * state_features(state, env)    
            else: 
                w += alpha(episode) * (reward + gamma * v_hat(next_state, w, env) - v_hat(state, w, env)) * state_features(state, env)
             
            if verbose:
                print (f"Episode {episode+1}, Step {i+1}: S={state}, A={action}, R={reward}, S'={next_state}, w={w}")

            state = next_state
            i += 1

    return w


def semi_gradient_Sarsa_0(env, n, epsilon, alpha, gamma, w = None, max_episode_length=1000, verbose = False, history = False):
    """
    Semi-gradient SARSA: on-policy control with function approximation.

    Implements the **semi-gradient SARSA** algorithm for estimating the optimal
    action-value function q_*(s, a) using a differentiable function approximator
    q̂(s, a, w). Actions are selected according to an ε-greedy policy derived
    from the current action-value estimate.

    Episodes are truncated after `max_episode_length` time steps.

    Parameters
    ----------
    env : Episodic environment used to generate experience.
    n : int
        Number of episodes over which to perform control learning.
    epsilon : float
        Exploration parameter for the epsilon-greedy behavior policy (0 <= epsilon <= 1).
    alpha : float
        Step-size parameter for the weight update (0 < alpha <= 1).
    gamma : float
        Discount factor (0 <= gamma <= 1).
    w : array-like or None, optional
        Initial weight vector for the action-value function approximator.
        If None, weights are initialized internally.
    max_episode_length : int, optional
        Maximum number of time steps per episode before truncation (default 1000).
    verbose : bool, optional
        If True, prints progress diagnostics during learning (default True).

    Returns
    -------
    w
        Returns the learned weight vector for the approximate value function.
    """
    
    assert gamma >= 0 and gamma <= 1, "Gamma must be in [0,1]"
    assert n > 0, "Number of episodes must be positive"
    assert max_episode_length > 0, "Max episode length must be positive"

    if isinstance(env.observation_space, gym.spaces.Discrete):
        warnings.warn("The environment has a discrete state space. Consider using a tabular method instead of function approximation.")

    if not isinstance(alpha, Schedule):
        alpha = ConstantSchedule(alpha)
    if not isinstance(epsilon, Schedule):
        epsilon = ConstantSchedule(epsilon)

    if w is None:
        state, _ = env.reset()
        w = np.zeros(len(state_action_features(state, 0, env)))

    if history:
        ws = []
        ws.append(w.copy())
        returns = []
        ep_lens = []
    
    for episode in tqdm(range(n), desc="Semi-Gradient SARSA(0)", disable=verbose):
        state, _ = env.reset()
        action = epsilon_greedy_action_w(env, w, state, epsilon(episode))
        done = False

        i = 0
        while not done and i < max_episode_length:
            if history:
                G = 0

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            x = state_action_features(state, action, env)
            
            if terminated:
                next_action = None
                w += alpha(episode) * (reward - q_hat(state, action, w, env)) * x
                
            else:
                next_action = epsilon_greedy_action_w(env, w, next_state, epsilon(episode))
                w += alpha(episode) * (reward + gamma * q_hat(next_state, next_action, w, env) - q_hat(state, action, w, env)) * x

            if verbose:
                print (f"Episode {episode+1}, Step {i+1}: S={state}, A={action}, R={reward}, S'={next_state}, w={w}")

            state = next_state
            action = next_action
            i += 1
            
            if history:
                G += reward * (gamma ** (i-1))
            
        if history:
            ws.append(w.copy())
            returns.append(G)
            ep_lens.append(i)

    if history:
        return w, {'ws': ws, 'returns': returns, 'ep_lens': ep_lens}
    
    return w  


# product from itertools is the cartesian product
def create_fourier_basis_coefs(dim, order): 
    """ Create Fourier basis coefficients for given dimension and order. 
        param dim: dimension of the state features
        param order: order of the Fourier basis
    """  
    return np.array(list(product(range(order+1), repeat=dim)))
    
def transformation_fourier_basis(min, max, order):
    """ Create a Fourier basis transformation function for given min/max ranges and order.
    
        To use this transformation with semi_gradient_Sarsa you need to overwrite the state_features 
        function like this:
        
        def state_features(s, env): return trans_fb(env.decode(s))
        gym_classics.algorithms.linear_approximation.state_features = state_features
        
        param min: minimum values for each dimension
        param max: maximum values for each dimension
        param order: order of the Fourier basis
    """  
    
    min = np.array(min)
    max = np.array(max)
    coefs = create_fourier_basis_coefs(len(min), order)
    
    def fourier_basis(s):
        # normalize state to [0,1]
        s = np.array(s)
        assert s.shape == min.shape, "State dimension does not match Fourier basis dimension"
        
        norm_s = (s - min) / (max - min)
        return np.cos(np.pi * np.dot(coefs, norm_s))
    
    return fourier_basis