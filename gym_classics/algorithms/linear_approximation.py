import numpy as np
from itertools import product

from gym_classics.algorithms.policy import random_policy, random_argmax
from gym_classics.envs.abstract.base_env import BaseEnv as GymClassicsBaseEnv

from tqdm import tqdm

def state_features(s):
    """
    Convert the state id into state features. This function needs to be overwritten for the environment
    
    :param s: state id
    :return a state feature vector
    """
    raise NotImplementedError("state_features function must be implemented and overwrite gym_classics.algorithms.linear_approximation.state_features.") 

def v_hat(s, w):
    """
    Estimate Value function
    
    :param s: state id
    :param w: weight vector
    :return the state value estimate
    """
    return np.dot(w, state_features(s))

def active_weights(a, sf_len):
    """helper for q_hat()"""
    return [0] + list(range(a*sf_len+1, a*sf_len+sf_len+1))

def q_hat(s, a, w):
    """
    Estimate the action value function.
    
    :param s: state id
    :param a: action
    :param w: weight vector
    :return the state-action value estimate
    """
    sf_len = state_features(0).shape[0]-1
    active_weights = lambda a: [0] + list(range(a*sf_len+1, a*sf_len+sf_len+1))
    return np.dot(w[active_weights(a)], state_features(s))

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


def schedule(episode, value_0=0.001, k=0.001):
    return value_0 / (1 + k * episode)


def semi_gradient_TD0_estimation(env, policy, n, alpha, gamma, max_episode_length=1000, verbose =False):
    """
    Estimate the state-value function using the semi-gradient TD(0) algorithm.

    This function runs TD(0) learning with function approximation over multiple
    episodes generated from a given policy and environment. Updates are performed
    using the semi-gradient of the value function approximation.

    Parameters
    ----------
    env : GymClassicsBaseEnv
        Environment following the Gym interface from which episodes are sampled.
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
    assert isinstance(env, GymClassicsBaseEnv), "env must be an instance of gym.Env"
    assert alpha > 0 and alpha <= 1, "Alpha must be in (0,1]"
    assert gamma >= 0 and gamma <= 1, "Gamma must be in [0,1]"
    assert n > 0, "Number of episodes must be positive"
    assert max_episode_length > 0, "Max episode length must be positive"
    
    w = np.zeros(state_features(0).shape[0])  # Initialize weights (intercept + x and y)

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
                w += alpha * (reward - v_hat(state, w)) * state_features(state)    
            else: 
                w += alpha * (reward + gamma * v_hat(next_state, w) - v_hat(state, w)) * state_features(state)
             
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
    env : GymClassicsBaseEnv
        Episodic environment used to generate experience.
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
    
    assert isinstance(env, GymClassicsBaseEnv), "env must be an instance of gym.Env"
    assert alpha > 0 and alpha <= 1, "Alpha must be in (0,1]"
    assert gamma >= 0 and gamma <= 1, "Gamma must be in [0,1]"
    assert epsilon >=0 and epsilon <=1, "Epsilon must be in [0,1]"
    assert n > 0, "Number of episodes must be positive"
    assert max_episode_length > 0, "Max episode length must be positive"

    sf_len = state_features(0).shape[0]-1
    active_weights = lambda a: [0] + list(range(a*sf_len+1, a*sf_len+sf_len+1))

    if w is None:
        w = np.zeros(1 + state_features(0).shape[0] * env.action_space.n)  # Initialize weights (intercept + action weights)

    if history:
        ws = []
        ws.append(w.copy())
        returns = []
        returns.append(None)

    # helper used later
    def epsilon_greedy_action(state, epsilon):
        if np.random.rand() < epsilon:
            return env.action_space.sample()
        else:
            q_values = [q_hat(state, a, w) for a in range(env.action_space.n)]
            return np.argmax(q_values)
    
    for episode in tqdm(range(n), desc="Semi-Gradient SARSA(0)", disable=verbose):
        state, _ = env.reset()
        action = epsilon_greedy_action(state, epsilon)
        done = False

        i = 0
        while not done and i < max_episode_length:
            if history:
                G = 0

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            if terminated:
                next_action = None
                w[active_weights(action)] += alpha * (reward - q_hat(state, action, w)) * state_features(state)
                
            else:
                next_action = epsilon_greedy_action(next_state, epsilon)
                w[active_weights(action)] += alpha * (reward + gamma * q_hat(next_state, next_action, w) - q_hat(state, action, w)) * state_features(state)
             
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

    if history:
        return ws, returns
    
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
        
        def state_features(s): return trans_fb(env.decode(s))
        gym_classics.algorithms.linear_approximation.state_features = state_features
        
        param min: minimum values for each dimension
        param max: maximum values for each dimension
        param order: order of the Fourier basis
    """  
    coefs = create_fourier_basis_coefs(len(min), order)
    
    def fourier_basis(s):
        # normalize state to [0,1]
        norm_s = (s - min) / (max - min)
        return np.cos(np.pi * np.dot(coefs, norm_s))
    
    return fourier_basis