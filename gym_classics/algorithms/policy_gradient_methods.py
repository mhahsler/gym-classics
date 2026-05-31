"""This file implements policy gradient methods for learning parameterized policies.
The main algorithm implemented is REINFORCE, which is a Monte Carlo policy gradient method that updates policy 
parameters based on the returns observed in sampled episodes. The policy is represented using a softmax function 
over linear state-action features, and the algorithm estimates the policy gradient using the log-likelihood of actions taken 
in the episodes. This implementation allows for learning stochastic policies that can handle exploration and exploitation in 
reinforcement learning tasks.

The user has to overwrite the state_features function to convert state ids into feature vectors suitable for the environment 
being used.
"""

from tqdm import tqdm
import numpy as np
import warnings

import gymnasium as gym
from gym_classics.algorithms.linear_approximation import state_features, state_action_features, active_weights, q_hat  
from gym_classics.envs.abstract.base_env import BaseEnv as GymClassicsBaseEnv
from gym_classics.algorithms.schedules import ConstantSchedule

def h(s,a,theta,env):
    return np.dot(theta, state_action_features(s,a,env))

def pi(s,theta,env):
    hs = np.array([h(s,a,theta,env) for a in range(env.action_space.n)])
    exp_hs = np.exp(hs)
    return exp_hs / np.sum(exp_hs)

def sample_episode_policy(env, pi, theta, max_episode_length=1000):
    """
    Sample an episode using the policy defined by pi and theta.
    Returns a list of (state, action, reward) tuples.
    """
    s, _ = env.reset()
    episode_data = []
    
    for t in range(max_episode_length):
        a = np.random.choice(range(env.action_space.n), p=pi(s,theta,env))
        next_s, r, done, _, _ = env.step(a)
        episode_data.append((s, a, r, next_s))
        s = next_s
        
        if done:
            break
    
    return episode_data

def REINFORCE(
    env,
    n,
    alpha,
    gamma,
    theta = None,
    max_episode_length=1000,
    verbose=False,
    history=False
    ):
    """REINFORCE: Monte Carlo policy gradient method with linear function approximation.
    Parameters
    ----------
    env : GymClassicsBaseEnv
        Episodic environment used to generate experience.
    n : int
        Number of episodes.
    alpha : float
        Step size.
    gamma : float
        Discount factor.
    theta : array-like or None
        Initial policy parameters. If None, initializes to zeros.
    max_episode_length : int
        Maximum number of steps per episode.
    verbose : bool
        Whether to print step-by-step diagnostics.
    history : bool
        Whether to return learning history (returns, episode lengths, parameter values).    
    """
    
    assert gamma >= 0 and gamma <= 1, "gamma must be in [0  ,1]"
    assert n > 0, "number of episodes must be positive"
    assert max_episode_length > 0, "max episode length must be positive"
    
    if isinstance(env.observation_space, gym.spaces.Discrete):
        warnings.warn("The environment has a discrete state space. Consider using a tabular method instead of function approximation.")
    
    if isinstance(alpha, float):
        alpha = ConstantSchedule(alpha)

    if theta is None:
        state, _ = env.reset()
        theta = np.zeros(1+len(state_features(state, env))*env.action_space.n)
    
    if history:
        returns = []        
        ep_lens = []
        thetas = []
        thetas.append(theta.copy())
        
    for episode in tqdm(range(n), desc="Episodes", disable=verbose):
        if verbose:
            print(f"Episode {episode+1}/{n}")
        
        # sample complete episode using pi (this is a MC method)
        episode_data = sample_episode_policy(env, pi, theta, max_episode_length)
        
        for t in range(len(episode_data)):
            # update policy for each step in the episode using the return observed from that step on
            #print(episode_data[t])
        
            G = np.sum([e[2] for e in episode_data[t:]] * (gamma ** np.arange(len(episode_data[t:]))))
            if history and t == 0:
                returns.append(G)   

            s,a,r,next_s = episode_data[t]
            
            # ln policy gradient= x(s,a)- sum_b pi(b|s,theta) x(s,b)
            grad_log_pi = state_action_features(s, a, env) - sum([pi(s, theta, env)[b] * state_action_features(s, b, env) for b in range(env.action_space.n)])
            
            if verbose: 
                print (f"t: {t}, G: {G:.2f}, grad_log_pi: {grad_log_pi}")
            
            theta += alpha(episode) * (gamma**t) * G * grad_log_pi
            
        if history:
            thetas.append(theta.copy())
            ep_lens.append(len(episode_data))
            
    if history:
        return theta, {'thetas': thetas, 'returns': returns, 'ep_lens': ep_lens}
       
    return theta


def AC(
    env,
    n,
    alpha_policy,
    alpha_value,
    gamma,
    max_episode_length=1000,
    verbose=False,
    history=False
    ):
    """Actor-Critic: Policy gradient method with linear function approximation and TD learning for the value function.
    
    Parameters
    ----------
    env : Episodic environment used to generate experience.
    n : int
        Number of episodes.
    alpha_policy : float or Schedule
        Step size for policy updates. If a float is provided, it will be converted to a ConstantSchedule. If a Schedule is provided, it will be used directly.
    alpha_value : float or Schedule
        Step size for value function updates. If a float is provided, it will be converted to a ConstantSchedule. If a Schedule is provided, it will be used directly. 
    gamma : float
        Discount factor.
    max_episode_length : int
        Maximum number of steps per episode.
    verbose : bool
        Whether to print step-by-step diagnostics.
    history : bool
        Whether to return learning history (returns, episode lengths, parameter values).
        
        Returns
        -------
        theta : array-like
            Final policy parameters after training.
        w : array-like
            Final value function parameters after training.
        history : dict (optional)
            If history=True, a dictionary containing the learning history with keys: 
                'thetas': list of policy parameter vectors at each episode,
                'ws': list of value function parameter vectors at each episode,
                'returns': list of returns observed at the end of each episode,
                'ep_lens': list of episode lengths (number of steps) for each episode.
        """
    
    assert gamma >= 0 and gamma <= 1, "gamma must be in [0  ,1]"
    assert n > 0, "number of episodes must be positive"
    assert max_episode_length > 0, "max episode length must be positive"
    
    if isinstance(env.observation_space, gym.spaces.Discrete):
        warnings.warn("The environment has a discrete state space. Consider using a tabular method instead of function approximation.")
    
    if isinstance(alpha_policy, float):
        alpha_policy = ConstantSchedule(alpha_policy)
    if isinstance(alpha_value, float):
        alpha_value = ConstantSchedule(alpha_value) 
    
    state, _ = env.reset()
    
    # for simplicity we use the same features for the value function and the policy approximation
    # value function weights
    w = np.zeros(len(state_features(state, env)))
    # policy weights
    theta = np.zeros(len(state_action_features(state, 0, env)))
    
    if history:
        returns = []        
        ep_lens = []
        ws = []
        ws.append(w.copy())
        thetas = []
        thetas.append(theta.copy())
        
    for episode in tqdm(range(n), desc="Episodes", disable=verbose):
        if verbose:
            print(f"Episode {episode+1}/{n}")
        
        disc_factor = 1.0
        state, _ = env.reset()
        done = False
        i = 0
        G = 0.0
        
        while not done and  i < max_episode_length:
            # use actor to determine next action
            a = np.random.choice(env.action_space.n, p=pi(state, theta, env))

            # execute action
            next_state, r, done, _, _ = env.step(a)
            
            # use critic to calculate  TD error
            td_error = r + gamma * np.dot(w, state_features(next_state, env)) - np.dot(w, state_features(state, env))
            
            # update critic
            w += alpha_value(episode) * td_error * state_features(state, env)
            
            # update actor
            grad_log_pi = state_action_features(state, a, env) - sum([pi(state, theta, env)[b] * state_action_features(state, b, env) for b in range(env.action_space.n)])
            theta += alpha_policy(episode) * disc_factor * td_error * grad_log_pi
        
            G += disc_factor * r
            disc_factor *= gamma      
            state = next_state
            i += 1
            
        if history:
            thetas.append(theta.copy())
            ws.append(w.copy())
            returns.append(G)
            ep_lens.append(i)
            
    if history:
        return theta, w, {'thetas': thetas, 'returns': returns, 'ep_lens': ep_lens}
       
    return theta, w