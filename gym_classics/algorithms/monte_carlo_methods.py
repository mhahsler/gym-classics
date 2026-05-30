"""This file implements tabular Monte Carlo methods for policy evaluation and control in gym-classics environments 
with discrete state spaces.
"""

import numpy as np
import random
from collections import defaultdict
import gymnasium as gym
from gym_classics.envs.abstract.gridworld import Gridworld
from gym_classics.envs.abstract.base_env import BaseEnv
from gym_classics.algorithms.policy import random_policy, random_argmax, make_multidiscrete_policy
from tqdm import tqdm

def states(env):
    """Returns a list of all states in the environment."""
    assert isinstance(env.observation_space, gym.spaces.Discrete) or isinstance(env.observation_space, gym.spaces.MultiDiscrete), "Tabular methods require discrete state space."  
    
    if isinstance(env.observation_space, gym.spaces.Discrete):
        return list(range(env.observation_space.n))
    elif isinstance(env.observation_space, gym.spaces.MultiDiscrete):
        return [tuple(s) for s in np.array(np.meshgrid(*[range(n) for n in env.observation_space.nvec])).T.reshape(-1, len(env.observation_space.nvec))]
    else:
        raise ValueError("Unsupported observation space type for state enumeration.")


def sample_episode(env, policy = None, start_state = None, start_action = None, epsilon = 0, max_len = 1000, verbose = False):
    """
    Samples an episode from the environment using the given policy and starting conditions.
    
    Args: 
        env: The environment to sample from.
        policy: A mapping from states to actions. For discrete state spaces a action list in the order of states. 
                For 
        If None, a random policy will be used.
        start_state: The state to start the episode from (if None, the environment's default starting state will be used).
        start_action: The action to take in the first step of the episode (if None, the action will be chosen according to the policy or randomly if no policy is given).
        epsilon: The probability of taking a random action instead of the policy's action at each step (for epsilon-greedy exploration).
        max_len: The maximum length of the episode to prevent infinite loops.
        verbose: If True, prints the state transitions and rewards for each step in the episode.
    
    Returns:
        A list of (state, action, reward, next_state) tuples representing the episode. 
        The episode ends when a terminal state is reached or when max_len steps have been taken.    
        Note that the last tuple in the episode will have a next_state that is either terminal or the 
        state at which the episode was truncated due to max_len.  
    """
    
    assert isinstance(env.observation_space, gym.spaces.Discrete) or isinstance(env.observation_space, gym.spaces.MultiDiscrete), "Tabular methods require discrete state space."  
    assert 0.0 <= epsilon <= 1.0
    assert max_len > 0
    
    if policy is None:
        policy = random_policy(env)
        epsilon = 1.0
        if verbose:
            print("*** No policy given, sampling using random actions!")
    
    if isinstance(env.observation_space, gym.spaces.MultiDiscrete):
        policy = make_multidiscrete_policy(policy, env)
    
    episode = []
    s, r = env.reset()
    
    # force custom start state could be a state or an state index.
    if not start_state is None:
        if isinstance(env, BaseEnv) and env.tabular:
            env.state = env.id2state(start_state)
        elif isinstance(env.observation_space, gym.spaces.Discrete):
            env.state = int(start_state)
        else:
            raise ValueError("Custom start state is only supported for tabular gym-classics environments and environments with a MultiDiscrete observation space.")
        s = start_state
            
    step = 0
    done = False
    while not done and step < max_len:
        a = policy[s]
        
        # epsilon greedy choice?
        if epsilon > 0:
            if (random.random() < epsilon):
                a = env.action_space.sample()
        
        # for exploring starts
        if step == 0 and not start_action is None:
            a = start_action
        
        sp, reward, done, _, _ = env.step(a)
        
        episode.append((s,a,reward,sp))
        
        if verbose:
            print ("Step", step, "(s,a,r,s')=",s,a,reward,sp)
        
        s = sp
        step += 1
        
    return(episode)


def on_policy_state_distribution(env, pol, discount = 1, epsilon = 0, n = 100):
    """Estimate the (discounted) state distribution of a policy by sampling episodes."""
    
    assert isinstance(env.observation_space, gym.spaces.Discrete), "Tabular methods require discrete state space."   
    assert 0.0 <= epsilon <= 1.0
    assert 0.0 < discount <= 1.0
    assert n > 0
    
    state_cnts = np.zeros(env.observation_space.n)
    
    for _ in tqdm(range(n), desc="Sampling Episodes", disable=verbose):
        episode = np.array(sample_episode(env, policy=pol, epsilon = epsilon))
        states = np.append(episode[:,0], episode[-1,3])
        discounts = np.array([discount**i for i in range(len(states))])
        for s, d in zip(states, discounts):
            state_cnts[int(s)] += d
    
    state_prob = state_cnts / sum(state_cnts)
    return state_prob


def MC_prediction(env, policy, discount, n = 100, max_episode_len = 100, verbose = False):
    assert isinstance(env.observation_space, gym.spaces.Discrete), "Tabular methods require discrete state space."  
    assert n > 0
    assert max_episode_len > 0
    
    Returns = defaultdict(list) # a list for each s
    
    for i in tqdm(range(n), desc="MC Prediction", disable=verbose):
        if verbose:
            print("episode", i," of ", n)
    
        episode = sample_episode(env, policy, max_len=max_episode_len)
        G = 0
        
        # for first visit check for s
        visited = [s for s,a,r,sp in episode]
        
        # process episode in reverse order
        for t in range(len(episode)-1, -1, -1):
            s,a,r,sp = episode[t]
            G = discount * G + r

            # use first visit of s only
            if not s in visited[0:t]:
                Returns[s].append(G)

    Vs = np.array([np.mean(Returns[s]) if len(Returns[s]) else np.nan for s in range(env.observation_space.n)])
    return Vs

# this version does not use incremental updates and is very slow!
def MC_control_ES_textbook(env, discount, n = 100, Q = None, max_episode_len = 100, history = False, verbose = False): 
    assert isinstance(env.observation_space, gym.spaces.Discrete), "Tabular methods require discrete state space."  
    assert n > 0
    assert max_episode_len > 0
    assert Q is None or Q.shape == (len(env.states()), env.action_space.n)
    
    policy = random_policy(env)
    
    if Q is None:
        Q = np.zeros((len(env.states()), env.action_space.n))
    
    # lists that grow are very slow. We should use a running average instead. But this is easier to read and understand.
    Returns = defaultdict(list) # a list for each (s,a)
    
    if history:
        Q_list = []
        Q_list.append(Q.copy())
        pol_list = []
        pol_list.append(policy.copy())
        ep_list = []
        return_list = []
    
    
    for i in tqdm(range(n), desc="MC Control", disable=verbose):
        if verbose:
            print("episode", i," of ", n)
        
        # Sample starting (s,a)
        s = env.observation_space.sample()
        a = env.action_space.sample()
        
        episode = sample_episode(env, policy, start_state = s, start_action = a, max_len=max_episode_len, verbose = verbose >1)
        G = 0
        
        # for first visit check for (s,a)
        visited = [(s,a) for s,a,r,sp in episode]
        
        # process episode in reverse order
        for t in range(len(episode)-1, -1, -1):
            s,a,r,sp = episode[t]
            G = discount * G + r

            # use first visit of (s,a) only
            if not (s,a) in visited[0:t]:
                Returns[(s,a)].append(G)

                # update policy
                Q[s,a] = np.mean(Returns[(s,a)])
                policy[s] = random_argmax(Q[s,:])
                
        if history:
            pol_list.append(policy.copy())
            Q_list.append(Q.copy())
            ep_list.append(episode.copy())
            return_list.append(G)
          
    if history:
        return policy, Q, {'policies': pol_list, 'Q_values': Q_list, 'episodes': ep_list, 'returns': return_list}
          
    return policy, Q


# incremental version of MC control with exploring starts. This is more efficient and can be used for infinite horizon problems. It gives the same results as the non-incremental version, but it does not store all returns in memory.
def MC_control_ES(env, discount, n=100, Q=None, max_episode_len=100, history=False, verbose=False):
    """Monte Carlo Control with Exploring Starts (incremental version).
    This algorithm estimates the optimal action-value function Q and the corresponding greedy policy by sampling episodes with exploring starts. It uses incremental updates to compute the average returns for each (s,a) pair, which is more memory efficient than storing all returns.
    Args: env: The environment to interact with. Must have discrete state and action spaces.
        discount: The discount factor (gamma) for future rewards. Should be in (0, 1].
        n: The number of episodes to sample for learning. Must be a positive integer.
        Q: Optional initial action-value function. If None, it will be initialized to zeros.
        max_episode_len: Maximum length of each episode to prevent infinite loops. Must be a positive integer.
        history: If True, the function will return the history of policies, Q-values, and
                 and episodes for each iteration. This can be useful for analysis and visualization, but it will consume more memory.
        verbose: If True, the function will print progress and episode details. If verbose > 1, it will also print the state transitions and rewards for each step in the episode.
    Returns:    If history is False: A tuple (policy, Q) where policy is the learned greedy policy and Q is the learned action-value function.
        If history is True: A tuple (pol_list, Q_list, ep_list) where pol_list is a list of policies for each iteration, Q          
        is a list of Q-value functions for each iteration, and ep_list is a list of episodes sampled in each iteration.
    """
    
    assert isinstance(env.observation_space, gym.spaces.Discrete), "Tabular methods require discrete state space."  
    assert n > 0
    assert max_episode_len > 0
    assert Q is None or Q.shape == (env.observation_space.n, env.action_space.n)

    policy = random_policy(env)

    if Q is None:
        Q = np.zeros((env.observation_space.n, env.action_space.n))

    # Count how many first-visit returns have been used for each (s, a)
    N = np.zeros((env.observation_space.n, env.action_space.n), dtype=int)

    if history:
        Q_list = []
        Q_list.append(Q.copy())
        pol_list = []
        pol_list.append(policy.copy())
        ep_list = []
        return_list = []

    for i in tqdm(range(n), desc="MC Control (Incremental)", disable=verbose):
        if verbose:
            print("episode", i, "of", n)

        # Exploring starts
        s = env.observation_space.sample()
        a = env.action_space.sample()

        episode = sample_episode(
            env,
            policy,
            start_state=s,
            start_action=a,
            max_len=max_episode_len,
            verbose=verbose > 1
        )

        G = 0

        # Process episode backward
        visited = set()
        for t in range(len(episode) - 1, -1, -1):
            s, a, r, sp = episode[t]
            G = discount * G + r

            # First-visit MC: only update the first occurrence from the start
            if (s, a) not in visited:
                visited.add((s, a))

                N[s, a] += 1
                Q[s, a] += (G - Q[s, a]) / N[s, a]

                # Improve policy greedily
                policy[s] = random_argmax(Q[s, :])

        if history:
            pol_list.append(policy.copy())
            Q_list.append(Q.copy())
            ep_list.append(episode.copy())
            return_list.append(G)

    if history:
        return policy, Q, {'policies': pol_list, 'Q_values': Q_list, 'episodes': ep_list, 'returns': return_list}

    return policy, Q