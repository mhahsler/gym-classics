import numpy as np
import random
from collections import defaultdict
import gymnasium as gym
from gym_classics.envs.abstract.gridworld import Gridworld
from gym_classics.algorithms.policy import random_policy, random_argmax
from tqdm import tqdm

def sample_episode(env, policy = None, start_state = None, start_action = None, epsilon = 0, max_len = 1000, verbose = False):
    
    assert 0.0 <= epsilon <= 1.0
    assert max_len > 0
    
    if policy is None:
        policy = random_policy(env)
        epsilon = 1.0
        if verbose:
            print("*** No policy given, sampling using random actions!")
    
    episode = []
    s, r = env.reset()
    
    if not start_state is None:
        s = start_state
        if isinstance(env, Gridworld):
            env.state = env.unwrapped.decode(start_state)
        else:
            env.state = start_state
            
    
    step = 0
    done = False
    while not done and step < max_len:
        a = policy[s]
        
        # epsilon greedy choice?
        if epsilon > 0:
            if (random.random() < epsilon):
                a = np.random.choice(env.action_space.n) 
        
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
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert isinstance(env.observation_space, gym.spaces.Discrete)
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
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert isinstance(env.observation_space, gym.spaces.Discrete)
    assert n > 0
    assert max_episode_len > 0
    assert Q is None or Q.shape == (env.observation_space.n, env.action_space.n)
    
    policy = random_policy(env)
    
    if Q is None:
        Q = np.zeros((env.observation_space.n, env.action_space.n))
    
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
        s = np.random.choice(env.observation_space.n)
        a = np.random.choice(env.action_space.n)
        
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
    
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert isinstance(env.observation_space, gym.spaces.Discrete)
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
        s = np.random.choice(env.observation_space.n)
        a = np.random.choice(env.action_space.n)

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