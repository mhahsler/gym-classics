"""This file implements temporal difference learning algorithms for policy evaluation and control in gym-classics    
environments with discrete state spaces. The algorithms include Sarsa(0) and Q-learning, which are fundamental 
methods in reinforcement learning for learning value functions and optimal policies from experience without requiring a 
model of the environment.
"""

import random
import numpy as np
from gym_classics.algorithms.policy import random_policy, random_argmax
from gym_classics.algorithms.schedules import Schedule, ConstantSchedule
from tqdm import tqdm
import gymnasium as gym

def Sarsa_0(env, discount, alpha, epsilon, Q=None, n = 100, verbose = False, history = False):
    assert isinstance(env.observation_space, gym.spaces.Discrete), "Tabular methods require discrete state space."  
        
    if not isinstance(alpha, Schedule):
        alpha = ConstantSchedule(alpha)
    if not isinstance(epsilon, Schedule):
        epsilon = ConstantSchedule(epsilon)

    if Q is None:
        Q = np.zeros((env.observation_space.n, env.action_space.n))
    
    if history:
        Q_list = []
        Q_list.append(Q.copy())
        return_list = []
        ep_len_list = []
        
    
    for i in tqdm(range(n), desc="Sarsa", disable=verbose):
        s, r = env.reset()
         
        if verbose:
            print(f"--- Episode {i} ---")      
          
        if (random.random() > epsilon(i)):
            a = random_argmax(Q[s,:])
        else:
            a = np.random.choice(env.action_space.n)  
          
        t = 0
        done = False
        G = 0
        while not done:            
            sp, r, done, _, _ = env.step(a)
            G += r * pow(discount, t)
            t += 1
            
        
            if (random.random() > epsilon(i)):
                ap = random_argmax(Q[sp,:])
            else:
                ap = np.random.choice(env.action_space.n) 
        
            Q[s,a] = Q[s,a] + alpha(i) * (r + discount * Q[sp,ap] - Q[s,a])
            
            if verbose:
                print(f"{t} - sarsa: {s},{a},{r},{sp},{ap}, - new Q(s,a): {Q[s,a]}")
                if done:
                    print("Total return:", G)
            
            s = sp
            a = ap
        
        if history:
            Q_list.append(Q.copy())
            return_list.append(G)   
            ep_len_list.append(t)
    
    if history:
        return Q, {'Qs': Q_list, 'returns': return_list, 'ep_lens': ep_len_list}
          
    return Q


def Q_learning(env, discount, alpha, epsilon, Q=None, n = 100, verbose = False, history = False):
    assert isinstance(env.observation_space, gym.spaces.Discrete), "Tabular methods require discrete state space."  
    
    if not isinstance(alpha, Schedule):
        alpha = ConstantSchedule(alpha)
    if not isinstance(epsilon, Schedule):
        epsilon = ConstantSchedule(epsilon)

    if Q is None:
        Q = np.zeros((env.observation_space.n, env.action_space.n))
    
    if history:
        Q_list = []
        Q_list.append(Q.copy())
        return_list = []
        ep_len_list = []

    for i in tqdm(range(n), desc="Q-Learning", disable=verbose):
        s, r = env.reset()
          
        done = False
        G = 0
        t = 0   
        while not done:
            # epsilon-greedy choice w.r.t. Q 
            if (random.random() > epsilon(i)):
                a = random_argmax(Q[s,:])
            else:
                a = np.random.choice(env.action_space.n)
            
            sp, r, done, _, _ = env.step(a)
        
            if history:
                G += r * pow(discount, t)
                t += 1
        
            Q[s,a] = Q[s,a] + alpha(i) * (r + discount * np.max(Q[sp,:]) - Q[s,a])
            
            s = sp
        
        if history:
            Q_list.append(Q.copy())
            return_list.append(G)
            ep_len_list.append(t)
    
    if history:
        return Q, {'Qs': Q_list, 'returns': return_list, 'ep_lens': ep_len_list}
          
    return Q