import random
import numpy as np
from gym_classics.algorithms.policy import random_policy, random_argmax
from tqdm import tqdm

def Sarsa(env, discount, alpha, epsilon, Q=None, n = 100, verbose = False, history = False):
        
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
            print ("--- Episode {i} ---")      
          
        if (random.random() > epsilon):
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
            
        
            if (random.random() > epsilon):
                ap = random_argmax(Q[sp,:])
            else:
                ap = np.random.choice(env.action_space.n) 
        
            Q[s,a] = Q[s,a] + alpha * (r + discount * Q[sp,ap] - Q[s,a])
            
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
            if (random.random() > epsilon):
                a = random_argmax(Q[s,:])
            else:
                a = np.random.choice(env.action_space.n)
            
            sp, r, done, _, _ = env.step(a)
        
            if history:
                G += r * pow(discount, t)
                t += 1
        
            Q[s,a] = Q[s,a] + alpha * (r + discount * np.max(Q[sp,:]) - Q[s,a])
            
            s = sp
        
        if history:
            Q_list.append(Q.copy())
            return_list.append(G)
            ep_len_list.append(t)
    
    if history:
        return Q, {'Qs': Q_list, 'returns': return_list, 'ep_lens': ep_len_list}
          
    return Q