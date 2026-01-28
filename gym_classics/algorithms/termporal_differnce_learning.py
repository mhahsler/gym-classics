import random
import numpy as np
from gym_classics.algorithms.policy import random_policy, random_argmax

def Sarsa(env, discount, alpha, epsilon, Q=None, n = 100, verbose = False, returns = False):
    if returns:
        Rs = [0] * n 
        
    if Q is None:
        Q = np.zeros((env.observation_space.n, env.action_space.n))
    
    for i in range(n):
        s, r = env.reset()
         
        if verbose:
            print ("--- Episode {i} ---")      
          
        if (random.random() > epsilon):
            a = random_argmax(Q[s,:])
        else:
            a = np.random.choice(env.action_space.n)  
          
        t = 0
        done = False
        while not done:            
            sp, r, done, _, _ = env.step(a)
            Rs[i] += r * pow(discount, t)
            t += 1
            
        
            if (random.random() > epsilon):
                ap = random_argmax(Q[sp,:])
            else:
                ap = np.random.choice(env.action_space.n) 
        
            Q[s,a] = Q[s,a] + alpha * (r + discount * Q[sp,ap] - Q[s,a])
            
            if verbose:
                print(f"{t} - sarsa: {s},{a},{r},{sp},{ap}, - new Q(s,a): {Q[s,a]}")
                if done:
                    print("Total return:", Rs[i])
            
            s = sp
            a = ap
          
    if returns:
        return Q, Rs
          
    return Q


def Q_learning(env, discount, alpha, epsilon, Q=None, n = 100):
    if Q is None:
        Q = np.zeros((env.observation_space.n, env.action_space.n))
    
    for i in range(n):
        s, r = env.reset()
          
        done = False
        while not done:
            # epsilon-greedy choice w.r.t. Q 
            if (random.random() > epsilon):
                a = random_argmax(Q[s,:])
            else:
                a = np.random.choice(env.action_space.n)
            
            sp, r, done, _, _ = env.step(a)
        
            Q[s,a] = Q[s,a] + alpha * (r + discount * np.max(Q[sp,:]) - Q[s,a])
            
            s = sp
          
    return Q