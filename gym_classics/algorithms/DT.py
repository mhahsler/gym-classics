import random

# np.argmax does not break ties randomly
def random_argmax(x):
    return np.random.choice(np.where(x == np.max(x))[0])


def Sarsa(env, discount, alpha, epsilon, n = 100, verbose = False, returns = False):
    if returns:
        Rs = [0] * n 
        
    Q = np.zeros((len(env.states()), len(env.actions())))
    
    for i in range(n):
        s, r = env.reset()
         
        if verbose:
            print ("--- Episode {i} ---")      
          
        if (random.random() > epsilon):
            a = random_argmax(Q[s,:])
        else:
            a = np.random.choice(env.actions())  
          
        t = 0
        done = False
        while not done:            
            sp, r, done, _, _ = env.step(a)
            Rs[i] += r * pow(discount, t)
            t += 1
            
        
            if (random.random() > epsilon):
                ap = random_argmax(Q[sp,:])
            else:
                ap = np.random.choice(env.actions()) 
        
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


def Q_learning(env, discount, alpha, epsilon, n = 100):
    Q = np.zeros((len(env.states()), len(env.actions())))
    
    for i in range(n):
        s, r = env.reset()
          
        done = False
        while not done:
            # epsilon-greedy choice w.r.t. Q 
            if (random.random() > epsilon):
                a = random_argmax(Q[s,:])
            else:
                a = np.random.choice(env.actions())
            
            sp, r, done, _, _ = env.step(a)
        
            Q[s,a] = Q[s,a] + alpha * (r + discount * np.max(Q[sp,:]) - Q[s,a])
            
            s = sp
          
    return Q