import numpy as np
from collections import defaultdict

# np.argmax does not break ties randomly
def random_argmax(x):
    return np.random.choice(np.where(x == np.max(x))[0])

def sample_episode(env, policy, start_state = None, start_action = None, max_len = 1000):
    episode = []
    
    s, r = env.reset()
    
    if not start_state is None:
        s = start_state
        env.state = env.decode(s)
    
    step = 0
    done = False
    while not done and step < max_len:
        a = policy[s]
        
        if step == 0 and not start_action is None:
            a = start_action
        
        sp, reward, done, _, _ = env.step(a)
        episode.append((s,a,reward,sp))
        
        s = sp
        step += 1
        
    return(episode)


def MC_control_ES(env, discount, n = 100, max_episode_len = 100):
    policy = random_policy(env)
    Q = np.zeros((len(env.states()), len(env.actions())))
    Returns = defaultdict(list) # a list for each (s,a)
    
    for i in range(n):
        # Sample starting (s,a)
        s = np.random.choice(env.states())
        a = np.random.choice(env.actions())
        
        episode = sample_episode(env, policy, start_state = s, start_action = a, max_len=max_episode_len)
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
          
    return policy, Q