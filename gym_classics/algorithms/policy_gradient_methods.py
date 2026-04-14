import numpy as np

from gym_classics.algorithms.linear_approximation import state_features, q_hat  
from gym_classics.envs.abstract.base_env import BaseEnv as GymClassicsBaseEnv

from tqdm import tqdm

def state_features(s):
    """
    Convert the state id into state features. This function needs to be overwritten for the environment
    
    :param s: state id
    :return a state feature vector
    """
    raise NotImplementedError("state_features function must be implemented and overwrite gym_classics.algorithms.linear_approximation.state_features.") 

def state_action_features(s,a,env):
    s = state_features(s,env)
    x = np.zeros(1+len(s)*env.action_space.n)
    x[active_weights(a, len(s)-1)] = s
    return x

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
    
    assert isinstance(env, GymClassicsBaseEnv), "env must be an instance of GymClassicsBaseEnv"
    assert alpha > 0 and alpha <= 1, "alpha must be in (0,1]"
    assert gamma >= 0 and gamma <= 1, "gamma must be in [0  ,1]"
    assert n > 0, "number of episodes must be positive"
    assert max_episode_length > 0, "max episode length must be positive"
    
    
    if theta is None:
        theta = np.zeros(1+len(state_features(0, env))*env.action_space.n)
    
    if history:
        returns = []        
        ep_lens = []
        thetas = []
        thetas.append(theta.copy())
        
    for episode in tqdm(range(n), desc="Episodes", disable=verbose):
        if verbose:
            print(f"Episode {episode+1}/{n}")
        
        episode_data = sample_episode_policy(env, pi, theta, max_episode_length)
        
        for t in range(len(episode_data)):
            #print(episode_data[t])
        
            G = np.sum([e[2] for e in episode_data[t:]] * (gamma ** np.arange(len(episode_data[t:]))))
            if history and t == 0:
                returns.append(G)   

            s,a,r,next_s = episode_data[t]
            
            # ln policy gradient= x(s,a)- sum_b pi(b|s,theta) x(s,b)
            grad_log_pi = state_action_features(s, a, env) - sum([pi(s, theta, env)[b] * state_action_features(s, b, env) for b in range(env.action_space.n)])
            
            if verbose: 
                print (f"t: {t}, G: {G:.2f}, grad_log_pi: {grad_log_pi}")
            
            theta += alpha * (gamma**t) * G * grad_log_pi
            
        if history:
            thetas.append(theta.copy())
            ep_lens.append(len(episode_data))
            
    if history:
        return theta, {'thetas': thetas, 'returns': returns, 'ep_lens': ep_lens}
       
    return theta