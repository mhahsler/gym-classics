import numpy as np
from itertools import product

from gym_classics.algorithms.policy import random_policy, random_argmax
from gym_classics.envs.abstract.base_env import BaseEnv as GymClassicsBaseEnv

def state_features(s):
    raise NotImplementedError("stat_features function must be implemented and overwrite gym_classics.algorithms.linear_approximation.state_features.") 

def v_hat(s, w):
    return np.dot(w, state_features(s))

def semi_gradient_TD0_estimation(env, policy, n, alpha, gamma, max_episode_length=1000, verbose = True):
    assert isinstance(env, GymClassicsBaseEnv), "env must be an instance of gym.Env"
    assert alpha > 0 and alpha <= 1, "Alpha must be in (0,1]"
    assert gamma >= 0 and gamma <= 1, "Gamma must be in [0,1]"
    assert n > 0, "Number of episodes must be positive"
    assert max_episode_length > 0, "Max episode length must be positive"
    
    w = np.zeros(state_features(0).shape[0])  # Initialize weights (intercept + x and y)

    for episode in range(n):
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

def active_weights(a, sf_len):
    return [0] + list(range(a*sf_len+1, a*sf_len+sf_len+1))

def q_hat(s, a, w): 
    sf_len = state_features(0).shape[0]-1
    active_weights = lambda a: [0] + list(range(a*sf_len+1, a*sf_len+sf_len+1))
    return np.dot(w[active_weights(a)], state_features(s))

def semi_gradient_Sarsa(env, n, epsilon, alpha, gamma, w = None, max_episode_length=1000, verbose = True):
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

    # helper used later
    def epsilon_greedy_action(state, epsilon):
        if np.random.rand() < epsilon:
            return env.action_space.sample()
        else:
            q_values = [q_hat(state, a, w) for a in range(env.action_space.n)]
            return np.argmax(q_values)
    
    for episode in range(n):
        state, _ = env.reset()
        action = epsilon_greedy_action(state, epsilon)
        done = False

        i = 0
        while not done and i < max_episode_length:
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