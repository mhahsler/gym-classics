import numpy as np
from itertools import product

from gym_classics.algorithms.linear_approximation import state_features, q_hat  
from gym_classics.algorithms.schedules import Schedule, ConstantSchedule
from gym_classics.envs.abstract.base_env import BaseEnv as GymClassicsBaseEnv

from tqdm import tqdm

def state_features(s, env):
    """
    Convert the state id into state features. This function needs to be overwritten for the environment
    
    :param s: state id
    :param env: environment instance

    :return a state feature vector
    """
    raise NotImplementedError("state_features function must be implemented and overwrite gym_classics.algorithms.linear_approximation.state_features.") 

def active_weights(a, sf_len):
    """helper for q_hat()"""
    return [0] + list(range(a*sf_len+1, a*sf_len+sf_len+1))

def state_action_features(s,a,env):
    s = state_features(s,env)
    x = np.zeros(1+len(s)*env.action_space.n)
    x[active_weights(a, len(s)-1)] = s
    return x

def semi_gradient_Sarsa_lambda(
    env,
    n,
    epsilon,
    alpha,
    gamma,
    lam,
    w=None,
    max_episode_length=1000,
    verbose=False,
    history=False
):
    """
    Semi-gradient SARSA(lambda): on-policy control with linear function approximation
    and eligibility traces.

    Parameters
    ----------
    env : GymClassicsBaseEnv
        Episodic environment used to generate experience.
    n : int
        Number of episodes.
    epsilon : float
        Exploration rate for epsilon-greedy policy.
    alpha : float
        Step size.
    gamma : float
        Discount factor.
    lam : float
        Trace-decay parameter lambda in [0, 1].
    w : array-like or None
        Initial weights. If None, initializes to zeros.
    max_episode_length : int
        Maximum number of steps per episode.
    verbose : bool
        Whether to print step-by-step diagnostics.

    Returns
    -------
    w : np.ndarray
        Learned weight vector.
    """

    assert isinstance(env, GymClassicsBaseEnv), "env must be an instance of GymClassicsBaseEnv"
    assert gamma >= 0 and gamma <= 1, "gamma must be in [0,1]"
    assert lam >= 0 and lam <= 1, "lambda must be in [0,1]"
    assert n > 0, "number of episodes must be positive"
    assert max_episode_length > 0, "max episode length must be positive"

    if not isinstance(alpha, Schedule):
        alpha = ConstantSchedule(alpha)
    if not isinstance(epsilon, Schedule):
        epsilon = ConstantSchedule(epsilon)

    if w is None:
        w = np.zeros(1 + state_features(0, env).shape[0] * env.action_space.n)

    if history:
        ws = []
        ws.append(w.copy())
        returns = []
        ep_lens = []

    def epsilon_greedy_action(state, epsilon):
        if np.random.rand() < epsilon:
            return env.action_space.sample()
        q_values = [q_hat(state, a, w, env) for a in range(env.action_space.n)]
        return np.argmax(q_values)

    for episode in tqdm(range(n), desc="Semi-Gradient SARSA(lambda)", disable=verbose):
        state, _ = env.reset()
        action = epsilon_greedy_action(state, epsilon(episode))

        # eligibility trace vector, same size as w
        z = np.zeros_like(w)
        Q_old = 0

        done = False
        i = 0
        
        G = 0  # for tracking returns if history is enabled

        while not done and i < max_episode_length:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            G += reward * (gamma ** i)  # accumulate return if history is enabled

            # current feature vector for (state, action)
            x = state_action_features(state, action, env)

            # update trace
            z = gamma * lam * z + (1 - alpha(episode) * gamma * lam * np.dot(z, x)) * x

            if terminated:
                delta = reward - q_hat(state, action, w, env)
            else:
                next_action = epsilon_greedy_action(next_state, epsilon(episode))
                delta = reward + gamma * q_hat(next_state, next_action, w, env) - q_hat(state, action, w, env)

            # semi-gradient weight update
            Q = q_hat(state, action, w, env)
            Q_prime = q_hat(next_state, next_action, w, env) if not terminated else 0            
            w += alpha(episode) * (delta + Q - Q_old) * z - alpha(episode) * (Q - Q_old) * x

            Q_old = Q_prime

            if verbose:
                if terminated:
                    print(
                        f"Episode {episode+1}, Step {i+1}: "
                        f"S={state}, A={action}, R={reward}, S'={next_state}, "
                        f"delta={delta}, z={z}, w={w}"
                    )
                else:
                    print(
                        f"Episode {episode+1}, Step {i+1}: "
                        f"S={state}, A={action}, R={reward}, S'={next_state}, A'={next_action}, "
                        f"delta={delta}, z={z}, w={w}"
                    )

            if done:
                break

            state = next_state
            action = next_action
            i += 1

        if history:
            returns.append(G)
            ws.append(w.copy())
            ep_lens.append(i)
            
        
    if history:        
        return w, {'ws': ws, 'returns': returns, 'ep_lens': ep_lens}
        
    return w