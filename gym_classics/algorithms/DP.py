import numpy as np
from gym_classics.algorithms.policy import random_policy, random_argmax
from gym_classics.envs.abstract.base_env import BaseEnv as GymClassicsBaseEnv

def backup(env, discount, V, state, action):
    """Computes the Bellman backup for a given state and action.
    
    Args:
        env: A gym-classics environment with model access.
        discount: The discount factor.
        V: The current value function.
        state: The current state.
        action: The action to evaluate.
    Returns:
        The computed Q-value for the given state and action.
    """

    V = np.array(V)

    next_states, rewards, terminals, probs = env.model(state, action)
    bootstraps = (1.0 - terminals) * V[next_states]
    return np.sum(probs * (rewards + discount * bootstraps))

### Value Iteration

def value_iteration(env, discount, precision=1e-3, history = False, verbose = False):
    """Performs value iteration for the given environment.

    Args:
        env: A gym-classics environment with model access.
        discount: The discount factor (0 <= discount <= 1).
        precision: The precision for convergence (default: 1e-3).
        history: If True, returns a list of intermediate value functions.
        verbose: If True, prints progress information.

    Returns:
        The optimal value function V. If Vs is True, also returns a list of intermediate value functions.
    """
    
    assert isinstance(env, GymClassicsBaseEnv)
    assert 0.0 <= discount <= 1.0
    assert precision > 0.0
    
    V = np.zeros(env.observation_space.n, dtype=np.float64)  
    if history:
        V_list = []
        V_list.append(V.copy())

    sweeps = 0
    while True:
        if verbose:
            print('.', end = '')
            sweeps += 1
        
        V_old = V.copy()

        for s in env.states():
            Q_values = [backup(env, discount, V, s, a) for a in range(env.action_space.n)]
            V[s] = np.max(Q_values)

        if history:
            V_list.append(V.copy())

        if np.abs(V - V_old).max() <= precision:
            break

    if verbose:
        print(f'\nConverged after {sweeps} sweeps.')

    if history:
        return V_list 
    
    return V


### Policy Iteration

def policy_evaluation(env, discount, policy, precision=1e-3, max_backups=1000):
    
    assert isinstance(env, GymClassicsBaseEnv)
    assert 0.0 <= discount <= 1.0
    assert precision > 0.0
    
    V = np.zeros(policy.shape, dtype=np.float64)

    while True:
        V_old = V.copy()

        for s in env.states():
            V[s] = backup(env, discount, V, s, policy[s])

        if np.abs(V - V_old).max() <= precision or max_backups <= 0:
            break

        max_backups -= 1
    return V


def policy_improvement(env, discount, policy, V_policy, precision=1e-3):
    policy_old = policy.copy()
    V_old = V_policy.copy()

    for s in env.states():
        Q_values = [backup(env, discount, V_policy, s, a) for a in env.actions()]
        policy[s] = np.argmax(Q_values)
        V_policy[s] = max(Q_values)

    stable = np.logical_or(
        policy == policy_old,
        np.abs(V_policy - V_old).max() <= precision,
    ).all()

    return policy, stable

def policy_iteration(env, discount, precision=1e-3, max_backups=1000, history = False, verbose = False):
    """Performs policy iteration for the given environment.

    Args:
        env: A gym-classics environment with model access.
        discount: The discount factor (0 <= discount <= 1).
        precision: The precision for convergence (default: 1e-3).
        max_backups: Maximum number of iterations used in policy evaluation. Note: this prevents an infinite loop for policies that do not reach a terminal state.
        history: If True, returns a list of intermediate value functions.
        verbose: If True, prints progress information.

    Returns:
        The optimal value function V. If Vs is True, also returns a list of intermediate value functions.
    """
    
    assert isinstance(env, GymClassicsBaseEnv)
    assert 0.0 <= discount <= 1.0
    assert precision > 0.0

    policy = random_policy(env)

    if history:
        pol_list = []
        pol_list.append(policy.copy())
        V_list = []

    iterations = 0
    while True:
        if verbose:
            print('.', end = '')
            iterations += 1

        V_policy = policy_evaluation(env, discount, policy, precision, max_backups)
        if history:
            V_list.append(V_policy.copy())

        policy, stable = policy_improvement(env, discount, policy, V_policy, precision)
             
        if stable:
            break

        if history:
            pol_list.append(policy.copy())

    if verbose:
        print(f'\nConverged after {iterations} iterations.')
        
    if history:
        return pol_list, V_list

    return policy