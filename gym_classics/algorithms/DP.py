import numpy as np

def backup(env, discount, V, state, action):
    """Computes the Bellman backup for a given state and action.
    
    Args:
        env: The environment.
        discount: The discount factor.
        V: The current value function.
        state: The current state.
        action: The action to evaluate.
    Returns:
        The computed Q-value for the given state and action.
    """

    next_states, rewards, terminals, probs = env.model(state, action)
    bootstraps = (1.0 - terminals) * V[next_states]
    return np.sum(probs * (rewards + discount * bootstraps))

### Value Iteration

def value_iteration(env, discount, precision=1e-3, history = False, verbose = False):
    """Performs value iteration for the given environment.

    Args:
        env: The environment to perform value iteration on.
        discount: The discount factor (0 <= discount <= 1).
        precision: The precision for convergence (default: 1e-3).
        history: If True, returns a list of intermediate value functions.
        verbose: If True, prints progress information.

    Returns:
        The optimal value function V. If Vs is True, also returns a list of intermediate value functions.
    """
    
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
            Q_values = [backup(env, discount, V, s, a) for a in env.actions()]
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

