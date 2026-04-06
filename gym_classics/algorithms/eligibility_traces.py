import numpy as np
from itertools import product

from gym_classics.algorithms.linear_approximation import state_features, q_hat  
from gym_classics.envs.abstract.base_env import BaseEnv as GymClassicsBaseEnv


def semi_gradient_Sarsa_lambda(
    env,
    n,
    epsilon,
    alpha,
    gamma,
    lam,
    w=None,
    max_episode_length=1000,
    verbose=True
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
    assert alpha > 0 and alpha <= 1, "alpha must be in (0,1]"
    assert gamma >= 0 and gamma <= 1, "gamma must be in [0,1]"
    assert lam >= 0 and lam <= 1, "lambda must be in [0,1]"
    assert epsilon >= 0 and epsilon <= 1, "epsilon must be in [0,1]"
    assert n > 0, "number of episodes must be positive"
    assert max_episode_length > 0, "max episode length must be positive"

    sf = state_features(0)
    sf_len = sf.shape[0] - 1
    active_weights = lambda a: [0] + list(range(a * sf_len + 1, a * sf_len + sf_len + 1))

    if w is None:
        w = np.zeros(1 + state_features(0).shape[0] * env.action_space.n)

    def epsilon_greedy_action(state, epsilon):
        if np.random.rand() < epsilon:
            return env.action_space.sample()
        q_values = [q_hat(state, a, w) for a in range(env.action_space.n)]
        return np.argmax(q_values)

    for episode in range(n):
        state, _ = env.reset()
        action = epsilon_greedy_action(state, epsilon)

        # eligibility trace vector, same size as w
        z = np.zeros_like(w)
        Q_old = 0

        done = False
        i = 0

        while not done and i < max_episode_length:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # current feature vector for (state, action)
            x = np.zeros_like(w)
            x[active_weights(action)] = state_features(state)

            # update trace
            z = gamma * lam * z + (1 - alpha * gamma * lam * np.dot(z, x)) * x

            if terminated:
                delta = reward - q_hat(state, action, w)
            else:
                next_action = epsilon_greedy_action(next_state, epsilon)
                delta = reward + gamma * q_hat(next_state, next_action, w) - q_hat(state, action, w)

            # semi-gradient weight update
            Q = q_hat(state, action, w)
            Q_prime = q_hat(next_state, next_action, w) if not terminated else 0            
            w += alpha * (delta + Q - Q_old) * z - alpha * (Q - Q_old) * x

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

    return w