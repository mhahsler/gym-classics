# Utilities for gym_classic gridworld environments

import numpy as np
import matplotlib.pyplot as plt

# for animation
from matplotlib import animation, rc
from IPython.display import HTML
rc('animation', html='html5')

def gridworld_to_matrix(env, value):
    """Converts a vector with values for states in a gridworld to a matrix for display. Values can be a value function, policy, etc.
    
    param env: The gridworld environment.
    param value: The value function as a vector.

    return: The value function as a matrix.
    """
    m = np.zeros(env.dims, dtype=value.dtype)

    for y in range(env.dims[1]):
        for x in range(env.dims[0]):
            state = (x, y)
            if env.is_reachable(state):
                s = env.encode(state)
                m[x,y] = value[s]
            else:
                m[x,y] = np.nan

    return m.transpose()

### Visualization functions

def image(m, labels=None, title=None,  cmap = 'bwr', origin='lower'):
    
    row_labels = range(m.shape[0])
    col_labels = range(m.shape[1])
    
    fig, ax = plt.subplots()
    if not cmap is None:
        im = ax.imshow(m, cmap=cmap, origin=origin)
    else:
        im = ax.imshow(np.zeros(m.shape), cmap = 'Grays',  origin=origin)

    ax.set_xticks(np.arange(m.shape[1]))
    ax.set_yticks(np.arange(m.shape[0]))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    num_rows, num_cols = m.shape
    ax.set_xticks(np.arange(-.5, num_cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, num_rows, 1), minor=True)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    
    if not labels is None:
        for (j, i), label in np.ndenumerate(labels):
            ax.text(i, j, label, ha='center', va='center', color='black', fontsize=10)

    if not cmap is None:
        plt.colorbar(im, ax=ax)
    
    plt.title(title)
    plt.show()

def gridworld_value_image(env, V=None, labels=None, title=None, cmap = 'bwr', origin='lower'):
    """
    Display the a gridworld as an image.
    
    :param env: The gridworld environment.
    :param value: The value (e.g., a value function) to display. If None, display state indices.
    :param labels: The labels to show on the grid cells in the same order as the value function. If True, show rounded values from V.
    :param title: Title of the plot.
    :param cmap: Colormap to use for the value function.
    :param origin: 'lower' means (0,0) is at the bottom-left, 'upper' means (0,0) is at the top-left.
    """
    
    if not V is None:
        m = gridworld_to_matrix(env, V)
    else:
        m = np.zeros(env.dims).transpose()
        labels = np.array(list(env.states())).astype(str)
        cmap = None

    if isinstance(labels, bool) and labels:
            labels = np.round(V, 2)

    if not labels is None:
        labels = gridworld_to_matrix(env, labels)

    image(m, title=title, labels=labels, cmap=cmap, origin=origin)

def gridworld_values_image(env, Vs, cmap = 'bwr', origin='lower'):
    for i in range(len(Vs)):
        gridworld_value_image(env, Vs[i], title=f'After Sweep {i}', cmap=cmap, origin=origin)    

def gridworld_value_animation(env, Vs, interval = 1000, repeat=False, cmap = 'bwr', origin='lower'):
    """
    Create an animation showing the evolution of value functions in a gridworld.

    :param env: The gridworld environment.
    :param Vs: A list of value functions to animate.
    :param repeat: Whether the animation should repeat.
    :param cmap: Colormap to use for the value function.
    :param origin: 'lower' means (0,0) is at the bottom-left, 'upper' means (0,0) is at the top-left.
    :return: An animation object.
    """
    mazes = [gridworld_to_matrix(env, V) for V in Vs]

    fig, ax = plt.subplots()
    im = ax.imshow(mazes[0], cmap=cmap, origin=origin)

    ax.set_xticks(np.arange(mazes[0].shape[1]))
    ax.set_yticks(np.arange(mazes[0].shape[0]))

    num_rows, num_cols = mazes[0].shape
    ax.set_xticks(np.arange(-.5, num_cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, num_rows, 1), minor=True)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

    title = ax.set_title("")

    plt.colorbar(im, ax=ax)

    def step(i):
        im = ax.imshow(mazes[i], cmap=cmap, origin=origin)
        title.set_text(f'After Sweep {i}')
        return im, title,

    ani = animation.FuncAnimation(
        fig,
        step,
        frames = len(mazes),
        interval = interval,
        repeat = repeat
    )

    plt.close()

    return ani  




### DP solver functions

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

def value_iteration(env, discount, precision=1e-3, Vs = False, verbose = False):
    """Performs value iteration for the given environment.

    Args:
        env: The environment to perform value iteration on.
        discount: The discount factor (0 <= discount <= 1).
        precision: The precision for convergence (default: 1e-3).
        Vs: If True, returns a list of intermediate value functions.
        verbose: If True, prints progress information.

    Returns:
        The optimal value function V. If Vs is True, also returns a list of intermediate value functions.
    """
    
    assert 0.0 <= discount <= 1.0
    assert precision > 0.0
    
    V = np.zeros(env.observation_space.n, dtype=np.float64)  
    if Vs:
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
            V[s] = max(Q_values)

        if Vs:
            V_list.append(V.copy())

        if np.abs(V - V_old).max() <= precision:
            break

    if verbose:
        print(f'\nConverged after {sweeps} sweeps.')

    if Vs:
        return V_list 
    
    return V