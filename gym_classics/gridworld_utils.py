# Utilities for gym_classic gridworld environments

import numpy as np
import matplotlib.pyplot as plt

# for animation
from matplotlib import animation, rc
from IPython.display import HTML
rc('animation', html='html5')

### Visualization functions

def gridworld_to_matrix(env, value):
    """Converts a vector with values for states in a gridworld to a matrix for display. Values can be a value function, policy, etc.
    
    param env: The gridworld environment.
    param value: The value function as a vector.

    return: The value function as a matrix.
    """
    
    value = np.array(value)
    m = np.zeros(env.dims, dtype=value.dtype)

    for y in range(env.dims[1]):
        for x in range(env.dims[0]):
            state = (x, y)
            if env.is_reachable(state):
                s = env.encode(state)
                m[x,y] = value[s]
            else:
                pass

    return m.transpose()


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


def gridworld_image(env, V=None, labels=None, policy=None, title=None, cmap = 'bwr', origin='lower'):
    """
    Display the a gridworld as an image.
    
    :param env: The gridworld environment.
    :param value: The value (e.g., a value function) to display. If None, display state indices.
    :param labels: The labels to show on the grid cells in the same order as the value function. If True, show rounded values from V.
    :param policy: The policy to display. If not None, show the policy.
    :param title: Title of the plot.
    :param cmap: Colormap to use for the value function.
    :param origin: 'lower' means (0,0) is at the bottom-left, 'upper' means (0,0) is at the top-left.
    """
    
    if not V is None:
        m = gridworld_to_matrix(env, V)
    else:
        m = np.zeros(env.dims).transpose()
        labels = env.states()
        cmap = None

    if not policy is None:
        labels = convert_policy_to_symbols(env, policy)

    if isinstance(labels, bool) and labels:
            labels = np.round(V, 2)

    if not labels is None:
        labels = gridworld_to_matrix(env, labels)

    image(m, title=title, labels=labels, cmap=cmap, origin=origin)


def gridworld_image_list(env, Vs = None, policies = None, cmap = 'bwr', origin='lower'):
    n_states = len(env.states())
    if Vs is not None:
        iterations = len(Vs)
    else:
        iterations = len(policies)
    
    V = None
    policy = None

    for i in range(iterations):
        if not Vs is None:
            V = Vs[i]
        if not policies is None:
            policy = policies[i]    

        gridworld_image(env, V, policy=policy, title=f'After Sweep {i}', cmap=cmap, origin=origin)    


def convert_policy_to_symbols(env, policy):
    action_symbols = {
        0: '↑',
        1: '→',
        2: '↓',
        3: '←',
    }
    return [action_symbols.get(a, '?') for a in policy]


def gridworld_animation(env, Vs, interval = 1000, repeat=False, cmap = 'bwr', origin='lower'):
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