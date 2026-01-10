# Utilities for gym_classic gridworld environments

import numpy as np
import matplotlib.pyplot as plt

# for animation
from matplotlib import animation, rc
from IPython.display import HTML
rc('animation', html='html5')

### Visualization function

def gridworld_animate(env, Vs, policies = None, interval = 1000, repeat=False, cmap = 'bwr', origin='lower'):
    """
    Create an animation showing the evolution of value functions in a gridworld.

    :param env: The gridworld environment.
    :param Vs: A list of value functions to animate.
    :param repeat: Whether the animation should repeat.
    :param cmap: Colormap to use for the value function.
    :param origin: 'lower' means (0,0) is at the bottom-left, 'upper' means (0,0) is at the top-left.
    :return: An animation object.
    """
    mazes = [env.to_matrix(V) for V in Vs]

    if not policies is None:
        policies = [env.to_matrix([env.decode_action(a, type = "arrow") for a in policy]) for policy in policies]

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
    labels = None

    plt.colorbar(im, ax=ax)

    def step(i):
        im = ax.imshow(mazes[i], cmap=cmap, origin=origin)
        title.set_text(f'After Iteration {i}')
        
        if not policies is None:
            labels = policies[i]
            for text_artist in ax.texts:
                text_artist.remove()
            for (j, i), label in np.ndenumerate(labels):
                ax.text(i, j, label, ha='center', va='center', color='black', fontsize=10)
    
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
