# Utilities for gym_classic gridworld environments
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# for animation
from matplotlib import animation, rc
from IPython.display import HTML
#rc('animation', html='html5')
rc('animation', html='jshtml')

### Visualization function

def gridworld_animate(env, Vs, policies = None, interval = 1000, repeat=False, cmap = "coolwarm", clim = None, origin='lower'):
    """
    Create an animation showing the evolution of value functions in a gridworld.

    :param env: The gridworld environment.
    :param Vs: A list of value functions to animate.
    :param repeat: Whether the animation should repeat.
    :param cmap: Colormap to use for the value function.
    :param origin: 'lower' means (0,0) is at the bottom-left, 'upper' means (0,0) is at the top-left.
    :return: An animation object.
    """
     
    if clim is None:
        vmin = None
        vmax = None
    else:
        vmin = clim[0]
        vmax = clim[1]
     
    cmap = cm.get_cmap(cmap).copy() 
    cmap.set_bad(color='black')
    
    mazes = [env.to_matrix(V) for V in Vs]

    if not policies is None:
        #policies = [env.to_matrix([env.decode_action(a, type = "arrow") for a in policy]) for policy in policies]
        policies = [[env.decode_action(a, type = "arrow") for a in policy] for policy in policies]

    fig, ax = plt.subplots()

    # use last slide for clims 
    im = ax.imshow(mazes[-1], cmap=cmap, origin=origin, vmin = vmin, vmax=vmax)
    title = ax.set_title("")
    labels = []
    if not policies is None:
        for s in range(env.observation_space.n):
            (i,j) = env.decode(s)
            labels.append(ax.text(i, j, '', ha='center', va='center', color='black', fontsize=10))
    
    plt.colorbar(im, ax=ax)
    
    def init():
        ax.set_xticks(np.arange(mazes[0].shape[1]))
        ax.set_yticks(np.arange(mazes[0].shape[0]))

        num_rows, num_cols = mazes[0].shape
        ax.set_xticks(np.arange(-.5, num_cols, 1), minor=True)
        ax.set_yticks(np.arange(-.5, num_rows, 1), minor=True)
        ax.tick_params(which='minor', bottom=False, left=False)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
          
        return im, title, *labels,
    
    if not policies is None:
        for s in range(env.observation_space.n):
            (i,j) = env.decode(s)
            labels.append(ax.text(i, j, '', ha='center', va='center', color='black', fontsize=10))

    def step(i):
        im = ax.imshow(mazes[i], cmap=cmap, origin=origin, vmin = vmin, vmax=vmax)
        title.set_text(f'After Iteration {i}')
        
        if not policies is None:
            for pos, a in enumerate(policies[i]):
                labels[pos].set_text(a)
        
        return im, title, *labels,

    ani = animation.FuncAnimation(
        fig,
        step,
        frames = len(mazes),
        init_func = init,
        interval = interval,
        repeat = repeat,
        blit = True
    )

    plt.close()

    return ani
