from gym_classics.envs.abstract.base_env import BaseEnv

import numpy as np
import matplotlib.pyplot as plt

class Gridworld(BaseEnv):
    """Abstract class for creating gridworld-type environments."""

    def __init__(self, layout_string, n_actions=None):
        self.dims, starts, self._goals, self._blocks = parse_gridworld(
            layout_string)

        if n_actions is None:
            n_actions = 4
        super().__init__(starts, n_actions)

    def _next_state(self, state, action):
        next_state = self._move(state, action)
        if self._is_blocked(next_state):
            next_state = state
        return self._clamp(next_state), 1.0

    def _move(self, state, action):
        x, y = state
        return {
            0: (x,   y+1),  # Up
            1: (x+1, y),    # Right
            2: (x,   y-1),  # Down
            3: (x-1, y)     # Left
        }[action]

    def _clamp(self, state):
        """Clamps the state within the grid dimensions."""
        x, y = state
        x = max(0, min(x, self.dims[0] - 1))
        y = max(0, min(y, self.dims[1] - 1))
        return (x, y)

    def _is_blocked(self, state):
        """Returns True if this state cannot be occupied, False otherwise."""
        return state in self._blocks

    def _generate_transitions(self, state, action):
        yield self._deterministic_step(state, action)

    ### Addition to the interface
    
    ### TODO: move to base class?
    def encode_action(self, action_label):
        """Converts a action label into a numeric action ID."""
        action_ids = {"up": 0, "right": 1, "down": 2, "left": 3}
        id = action_ids.get(action_label, -1)
        return id


    def decode_action(self, action, type="text"):
        """Converts a numeric action ID into a label. Choices for type are 'text' and 'arrow'."""
        action = int(action)
        
        # empty has index 4 and is used to hide actions
        if (type == "arrow"):
             action_labels = ['↑', '→', '↓', '←', '']
        else: 
            action_labels =["up", "right", "down", "left", ""]
            
        return action_labels[action]

    def to_matrix(self, value = None):
        """Converts a vector with values for states in a gridworld to a matrix for display. Values can be a value function, policy, etc.
        
        param value: The value function as a vector.

        return: The value function as a matrix.
        """
        
        if value is None:
            value = list(self.states())
        
        value = np.array(value)
      
        if np.issubdtype(value.dtype, np.integer):
            m = np.full(self.dims, -1, dtype=value.dtype)
        elif np.issubdtype(value.dtype, np.str_):
            m = np.full(self.dims, "", dtype=value.dtype)
        elif np.issubdtype(value.dtype, np.floating):
            m = np.full(self.dims, np.nan, dtype=value.dtype)
        else:
            m = np.zeros(self.dims, dtype=value.dtype)

        for y in range(self.dims[1]):
            for x in range(self.dims[0]):
                state = (x, y)
                if self.is_reachable(state):
                    s = self.encode(state)
                    m[x,y] = value[s]
                else:
                    pass

        return m.transpose() 
    
    def image(self, V=None, policy=None, episode = None, labels=None, title=None, cmap = 'auto', origin='lower', clim = None):
        """
        Display the a gridworld as an image.
        
        :param value: The value (e.g., a value function) to display. If None, display state indices.
        :param labels: The labels to show on the grid cells in the same order as the value function. If True, show rounded values from V.
        :param policy: The policy to display. If not None, show the policy.
        :param episode: Show an episode
        :param title: Title of the plot.
        :param cmap: Colormap to use for the value function.
        :param origin: 'lower' means (0,0) is at the bottom-left, 'upper' means (0,0) is at the top-left.
        """
        
        colorbar = True
        
        if not V is None:
            m = self.to_matrix(V)
        else:
            m = np.zeros(self.dims).transpose()
            # missing positions have -1
            m[self.to_matrix(labels) == -1] = np.nan
            labels = self.states()
            colorbar = False

        if not episode is None:
            policy = np.full(self.observation_space.n, 4)
            for step in episode:
                policy[step[0]] = step[1]

        if not policy is None:
            labels = [self.decode_action(a, type = "arrow") for a in policy]

        if isinstance(labels, bool) and labels:
                labels = np.round(V, 2)

        if not labels is None:
            labels = self.to_matrix(labels)

        _image(m, title=title, labels=labels, cmap=cmap, clim = clim, origin=origin, colorbar=colorbar)  
        
        
    def image_list(self, Vs = None, policies = None, episodes = None, cmap = 'auto', clim = None, origin='lower'):
        """
        Creates a sequence of images, one for each episode.
        """
        n_states = len(self.states())
        if Vs is not None:
            iterations = len(Vs)
        elif policies is not None:
            iterations = len(policies)
        else:
            iterations = len(episodes)
        
        V = None
        policy = None
        episode = None

        for i in range(iterations):
            if not Vs is None:
                V = Vs[i]
            if not policies is None:
                policy = policies[i]
            if not episodes is None:
                episode = episodes[i]    

            self.image(V, policy=policy, episode=episode, title=f'After Iteration {i}', cmap=cmap, clim = clim, origin=origin)  

### helper functions
import matplotlib.cm as cm

def _image(m, labels=None, title=None, cmap = 'auto', clim = None, origin='lower', colorbar=True):
    
    if cmap == 'auto':      
        if (np.any(m < 0.0) and np.any(m > 0.0)) or (not clim is None and clim[0]<0 and clim[1]>0):
            cmap = "coolwarm"
            cmap = cm.get_cmap(cmap).copy()
            cmap.set_bad(color='black')
        else:
            cmap = "Reds"
            cmap = cm.get_cmap(cmap).copy() 
            cmap.set_bad(color='black')
    
    row_labels = range(m.shape[0])
    col_labels = range(m.shape[1])
    
    fig, ax = plt.subplots()
    if not clim is None:
        im = ax.imshow(m, cmap=cmap, origin=origin, vmin = clim[0], vmax=clim[1])
    else:
        im = ax.imshow(m, cmap=cmap, origin=origin)


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

    if colorbar:
        plt.colorbar(im, ax=ax)
    
    plt.title(title)
    plt.show()


def parse_gridworld(layout_string):
    """Parse the layout string"""
    layout_string = layout_string.replace(
        '|', '')  # Remove optional pipe characters
    lines = layout_string.split('\n')
    lines = [l for l in lines if l != '']  # Remove empty lines

    # Get dimensions: assume rectangular (width, height)
    H = len(lines)
    W = len(lines[0])
    for l in lines:
        assert len(l) == W, "layout string is not rectangular; check dimensions"
    dims = (W, H)

    starts = set()
    goals = set()
    blocks = set()

    for row in range(H):
        for col in range(W):
            # Makes (0,0) the bottom-left cell in the gridworld
            coords = (col, H - 1 - row)
            char = lines[row][col]

            if char == 'S':  # Start (may be more than one)
                starts.add(coords)
            elif char == 'G':  # Goal (may be more than one)
                goals.add(coords)
            elif char == 'X':  # Block (agent cannot occupy these cells)
                blocks.add(coords)
            elif char == ' ':  # Empty (agent can occupy these cells)
                pass
            else:
                raise ValueError(f"invalid character '{char}' at {coords}")

    return dims, frozenset(starts), frozenset(goals), frozenset(blocks)
