from cProfile import label

import pygame
from gym_classics.envs.abstract.base_env import BaseEnv
from gymnasium.spaces import MultiDiscrete

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

# we need to overwrite:
# * _next_state
# * _reward
# * _done
# * _generate_transitions

class Gridworld(BaseEnv):
    """Abstract class for creating gridworld-type environments."""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, layout_string, action_labels = ["up", "right", "down", "left"], tabular = True, render_mode=None):
        """Initializes the gridworld environment from a layout string. The layout string should be a rectangular grid of characters, where each character represents a type of cell:
        - 'S': Start (may be more than one)
        - 'G': Goal (may be more than one)
        - 'X': Block (agent cannot occupy these cells)
        - ' ': Empty (agent can occupy these cells)
        - All other characters are treated as empty cells that the agent can occupy.
        
        param layout_string: The string representation of the gridworld layout.
        param action_labels: The labels for the actions. Defaults to ["up", "right", "down", "left"]. You can specify additional labels for extra actions.
        param tabular: If True, the environment will use a tabular state representation (i.e., states are represented as integer IDs). 
            If False, states will be represented as their (x,y) coordinates. Defaults to True.
        """
        
        self.dims, starts, self._goals, self._blocks, self._extra_labels = parse_gridworld(layout_string)
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.PyGame_window_size = 512  # The size of the PyGame window in pixels
        self.PyGame_window = None
        self.PyGame_clock = None

        super().__init__(starts, action_labels = action_labels, tabular = tabular, reachable_states = None)
        
        if not tabular:
            self.observation_space = MultiDiscrete(self.dims)

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
        
    def _reward(self, state, action, next_state):      
        if next_state in self._goals: 
            return 1.0  
        return 0.0
    
    def _done(self, state, action, next_state):
        return state in self._goals
    
    def step(self, action):
        next_state, reward, done, x, info = super().step(action)
        
        if self.render_mode == "human":
            self._render_frame()
        
        return next_state, reward, done, x, info

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, info
    
    def close(self):
        if self.PyGame_window is not None:
            pygame.display.quit()
            pygame.quit()
            self.PyGame_window = None
            self.PyGame_clock = None
        
        super().close()
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        pix_square_size = self.PyGame_window_size / max(self.dims)
        display_size = np.array(pix_square_size) * self.dims
        
        if self.PyGame_window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.PyGame_window = pygame.display.set_mode(display_size)
        if self.PyGame_clock is None and self.render_mode == "human":
            self.PyGame_clock = pygame.time.Clock()

        canvas = pygame.Surface(display_size)
        canvas.fill((255, 255, 255))

        # draw the goal
        for g in self._goals:
            pos = np.array(g)
            pos[1] = self.dims[1] - pos[1] - 1
            pygame.draw.rect(
                canvas,
                (0, 128, 0),
                pygame.Rect(
                    pix_square_size * pos,
                    (pix_square_size, pix_square_size),
                ),
            )
     
        # unreachable squares
        for g in np.argwhere(self.to_matrix() == -1):
            pos = np.array(g)
            pos = np.flip(pos)
            pos[1] = self.dims[1] - pos[1] - 1
            pygame.draw.rect(
                canvas,
                (72, 72, 72),
                pygame.Rect(
                    pix_square_size * pos,
                    (pix_square_size, pix_square_size),
                ),
            )
        
        # draw the agent
        pos = np.array(self.state)
        pos[1] = self.dims[1] - pos[1] - 1
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (pos + .5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.dims[1] + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (display_size[0], pix_square_size * x),
                width=3,
            )
            
        for x in range(self.dims[0] + 1):
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, display_size[1]),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.PyGame_window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.PyGame_clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    ### Addition to the interface
    
    ## overwrite so we have arrows
    def id2action(self, action, type="text"):
        """Converts a numeric action ID into a label. Choices for type are 'text' and 'arrow'."""
        action = int(action)
        
        action_labels = self.action_labels + [""]  # Add empty label for hidden actions
       
        # empty has index 4 and is used to hide actions
        if (type == "arrow"):
             action_labels[0:4] = ['↑', '→', '↓', '←']
            
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
                    m[x,y] = value[self.state2id(state)]
                else:
                    pass

        return m.transpose() 
    
    def print(self, array, decimals=2, separator=' ' * 2, signed=True, transpose=False):
        """Prints a gridworld array in a human-readable format. The array should be a vector with values for states in the gridworld, 
        such as a value function or policy."""
        
        def formatter(x):
            string = '{:' + ('+' if signed else '') + '.' + str(decimals) + 'f}'
            return string.format(x)
        maxlen = max([len(formatter(x)) for x in array])

        # Now we can actually print the values
        for y in reversed(range(self.dims[1])):
            for x in range(self.dims[0]):
                state = (x, y) if not transpose else (y, x)
                if self.is_reachable(state):
                    s = self.state2id(state)
                    print(formatter(array[s]).rjust(maxlen), end=separator)
                else:
                    print(' ' * maxlen, end=separator)
            print()
    
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
            # start with policy that hides all actions with an index past the last action.
            policy = np.full(len(self.states()), len(self.actions()))
            for step in episode:
                policy[step[0]] = step[1]

        if not policy is None:
            labels = [self.id2action(a, type = "arrow") for a in policy]

        if isinstance(labels, bool) and labels:
                labels = np.round(V, 2)

        if not labels is None:
            labels = self.to_matrix(labels)
            
        extra = [""] * self.dims[0] * self.dims[1]
        for s in self._starts:
            extra[self.state2id(s)] = "S"
        for s in self._goals:
            extra[self.state2id(s)] = "G"
        for s, label in self._extra_labels:
            extra[self.state2id(s)] = label
        extra = self.to_matrix(extra)
        
        _image(m, title=title, labels=labels, extra=extra, cmap=cmap, clim = clim, origin=origin, colorbar=colorbar)  
        
        
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



def _image(m, labels=None, extra = None, title=None, cmap = 'auto', clim = None, origin='lower', colorbar=True):
    
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

    
    num_rows, num_cols = m.shape

    if not clim is None:
        im = ax.imshow(m, cmap=cmap, interpolation="nearest", origin=origin, vmin = clim[0], vmax=clim[1])
    else:
        im = ax.imshow(m, cmap=cmap, interpolation="nearest", origin=origin)
    
    ax.set_aspect("equal")
    
    # major tickmarks at the center of each cell for labels
    ax.set_xticks(np.arange(m.shape[1]))
    ax.set_yticks(np.arange(m.shape[0]))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # minor tickmarks at the edges of the cells for gridlines
    ax.set_xticks(np.arange(-.5, num_cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, num_rows, 1), minor=True)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=.5)
    
    if not extra is None:
        for (j, i), label in np.ndenumerate(extra):
            ax.text(i, j, label, ha='center', va='center', color='grey', fontsize=15, fontweight='bold')
    
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

    starts = list() # so we can sample from it
    
    # all others are hashed into sets for fast lookup
    goals = set()
    blocks = set()
    extra_labels = set()

    for row in range(H):
        for col in range(W):
            # Makes (0,0) the bottom-left cell in the gridworld
            coords = (col, H - 1 - row)
            char = lines[row][col]

            if char == 'S':  # Start (may be more than one)
                starts.append(coords)
            elif char == 'G':  # Goal (may be more than one)
                goals.add(coords)
            elif char == 'X':  # Block (agent cannot occupy these cells)
                blocks.add(coords)
            elif char == ' ':  # Empty (agent can occupy these cells)
                pass
            else:
                extra_labels.add((coords, char))
            #    raise ValueError(f"invalid character '{char}' at {coords}")

    return dims, starts, goals, blocks, extra_labels
