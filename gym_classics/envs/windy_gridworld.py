from gym_classics.envs.abstract.gridworld import Gridworld


class WindyGridworld(Gridworld):
    """A 10x7 deterministic gridworld where some columns are affected by an upward wind.
    The agent starts in cell (0, 3) and the goal is at cell (7, 3). If an agent executes
    an action from a cell with wind, the resulting position is given by the vector sum
    of the action's effect and the wind.

    **reference:** cite{3} (page 130, example 6.5).

    **state:** Grid location.

    **actions:** Move up/right/down/left.

    **rewards:** -1 for all transitions unless the episode terminates.

    **termination:** Reaching the goal.
    """

    layout = """
|          |
|          |
|          |
|S      G  |
|          |
|          |
|          |
"""

    def __init__(self, **args):
        super().__init__(WindyGridworld.layout, **args)

    def _next_state(self, state, action):
        wind_strength = self._wind_strength(state)
        state, _ = super()._next_state(state, action)
        state = self._apply_wind(state, wind_strength)
        return self._clamp(state), 1.0

    def _apply_wind(self, state, strength):
        x, y = state
        return (x, y + strength)

    def _wind_strength(self, state):
        """Returns wind strength in the given state."""
        x, _ = state
        if x in {3, 4, 5, 8}:
            return 1
        elif x in {6, 7}:
            return 2
        else:
            return 0