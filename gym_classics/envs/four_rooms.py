from .abstract.gridworld import Gridworld

class FourRooms(Gridworld):
    """An 11x11 gridworld segmented into four rooms. The agent begins in the bottom-left
    cell; the goal is in the top-right cell.

    **reference:** cite{2} (page 192).

    **state**: Grid location.

    **actions**: Move up/right/down/left.

    **rewards**: +1 for episode termination.

    **termination**: Taking any action in the goal.
    """

    layout = """
|     X     |
|     X   G |
|           |
|     X     |
|     X     |
|X XXXX     |
|     XXX XX|
|     X     |
|     X     |
|           |
|S    X     |
"""

    def __init__(self, **args):
        super().__init__(FourRooms.layout, **args)

#   def _reward(self, state, action, next_state):
#       return 1.0 if self._done(state, action, next_state) else 0.0
#
#    def _done(self, state, action, next_state):
#        return state in self._goals
