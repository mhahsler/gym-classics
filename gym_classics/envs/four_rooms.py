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