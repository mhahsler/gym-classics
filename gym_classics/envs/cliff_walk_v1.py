from gym_classics.envs.abstract.gridworld import Gridworld


class CliffWalk(Gridworld):
    """The Cliff Walking task, a 12x4 gridworld often used to contrast Sarsa with
    Q-Learning. The agent begins in the bottom-left cell and must navigate to the goal
    (bottom-right cell) without entering the region along the bottom ("The Cliff").
    
    v1 follows the textbook and does not end episodes when the cliff is reached. Also, the goal is 
    a real state.

    **reference:** cite{3} (page 132, example 6.6).

    **state**: Grid location.

    **actions**: Move up/right/down/left.

    **rewards**: -100 for entering The Cliff. -1 for all other transitions.

    **termination**: reaching the goal.
    """

    layout = """
|            |
|            |
|            |
|S          G|
"""

    def __init__(self):
        self._cliff = frozenset((x, 0) for x in range(1, 11))
        super().__init__(CliffWalk.layout)
    
    # cliff is unreachable. Leads to the start state    
    def _next_state(self, state, action):
        state, _ = super()._next_state(state, action)
        if (state in self._cliff):
            state = self._starts[0]
        return state, 1.0

    def _reward(self, state, action, next_state):
        if state in self._goals: 
            return 0.0
        
        n_state, _ = super()._next_state(state, action)
        if n_state in self._cliff: 
            return -100.0
         
        return -1.0

    def _done(self, state, action, next_state):
        return state in self._goals
