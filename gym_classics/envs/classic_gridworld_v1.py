import numpy as np
from gym_classics.envs.abstract.gridworld import Gridworld


class ClassicGridworld(Gridworld):
    """A 4x3 pedagogical gridworld. The agent starts in the bottom-left cell. Actions
    are noisy; with a 10% chance each, a move action may be rotated by 90 degrees
    clockwise or counter-clockwise (the "80-10-10 rule"). Cell (1, 1) is blocked and
    cannot be occupied by the agent.

    **reference:** cite{1} (page 646).

    **state**: Grid location.

    **actions**: Move up/right/down/left.

    **rewards**: +1 for taking any action in cell (3, 2). -1 for taking any
    action in cell (3, 1). *NOTE:*  v1 uses the original -0.04 penalty for each state.

    **termination**: Earning a nonzero reward.
    """

    layout = """
|   G|
| X G|
|S   |
"""

    def __init__(self):
        super().__init__(ClassicGridworld.layout)

    def _reward(self, state, action, next_state):
        if state in self._goals:
            return 0.0
        return {(3, 1): -1.0, (3, 2): 1.0}.get(next_state, -0.04)

    def _done(self, state, action, next_state):
        return next_state in self._goals
    
    # Implement the non-deterministic actions
    
    # This method is called in the step function to create random events.
    # Here, the random event is that the environment executes a different
    # noisy action instead of the action the agent asked for.  
    # We return actually executed action as a list of random elements.
    def _sample_random_elements(self, state, action):
        noisy_action = (action + np.random.choice([-1, 0, 1], p=[.1,.8,.1])) % self.action_space.n
        return [noisy_action]

    # Returns an iterator for all possible outcomes. The random element is that
    # we have a noisy action, that may not be the intended action. 
    def _generate_transitions(self, state, action):
        # goal state is absorbing
        if state in self._goals:
            yield state, 0, True, 1.0

        else:
            for i in [-1, 0, 1]:
                noisy_action = (action + i) % self.action_space.n
                yield self._deterministic_step(state, action, noisy_action)

    # execute the noisy action and the probability
    def _next_state(self, state, action, noisy_action):
        next_state, _ = super()._next_state(state, noisy_action)
        p = 0.8 if action == noisy_action else 0.1
        return next_state, p
