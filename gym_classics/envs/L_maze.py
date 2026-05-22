from gym_classics.envs.abstract.gridworld import Gridworld

class LMazeGridworld(Gridworld):
    layout = """
|          |
|        G |
|          |
| XXXXXX   |
|      X   |
|      X   |
|      X   |
|   S  X   |
|          |
|          |
"""

    def __init__(self):
        super().__init__(self.layout)
        
    def _reward(self, state, action, next_state):      
        if next_state in self._goals: 
            return 1.0  
        
        return 0.0    
        
    def _done(self, state, action, next_state):
        return next_state in self._goals    
