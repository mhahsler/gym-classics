from abc import ABCMeta, abstractmethod
import numpy as np
import gym_classics
import random

from gymnasium import Env
from gymnasium.spaces import Discrete


class BaseEnv(Env, metaclass=ABCMeta):
    """Abstract base class for shared functionality between all environments."""

    # needs to provide the following Gymnasium Env functions:
    # - reset
    # - step
    # - render
    # - close

    # added for gym_classics
    # - action_space
    # - observation_space
    # - start states
    # - transitions
    # - rewards
    # - done conditions
    # - encode
    # - decode
    # - is_reachable
    # - actions
    # - model
    # - generate_transitions

    def __init__(self, starts, action_labels = None, tabular = True, reachable_states=None):   
        assert action_labels is not None, "action_labels must be provided"
        
        self.action_labels = action_labels
        
        n_actions = len(action_labels)
        self.action_space = Discrete(n_actions)
        
        # observation space is defined by the subclass
        #self.observation_space = Discrete(len(self._reachable_states))
        
        self._starts = starts
        self.state = None
        
        self._transition_cache = {}

        if reachable_states is None:
            # Get reachable states by searching through the state space
            reachable_states = set()
            for s in self._starts:
                self._search(s, reachable_states)
       
        # organize the reachable states in a consistent order
        self._reachable_states = reachable_states
        
        # build encoder and decoder for tabular access
        self._decoder = [s for s in sorted(self._reachable_states)]
        self._encoder = {s: i for i, s in enumerate(self._decoder)}
        
        self.tabular = tabular
        if self.tabular:
            self.observation_space = Discrete(len(self._reachable_states))
        
                              
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = random.choice(self._starts)
        observation = self.state
        info = {}
        
        #if self.render_mode == "human":
        #    self._render_frame()

        if self.tabular:
            observation = self.state2id(observation)

        return observation, info

    def step(self, action):
        assert self.action_space.contains(action)
        
        # after the random elements are sampled, the environment transition is deterministic
        elements = self._sample_random_elements(self.state, action)
        
        next_state, reward, done, _ = self._deterministic_step(self.state, action, *elements)
        self.state = next_state
        info = {}
        
        if self.tabular:
            next_state = self.state2id(next_state)
        
        return next_state, reward, done, False, info


    ## render needs to be implemented in each environment, so we don't provide a default implementation here
    ## we use the default close, can be overwritten.

    ## Additional Interface

    ## Tabular access
    def states(self):
        """Returns a generator over all possible environment states."""
        return range(len(self._encoder))
    
    def state2id(self, state):
        """Converts a raw state into a unique integer."""
        
        if isinstance(state, list):
            return [self._encoder[s] for s in state]
        
        return self._encoder[state]

    def id2state(self, i):
        """Reverts an encoded integer back to its raw state."""
        if isinstance(i, list):
            return [self._decoder[j] for j in i]
        
        return self._decoder[i]
    
    def is_reachable(self, state):
        """Returns True if the state can be reached from at least one start location,
        False otherwise."""
        return state in self._reachable_states

    def actions(self):
        """Returns a generator over all possible agent actions."""
        return range(self.action_space.n)

    def action2id(self, action_label):
        """Converts a action label into a numeric action ID."""
        action_ids = dict(zip(self.action_labels, range(len(self.action_labels))))
        id = action_ids.get(action_label, -1)
        return id

    def id2action(self, action, type="text"):
        """Converts a numeric action ID into a label. Choices for type are 'text' and 'arrow'."""
        action_labels = self.action_labels + [""]  # Add empty label for hidden actions

        return action_labels[int(action)]

    def model(self, state, action):
        """Returns the transitions from the given state-action pair."""
        
        if self.tabular:
            state = self.id2state(state)
        
        # caching the model for efficiency    
        sa_pair = (state, action)
        if sa_pair in self._transition_cache:
            return self._transition_cache[sa_pair]

        tr = list(self._generate_transitions(state, action))
        tr = [[t[0] for t in tr], np.array([t[1] for t in tr]), np.array([t[2] for t in tr]), np.array([t[3] for t in tr])]

        assert (tr[3] >= 0.0).all(), "transition probabilities must be nonnegative"
        assert np.sum(tr[3]) == 1.0, "transition probabilities must sum to 1"

        if self.tabular:
            tr[0] = self.state2id(tr[0])

        self._transition_cache[sa_pair] = tr
        return tr

    def _search(self, state, visited):
        """A recursive depth-first search that adds all reachable states to the visited set."""
        visited.add(state)
        for a in self.actions():
            for transition in self._generate_transitions(state, a):
                next_state, _, done, prob = transition
                if prob > 0.0:
                    if not done and next_state not in visited:
                        self._search(next_state, visited)
                    # MFH add the final state!
                    if done and next_state not in visited:
                        visited.add(next_state)

    # Do not overwrite!
    def _deterministic_step(self, state, action, *random_elements):
        """An environment step that is deterministic conditioned on the given values
        of the random variables (if there are any).

        Do not override.
        """
        next_state, prob = self._next_state(state, action, *random_elements)
        reward = self._reward(state, action, next_state)
        done = self._done(state, action, next_state)
        #if done:
        #    next_state = state
        return next_state, reward, done, prob

    # these need to be implemented in the subclass
    def _sample_random_elements(self, state, action):
        """Samples values for random elements (if any) that influence the environment
        transition from the current state-action pair (S, A).

        If the environment is deterministic, no need to override this method.
        """
        return ()
    
    @abstractmethod
    def _next_state(self, state, action, *random_elements):
        """Returns the next state S' induced by the state-action pair (S, A), which must
        be deterministic conditioned on the values of any random_elements. Also returns
        the probability that this particular transition occurred."""
        raise NotImplementedError

    @abstractmethod
    def _reward(self, state, action, next_state):
        """Returns the reward yielded by this (S,A,S') outcome."""
        raise NotImplementedError

    @abstractmethod
    def _done(self, state, action, next_state):
        """Returns True if this (S,A,S') outcome should terminate, False otherwise."""
        raise NotImplementedError

    @abstractmethod
    def _generate_transitions(self, state, action):
        """Returns a generator over all transitions from this state-action pair.

        Should be overridden in the subclass.
        """
        raise NotImplementedError
