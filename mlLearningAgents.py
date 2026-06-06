# mlLearningAgents.py
# parsons/27-mar-2017
"""
A stub for a reinforcement learning agent to work with the Pacman
piece of the Berkeley AI project:

http://ai.berkeley.edu/reinforcement.html

As required by the licensing agreement for the PacMan AI we have:

Licensing Information:  You are free to use or extend these projects for
educational purposes provided that (1) you do not distribute or publish
solutions, (2) you retain this notice, and (3) you provide clear
attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
 
Attribution Information: The Pacman AI projects were developed at UC Berkeley.
The core projects and autograders were primarily created by John DeNero
(denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
Student side autograding was added by Brad Miller, Nick Hay, and
Pieter Abbeel (pabbeel@cs.berkeley.edu).
----------------------------------------------------------------------------
This template was originally adapted to KCL by Simon Parsons, but then
revised and updated to Python3 for the 2022 course by Dylan Cope and Lin Li
----------------------------------------------------------------------------

----------------------------------------------------------------------------
Solution to CW2 for [Redacted] - Machine Learning
Project: Q-learning agent for the Pacman game
Authors: Daniel Bate & Mark Saviour Farrugia
Date: 13 Mar 2026

References:
    - [1] [Redacted], 'Lab 2: Q-learning solution', [Redacted] Machine Learning, 2026
        Last Accessed: Feb. 18, 2026. [Online]. 
        Available: [Redacted]

Bibliography:
    - [1] H. N and P. G, ‘A Brief Study of Deep Reinforcement Learning with Epsilon-Greedy Exploration’, International Journal of Computing and Digital Systems, 
        Last Accessed: Mar. 23, 2026. [Online].
        Available: https://journal.uob.edu.bh/items/16a655ec-eb56-4cc2-977d-ae24a87069b1
    - [2] A. Dos Santos Mignon and R. L. De Azevedo Da Rocha, ‘An Adaptive Implementation of ε-Greedy in Reinforcement Learning’, Procedia Computer Science, 
        Last Accessed: Mar. 23, 2026. [Online]. 
        Available: https://www.sciencedirect.com/science/article/pii/S1877050917311134?via%3Dihub
    - [4] ‘Q-Learning in Reinforcement Learning’, GeeksforGeeks. 
        Last Accessed: Mar. 23, 2026. [Online]. 
        Available: https://www.geeksforgeeks.org/machine-learning/q-learning-in-python/
    - [5] T. ABDULLAHI, ‘Reinforcement Learning with Python: A Comprehensive Guide with Code Examples’, Medium. 
        Last Accessed: Mar. 23, 2026. [Online]. 
        Available: https://medium.com/@aaltanim/reinforcement-learning-with-python-a-comprehensive-guide-with-code-examples-8d055fc54514
    - [6] R. S. Sutton and A. G. Barto, ‘Reinforcement Learning: An Introduction’.
        Last Accessed: Mar. 23, 2026. [Online]. 
        Available: https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf
    - [7] C. Watkins and P. Dayan, ‘Technical Note: Q-Learning’, Machine Learning, 
        Last Accessed: Mar. 23, 2026. [Online]. 
        Available: https://link.springer.com/article/10.1007/BF00992698
    - [8] M. G. Bellemare, S. Srinivasan, G. Ostrovski, T. Schaul, D. Saxton, and R. Munos, ‘Unifying Count-Based Exploration and Intrinsic Motivation’, 
        Last Accessed: Mar. 23, 2026. [Online]. 
        Available: https://arxiv.org/abs/1606.01868
    - [9] ‘What is Reinforcement Learning? - Reinforcement Learning Explained - AWS’, Amazon Web Services, Inc. 
        Last Accessed: Mar. 23, 2026. [Online]. 
        Available: https://aws.amazon.com/what-is/reinforcement-learning/

Author notes:
    - Docs should adhere to PEP docstrings
    - I'd usally include inline tests or a seperate test file to demonstrate good programming 
    style as mentioned in the mark shceme but I'm concerned about environment dependencies for this.
----------------------------------------------------------------------------
"""
from __future__ import absolute_import
from __future__ import print_function

import random

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util

# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------

def getLegalActionsWithoutStop(state: GameState) -> list:
    """
    Return legal Pacman actions with STOP removed

    Args:
        state: The current game state

    Returns:
        List of legal actions excluding STOP
    """
    legal = list(state.getLegalPacmanActions()) # Create a copy for safe modification
    if Directions.STOP in legal:
        legal.remove(Directions.STOP)
    return legal

# ----------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------

class GameStateFeatures:
    """
    Wrapper class around a game state reference with added
    relevant features and information for a Q learning agent
    """

    def __init__(self, state: GameState) -> None:
        """
        Extract position, ghost, and food information from the given game state

        Args:
            state: A given game state object
        """
        # Underlying game state reference
        self.state = state

        # (x, y) tuple representing Pacman's current position
        self.pacman_position = state.getPacmanPosition()
        # Tuple of (x, y) tuples for each ghost's current position
        self.ghost_positions = tuple(state.getGhostPositions())
        # Grid object representing food locations (available on the underlying game state)
        self.food = state.getFood()

    # Game states are considered equal if they have the same position, ghost positions
    # and food configuration. None check prevents crash on init
    def __eq__(self, other) -> bool:
        if other is None:
            return False
        return (
            self.pacman_position == other.pacman_position
            and self.ghost_positions == other.ghost_positions
            and self.food == other.food
        )

    # Converts a state object into hashable values to be used as a key in dicts
    def __hash__(self) -> int:
        return hash((
            self.pacman_position,
            self.ghost_positions,
            self.food,
        ))

    def getLegalPacmanActions(self) -> list:
        """
        Obtains legal Pacman actions from the underlying game state

        Returns:
            List of legal Pacman action strings
        """
        return self.state.getLegalPacmanActions()


class QLearnAgent(Agent):
    """
    Q learning agent for the Pacman game    
    """
    def __init__(self,
                 alpha: float = 0.3,
                 epsilon: float = 0.01,
                 gamma: float = 0.8,
                 maxAttempts: int = 5,
                 numTraining: int = 10) -> None:
        """
        Initialize the Q learning agent with the given parameters

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)           # learning rate
        self.epsilon = float(epsilon)       # exploration rate
        self.gamma = float(gamma)           # discount factor
        self.maxAttempts = int(maxAttempts) # How many times to try each action in each state
        self.numTraining = int(numTraining) # number of training episodes

        self.episodesSoFar = 0 # Count the number of games we have played

        # Persistent storage for learning actions, essentially agent knowledge
        # as it works across episodes
        # This maps a (state, action) pair to a q value, gets updated by learn
        self.q_values = {}

        # Persistent storage for counts of state action pairs
        # This maps a (state, action) pair to a count, gets updated by learn
        self.counts = {}

        # Per episode tracking vars, should be reset every episode
        self.last_state = None       # Previous GameStateFeatures
        self.last_action = None      # Previous action taken
        self.last_state_raw = None   # Previous raw GameState

    def incrementEpisodesSoFar(self) -> None:
        """
        Increment the episode counter by one
        """
        self.episodesSoFar += 1

    def getEpisodesSoFar(self) -> int:
        """
        Return the number of episodes completed so far
        """
        return self.episodesSoFar

    def getNumTraining(self) -> int:
        """
        Return the total number of training episodes
        """
        return self.numTraining

    def setEpsilon(self, value: float) -> None:
        """
        Set the exploration rate epsilon

        Args:
            value: New epsilon value
        """
        self.epsilon = value

    def getAlpha(self) -> float:
        """
        Return the learning rate alpha
        """
        return self.alpha

    def setAlpha(self, value: float) -> None:
        """
        Set the learning rate alpha

        Args:
            value: New alpha value
        """
        self.alpha = value

    def getGamma(self) -> float:
        """
        Return the discount factor gamma
        """
        return self.gamma

    def getMaxAttempts(self) -> int:
        """
        Return the minimum visit threshold for the exploration function
        """
        return self.maxAttempts

    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Compute the reward by the change in game score for a
        given trajectory

        Args:
            startState: The state before the action
            endState: The state after the action

        Returns:
            The reward assigned for the given trajectory
        """
        return endState.getScore() - startState.getScore()

    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Look up Q(state, action) from the Q-table, unseen pairs
        assigned a value of 0.0

        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        return self.q_values.get((state, action), 0.0)

    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Returns the maximum achievable Q value from the given state
        over all legal moves

        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q value attainable from the state
        """
        legal = getLegalActionsWithoutStop(state.state)
        if not legal:
            return 0.0
        return max(self.getQValue(state, a) for a in legal)

    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Perform a single Q learning update

        Formula from lecture on Reinforcement Learning 2:
            Q(s,a) <- Q(s,a) + alpha * [R + gamma * max_a' Q(s',a') - Q(s,a)]

        Args:
            state: the initial state (s)
            action: the action that was taken (a)
            reward: the reward received (R)
            nextState: the resulting state (s')
        """
        old_q = self.getQValue(state, action)                       # Q(s, a)
        best_next_q = self.maxQValue(nextState)                     # max_a' Q(s', a')
        td_target = reward + (self.getGamma() * best_next_q)        # R + gamma * max_a' Q(s', a')
        updated_q = old_q + (self.getAlpha() * (td_target - old_q)) # Q(s,a) + alpha * [target - Q(s,a)]
        self.q_values[(state, action)] = updated_q                  # Store updated Q(s, a)

    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Increment the stored visitation counts

        Args:
            state: The state in which the action was taken
            action: The action that was taken
        """
        key = (state, action)
        self.counts[key] = self.counts.get(key, 0) + 1

    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Retrieve the visitiation count for a state-action pair

        Returns 0 for pairs that have never been visited

        Args:
            state: The state to query
            action: The action to query

        Returns:
            Number of times that the action has been taken in a given state
        """
        return self.counts.get((state, action), 0)

    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Count-based exploration function as discussed in
        lecture on Reinforcement Learning 2

        If the count for a state-action pair is below a certain threshold, return
        infinity to force exploration, otherwise return the learned utility

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: how many times this state-action pair has been visited

        Returns:
            The exploration value
        """        
        # Prefer under tried actions while learning, otherwise use learned utility
        if counts < self.getMaxAttempts():
            return float('inf')
        return utility

    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action for the current game step

        Args:
            state: the current game state

        Returns:
            The action to take
        """
        legal = getLegalActionsWithoutStop(state) # Legal pacman actions without STOP
        state_features = GameStateFeatures(state) # Init state obj for current state


        # Step 1: Learn from previous transition
        # Can follow the same reward pattern as the BanditAgent in the
        # RL-2 lab but extended to state action pairs
        if self.last_state is not None:
            reward = self.computeReward(self.last_state_raw, state)
            self.learn(
                self.last_state,
                self.last_action,
                reward,
                state_features,
            )
            self.updateCount(self.last_state, self.last_action)

        # Step 2: Choose an action using epsilon greedy and exploration function
        # Picks a random action with epsilon probability OR otherwise we highest computed
        # exploration adjusted Q value
        # Epsilon greedy pattern is based on the RL 2 Lab solution
        if util.flipCoin(self.epsilon):
            action = random.choice(legal) 
        else:
            best_value = float('-inf')                # Start below any real value so first action always wins
            best_actions = []                         # Tracks all actions tied for the best value
            for a in legal:
                q = self.getQValue(state_features, a) # Q value for this state action pair
                n = self.getCount(state_features, a)  # Count for this action
                value = self.explorationFn(q, n)      # Either get q or inf based on count
                if value > best_value:
                    # New best action found, replace the list
                    best_value = value
                    best_actions = [a]
                elif value == best_value:
                    # Tie with current best, add to candidates
                    best_actions.append(a)
            # Break ties randomly to avoid bias
            action = random.choice(best_actions)

        # Step 3: Store state action for next calls learning step
        self.last_state = state_features
        self.last_action = action
        self.last_state_raw = state

        return action

    def final(self, state: GameState) -> None:
        """
        Handle the end of an episode, called after a win or a loss

        Args:
            state: the final game state
        """
        # Learn from the terminal transition
        if self.last_state is not None and self.last_action is not None and self.last_state_raw is not None:
            terminal_features = GameStateFeatures(state)
            reward = self.computeReward(self.last_state_raw, state)
            self.updateCount(self.last_state, self.last_action)
            self.learn(self.last_state, self.last_action, reward, terminal_features)

        # Reset episode based tracking variables.
        self.last_state = None
        self.last_action = None
        self.last_state_raw = None

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            self.setAlpha(0)
            self.setEpsilon(0)
