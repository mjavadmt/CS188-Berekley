# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random
import util

from game import Agent

max_assigned_number = 999999999


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        manhattan_distance = 0
        cuurent_state_foods = currentGameState.getFood()
        foodList = cuurent_state_foods.asList()
        manhattanDistancesList = []
        for food in foodList:
            manhattan_distance = manhattanDistance(food, newPos)
            manhattanDistancesList.append(-manhattan_distance)
        maximumTilNow = max(manhattanDistancesList)
        for i, state in enumerate(newGhostStates):
            if newScaredTimes[i] == 0:
                if state.getPosition() == newPos:
                    return -max_assigned_number
        return maximumTilNow


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class StateInfo:
    def __init__(self, current_deep, current_agent):
        self.current_deep = current_deep
        self.current_agent = current_agent


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
          Here are some method calls that might be useful when implementing minimax.
          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1
          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action
          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        state_info = StateInfo(0, self.index)
        return self.assign_value(gameState, state_info)

    def base_case_condition(self, gameState, state_info):
        return state_info.current_deep == self.depth or gameState.isWin() or gameState.isLose()

    def assign_value(self, gameState, state_info):
        if state_info.current_agent == gameState.getNumAgents():
            state_info = StateInfo(state_info.current_deep + 1, 0)

        if self.base_case_condition(gameState, state_info):
            return self.evaluationFunction(gameState)
        elif state_info.current_deep == 0 and state_info.current_agent == 0:
            return self.assign_root(gameState, state_info)
        elif state_info.current_agent == self.index:
            return self.assign_max(gameState, state_info)
        elif state_info.current_agent != self.index:
            return self.assign_min(gameState, state_info)

    def assign_root(self, gameState, state_info):
        actions_values = []
        legal_actions = gameState.getLegalActions(state_info.current_agent)
        for action in legal_actions:
            successor = gameState.generateSuccessor(
                state_info.current_agent, action)
            updated_state_info = StateInfo(
                state_info.current_deep, state_info.current_agent + 1)
            value = self.assign_value(successor, updated_state_info)
            actions_values.append((value, action))
        return max(actions_values, key=lambda x: x[0])[1]

    def assign_min(self, gameState, state_info):
        current_min = max_assigned_number + 1
        legal_actions = gameState.getLegalActions(state_info.current_agent)
        for action in legal_actions:
            successor = gameState.generateSuccessor(
                state_info.current_agent, action)
            updated_state_info = StateInfo(
                state_info.current_deep, state_info.current_agent + 1)
            value = self.assign_value(
                successor, updated_state_info)
            current_min = min(value, current_min)
        return current_min

    def assign_max(self, gameState, state_info):
        current_max = - (max_assigned_number + 1)
        legalActions = gameState.getLegalActions(state_info.current_agent)
        for action in legalActions:
            successor = gameState.generateSuccessor(
                state_info.current_agent, action)
            updated_state_info = StateInfo(
                state_info.current_deep, state_info.current_agent + 1)
            value = self.assign_value(successor, updated_state_info)
            current_max = max(value, current_max)
        return current_max


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
          Here are some method calls that might be useful when implementing minimax.
          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1
          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action
          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        state_info = StateInfo(0, self.index)
        alpha = - (max_assigned_number + 1)
        beta = max_assigned_number + 1
        return self.assign_value(gameState, state_info, alpha, beta)

    def base_case_condition(self, gameState, state_info):
        return state_info.current_deep == self.depth or gameState.isWin() or gameState.isLose()

    def assign_value(self, gameState, state_info, alpha, beta):
        if state_info.current_agent == gameState.getNumAgents():
            state_info = StateInfo(state_info.current_deep + 1, 0)

        if self.base_case_condition(gameState, state_info):
            return self.evaluationFunction(gameState)
        elif state_info.current_deep == 0 and state_info.current_agent == 0:
            return self.assign_root(gameState, state_info)
        elif state_info.current_agent == self.index:
            return self.assign_max(gameState, state_info, alpha, beta)
        elif state_info.current_agent != self.index:
            return self.assign_min(gameState, state_info, alpha, beta)

    def assign_root(self, gameState, state_info):
        actions_values = []
        legal_actions = gameState.getLegalActions(state_info.current_agent)
        alpha = - (max_assigned_number + 1)
        beta = max_assigned_number + 1
        for action in legal_actions:
            successor = gameState.generateSuccessor(
                state_info.current_agent, action)
            updated_state_info = StateInfo(
                state_info.current_deep, state_info.current_agent + 1)
            value = self.assign_value(
                successor, updated_state_info, alpha, beta)
            if value > beta:
                return max(actions_values, key=lambda x: x[0])[1]
            alpha = max(alpha, value)
            actions_values.append((value, action))
        return max(actions_values, key=lambda x: x[0])[1]

    def assign_min(self, gameState, state_info, alpha, beta):
        current_min = max_assigned_number + 1
        legal_actions = gameState.getLegalActions(state_info.current_agent)
        for action in legal_actions:
            successor = gameState.generateSuccessor(
                state_info.current_agent, action)
            updated_state_info = StateInfo(
                state_info.current_deep, state_info.current_agent + 1)
            value = self.assign_value(
                successor, updated_state_info, alpha, beta)
            current_min = min(value, current_min)
            if current_min < alpha:
                return current_min
            beta = min(beta, current_min)

        return current_min

    def assign_max(self, gameState, state_info, alpha, beta):
        current_max = - (max_assigned_number + 1)
        legalActions = gameState.getLegalActions(state_info.current_agent)
        for action in legalActions:
            successor = gameState.generateSuccessor(
                state_info.current_agent, action)
            updated_state_info = StateInfo(
                state_info.current_deep, state_info.current_agent + 1)
            value = self.assign_value(
                successor, updated_state_info, alpha, beta)
            current_max = max(value, current_max)
            if current_max > beta:
                return current_max
            alpha = max(alpha, current_max)

        return current_max


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
          Here are some method calls that might be useful when implementing minimax.
          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1
          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action
          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        state_info = StateInfo(0, self.index)
        return self.assign_value(gameState, state_info)

    def base_case_condition(self, gameState, state_info):
        return state_info.current_deep == self.depth or gameState.isWin() or gameState.isLose()

    def assign_value(self, gameState, state_info):
        if state_info.current_agent == gameState.getNumAgents():
            state_info = StateInfo(state_info.current_deep + 1, 0)

        if self.base_case_condition(gameState, state_info):
            return self.evaluationFunction(gameState)
        elif state_info.current_deep == 0 and state_info.current_agent == 0:
            return self.assign_root(gameState, state_info)
        elif state_info.current_agent == self.index:
            return self.assign_max(gameState, state_info)
        elif state_info.current_agent != self.index:
            return self.expecti_max(gameState, state_info)

    def assign_root(self, gameState, state_info):
        actions_values = []
        legal_actions = gameState.getLegalActions(state_info.current_agent)
        for action in legal_actions:
            successor = gameState.generateSuccessor(
                state_info.current_agent, action)
            updated_state_info = StateInfo(
                state_info.current_deep, state_info.current_agent + 1)
            value = self.assign_value(successor, updated_state_info)
            actions_values.append((value, action))
        return max(actions_values, key=lambda x: x[0])[1]

    def expecti_max(self, gameState, state_info):
        legal_actions = gameState.getLegalActions(state_info.current_agent)
        successor_probability = 1 / len(legal_actions)
        current_sum = 0
        for action in legal_actions:
            successor = gameState.generateSuccessor(
                state_info.current_agent, action)
            updated_state_info = StateInfo(
                state_info.current_deep, state_info.current_agent + 1)
            value = self.assign_value(
                successor, updated_state_info)
            current_sum += successor_probability * value
        return current_sum

    def assign_max(self, gameState, state_info):
        current_max = - (max_assigned_number + 1)
        legalActions = gameState.getLegalActions(state_info.current_agent)
        for action in legalActions:
            successor = gameState.generateSuccessor(
                state_info.current_agent, action)
            updated_state_info = StateInfo(
                state_info.current_deep, state_info.current_agent + 1)
            value = self.assign_value(successor, updated_state_info)
            current_max = max(value, current_max)
        return current_max


def ghost_is_scared(ghost_position):
    return ghost_position.scaredTimer > 0

import random

class AssignValueToAgents:
    def __init__(self):
        self.ghost = random.randint(8,12)
        self.scared_ghost = random.randint(48,52)
        self.food = self.ghost

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    agent_values = AssignValueToAgents()
    score = currentGameState.getScore()

    ghostValue = 10.0
    foodValue = 10.0  
    scaredGhostValue = 50.0  

    values = []
    for ghost_position in newGhostStates:
        manhattan_distance = manhattanDistance(
            newPos, ghost_position.getPosition())
        try:
            values.append(("+" , agent_values.scared_ghost / manhattan_distance)) if ghost_is_scared(
                ghost_position) else values.append(("-" , agent_values.ghost / manhattan_distance))
        except:
            pass
    for val in values:
        if val[0] == "+":
            score += val[1]
        else:
            score -= val[1]
    food_distances = [manhattanDistance(newPos, food_pos) for food_pos in newFood.asList()]
    score += agent_values.food / min(food_distances) if food_distances else 0
    return score


# Abbreviation
better = betterEvaluationFunction
