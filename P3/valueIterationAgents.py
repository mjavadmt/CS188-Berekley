# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp
import util

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def update_values(self, updated_values):
        self.values = updated_values

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        iteration_count = self.iterations
        i = 0
        while i < iteration_count:
            manipulated_values = self.values.copy()
            all_states = self.mdp.getStates()
            non_terminal_states = [
                s for s in all_states if not self.mdp.isTerminal(s)]
            for state in non_terminal_states:
                get_possible_actions = self.mdp.getPossibleActions(state)
                all_Qk = []
                for action in get_possible_actions:
                    Qk = self.getQValue(state, action)
                    all_Qk.append(Qk)
                Vk = max(all_Qk)
                manipulated_values[state] = Vk
            self.update_values(manipulated_values)
            i += 1

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        states, probabilities = zip(
            *self.mdp.getTransitionStatesAndProbs(state, action))
        values_calculated = []
        for i in range(len(states)):
            current_state = states[i]
            current_probability = probabilities[i]
            current_reward = self.mdp.getReward(state, action, current_state)
            value = current_probability * \
                (current_reward + self.discount * self.values[current_state])
            values_calculated.append(value)
        return sum(values_calculated)

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        elif not self.mdp.isTerminal(state):
            Vk = util.Counter().copy()
            for action in self.mdp.getPossibleActions(state):
                Vk[action] = self.getQValue(state, action)
            return Vk.argMax()
        return ""

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        all_states = self.mdp.getStates()
        all_length = len(all_states)
        states_to_update = [all_states[i % all_length] for i in range(
            self.iterations) if not self.mdp.isTerminal(all_states[i % all_length])]

        for state in states_to_update:
            actions = self.mdp.getPossibleActions(state)
            Q_values = [self.getQValue(state, action) for action in actions]
            v_star = max(Q_values)
            self.values[state] = v_star


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def update_predecessors(self, current_state, action, predecessors: dict):
        for new_state, probability in self.mdp.getTransitionStatesAndProbs(current_state, action):
            new_state_value = predecessors.get(new_state, {})
            new_state_value = {*new_state_value, current_state}
            predecessors[new_state] = new_state_value

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        priority_queue = util.PriorityQueue()

        all_states = self.mdp.getStates()
        predecessors = dict()
        non_terminal_states = [
            s for s in all_states if not self.mdp.isTerminal(s)]
        for state in non_terminal_states:
            for action in self.mdp.getPossibleActions(state):
                self.update_predecessors(state, action, predecessors)

        for state in non_terminal_states:
            actions = self.mdp.getPossibleActions(state)
            Q_values = [self.getQValue(state, action) for action in actions]
            v_star = max(Q_values)
            diff = abs(self.values[state] - v_star)
            priority_queue.update(state, -diff)
        i = 0
        while i < self.iterations and not priority_queue.isEmpty():
            state = priority_queue.pop()

            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)
                Q_values = [self.getQValue(state, action)
                            for action in actions]
                v_star = max(Q_values)
                self.values[state] = v_star

            non_terminal_predecessors = [
                predecessor for predecessor in predecessors[state] if not self.mdp.isTerminal(predecessor)]

            for predecessor in non_terminal_predecessors:
                actions = self.mdp.getPossibleActions(predecessor)
                Q_values = [self.getQValue(predecessor, action)
                            for action in actions]
                v_star = max(Q_values)
                diff = abs(self.values[predecessor] - v_star)
                if diff > self.theta:
                    priority_queue.update(predecessor, -diff)
            i += 1
