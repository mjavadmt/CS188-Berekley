# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from util import *
# from searchAgents import PositionSearchProblem


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def extract_tuple_elements(tup, elem):
    new_tup = [i for i in tup]
    new_tup.append(elem)
    return tuple(new_tup)


def StackDFS(problem):
    stack = Stack()
    stack.push(problem.getStartState())
    visited = set()
    parents = dict()
    while not stack.isEmpty():
        popped = stack.pop()
        visited.add(popped[0])
        for neighbor in problem.getSuccessors(popped):
            if neighbor[0] not in visited:
                stack.push(neighbor)
                parents[neighbor] = popped


def DFS_Explore(node, parents, visited, problem, parent):
    visited.add(node[0])
    parents[node] = parent
    if problem.isGoalState(node[0]):
        return True
    for neighbor in problem.getSuccessors(node[0])[::-1]:
        if neighbor[0] not in visited:
            if DFS_Explore(neighbor, parents, visited, problem, node):
                return True
    return False


def traverse_to_start(last, parents, problem):
    path = []
    while last != problem.getStartState():
        path.append(last[1])
        last = parents[last]
    return path


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    stack=util.Stack()
    visited=[]
    path = []
    parents = {}
    startNode=problem.getStartState()
    stack.push((startNode,"nothing",0))
    while not stack.isEmpty():
        current=stack.pop()
        if current[0] not in visited :
            visited.append(current[0])
            if problem.isGoalState(current[0]):
                while current[0] != startNode:
                    path.append(current[1])
                    current = parents[current]
                return path[::-1]
            # neighbors=problem.getSuccessors(current[0])
            for neighbor in problem.getSuccessors(current[0]):
                #((2,3) , "west",1)
                if neighbor[0] not in visited:
                    stack.push(neighbor)
                    parents[neighbor] = current


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    parents = {problem.getStartState(): None}
    visited = set()
    visited.add(problem.getStartState())
    queue = Queue()
    for neighbor in problem.getSuccessors(problem.getStartState()):
        visited.add(neighbor[0])
        queue.push(neighbor)
        parents[neighbor] = problem.getStartState()
    while not queue.isEmpty():
        dequeued = queue.pop()
        if problem.isGoalState(dequeued[0]):
            break
        for neighbor in problem.getSuccessors(dequeued[0]):
            if neighbor[0] not in visited:
                visited.add(neighbor[0])
                queue.push(neighbor)
                parents[neighbor] = dequeued
    if problem.isGoalState(dequeued[0]):
        return traverse_to_start(dequeued, parents, problem)[::-1]
    else:
        return []

class MyPriorityQueue:
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.
    """
    def  __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        return heapq.heappop(self.heap)
        # (_, _, item) = heapq.heappop(self.heap)
        # return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i[0] == item[0]:
                if p <= priority:
                    return False
                    # break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                # break
                return True
        else:
            self.push(item, priority)
            return True


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    parents = {problem.getStartState(): None}
    visited = set()
    visited.add(problem.getStartState())
    queue = MyPriorityQueue()
    for neighbor in problem.getSuccessors(problem.getStartState()):
        queue.push(neighbor, neighbor[2])
        parents[neighbor] = problem.getStartState()
    while not queue.isEmpty():
        dequeued = queue.pop()
        visited.add(dequeued[2][0])
        if problem.isGoalState(dequeued[2][0]):
            break
        for neighbor in problem.getSuccessors(dequeued[2][0]):
            if neighbor[0] not in visited:
                updated = queue.update(neighbor, neighbor[2] + dequeued[0])
                if updated:
                    parents[neighbor] = dequeued[2]

    if problem.isGoalState(dequeued[2][0]):
        return traverse_to_start(dequeued[2], parents, problem)[::-1]
    else:
        return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    parents = {problem.getStartState(): None}
    visited = set()
    visited.add(problem.getStartState())
    queue = PriorityQueue()
    for neighbor in problem.getSuccessors(problem.getStartState()):
        estimate = neighbor[2] + heuristic(neighbor[0], problem)
        queue.push(neighbor, estimate)
        parents[neighbor] = problem.getStartState()
    while not queue.isEmpty():
        dequeued = queue.pop()
        real_cost = dequeued[0] - heuristic(dequeued[2][0], problem)
        visited.add(dequeued[2][0])
        if problem.isGoalState(dequeued[2][0]):
            break
        for neighbor in problem.getSuccessors(dequeued[2][0]):
            if neighbor[0] not in visited:
                updated = queue.update(neighbor, neighbor[2] + real_cost + heuristic(neighbor[0], problem))
                if updated:
                    parents[neighbor] = dequeued[2]

    if problem.isGoalState(dequeued[2][0]):
        return traverse_to_start(dequeued[2], parents, problem)[::-1]
    else:
        return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
