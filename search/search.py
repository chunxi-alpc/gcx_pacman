# coding=UTF-8
# search.py

"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

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
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    #初始状态
    s = problem.getStartState()
    #标记已经搜索过的状态集合exstates
    exstates = []
    #用栈实现dfs
    states = util.Stack()
    states.push((s, []))
    #循环终止条件：遍历完毕/目标测试成功
    while not states.isEmpty() and not problem.isGoalState(s):
        state, actions = states.pop()
        exstates.append(state)
        successor = problem.getSuccessors(state)
        for node in successor:
            coordinates = node[0]
            direction = node[1]
            #判断状态是否重复
            if not coordinates in exstates:
                states.push((coordinates, actions + [direction]))
            #把最后搜索的状态赋值到s，以便目标测试
            s = coordinates
    #返回动作序列
    return actions + [direction]
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    #初始状态
    s = problem.getStartState()
    #标记已经搜索过的状态集合exstates
    exstates = []
    #用队列queue实现bfs
    states = util.Queue()
    states.push((s, []))
    while not states.isEmpty():
        state, action = states.pop()
        #目标测试
        if problem.isGoalState(state):
            return action
        #检查重复
        if state not in exstates:
            successor = problem.getSuccessors(state)
            exstates.append(state)
            #把后继节点加入队列
            for node in successor:
                coordinates = node[0]
                direction = node[1]
                if coordinates not in exstates:
                    states.push((coordinates, action + [direction]))
    #返回动作序列
    return action
    util.raiseNotDefined()

def uniformCostSearch(problem):
    #初始状态
    start = problem.getStartState()
    #标记已经搜索过的状态集合exstates
    exstates = []
    #用优先队列PriorityQueue实现ucs
    states = util.PriorityQueue()
    states.push((start, []) ,0)
    while not states.isEmpty():
        state, actions = states.pop()
        #目标测试
        if problem.isGoalState(state):
            return actions
        #检查重复
        if state not in exstates:
            #扩展
            successors = problem.getSuccessors(state)
            for node in successors:
                coordinate = node[0]
                direction = node[1]
                if coordinate not in exstates:
                    newActions = actions + [direction]
                    #ucs比bfs的区别在于getCostOfActions决定节点扩展的优先级
                    states.push((coordinate, actions + [direction]), problem.getCostOfActions(newActions))
        exstates.append(state)
    return actions
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    启发式函数有两个输入变量：搜索问题的状态 (主参数), 和问题本身(相关参考信息)
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost f(n) and heuristic g(n) first."
    start = problem.getStartState()
    exstates = []
    # 使用优先队列，每次扩展都是选择当前代价最小的方向
    states = util.PriorityQueue()
    states.push((start, []), nullHeuristic(start, problem))
    nCost = 0
    while not states.isEmpty():
        state, actions = states.pop()
        #目标测试
        if problem.isGoalState(state):
            return actions
        #检查重复
        if state not in exstates:
            #扩展
            successors = problem.getSuccessors(state)
            for node in successors:
                coordinate = node[0]
                direction = node[1]
                if coordinate not in exstates:
                    newActions = actions + [direction]
                    #计算动作代价和启发式函数值得和
                    newCost = problem.getCostOfActions(newActions) + heuristic(coordinate, problem)
                    states.push((coordinate, actions + [direction]), newCost)
        exstates.append(state)
    #返回动作序列
    return actions
    util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
