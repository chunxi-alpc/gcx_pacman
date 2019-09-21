# coding=UTF-8
# searchAgents.py
# ---------------



"""
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

Commands to invoke other search strategies can be found in the project
description.

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the project
description for details.

Good luck and happy searching!
"""

from game import Directions
from game import Agent
from game import Actions
import util
import time
import search
import sys

class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP

#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    As a default, this agent runs DFS on a PositionSearchProblem to find
    location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='uniformCostSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError, fn + ' is not a search function in search.py.'
        func = getattr(search, fn)
        if 'heuristic' not in func.func_code.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError, heuristic + ' is not a function in searchAgents.py or search.py.'
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError, prob + ' is not a search problem type in SearchAgents.py.'
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception, "No search function provided for SearchAgent"
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions  = self.searchFunction(problem) # Find a path
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP

class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print 'Warning: this does not look like a regular search maze'
        self._visited, self._visitedlist, self._expanded = {}, [], 0

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

class StayEastSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the West side of the board.

    The cost function for stepping into a position (x,y) is 1/2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: .5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn, (1, 1), None, False)

class StayWestSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the East side of the board.

    The cost function for stepping into a position (x,y) is 2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################

class CornersProblem(search.SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.
    You must select a suitable state space and successor function
    """

    def __init__(self, startingGameState):
        """
        Stores the walls, pacman's starting position and corners.
        """
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height-2, self.walls.width-2
        self.corners = ((1,1), (1,top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print 'Warning: no food in corner ' + str(corner)
        self._expanded = 0 # Number of search nodes expanded
        # Please add any code here which you would like to use
        # in initializing the problem
        "*** YOUR CODE HERE ***"
        self.right = right
        self.top = top
        
    def getStartState(self):
        """
        Returns the start state (in your state space, not the full Pacman state
        space)
        """
        "*** YOUR CODE HERE ***"
        #初始节点（开始位置，角落情况）
        allCorners = (False, False, False, False)
        start = (self.startingPosition, allCorners)
        return start
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
        Returns whether this search state is a goal state of the problem.
        """
        "*** YOUR CODE HERE ***"
        #目标测试：四个角落都访问过
        corners = state[1]
        boolean = corners[0] and corners[1] and corners[2] and corners[3]
        return boolean
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
            For a given state, this should return a list of triples, (successor,
            action, stepCost), where 'successor' is a successor to the current
            state, 'action' is the action required to get there, and 'stepCost'
            is the incremental cost of expanding to that successor
        """
        successors = []
        #遍历能够做的后续动作
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            # Add a successor state to the successor list if the action is legal
            "*** YOUR CODE HERE ***"
             #   x,y = currentPosition
            x,y = state[0]
            holdCorners = state[1]
            #   dx, dy = Actions.directionToVector(action)
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            hitsWall = self.walls[nextx][nexty]
            newCorners = ()
            nextState = (nextx, nexty)
            #不碰墙
            if not hitsWall:
                #能到达角落，四种情况判断
                if nextState in self.corners:
                    if nextState == (self.right, 1):
                        newCorners = [True, holdCorners[1], holdCorners[2], holdCorners[3]]
                    elif nextState == (self.right, self.top):
                        newCorners = [holdCorners[0], True, holdCorners[2], holdCorners[3]]
                    elif nextState == (1, self.top):
                        newCorners = [holdCorners[0], holdCorners[1], True, holdCorners[3]]
                    elif nextState == (1,1):
                        newCorners = [holdCorners[0], holdCorners[1], holdCorners[2], True]
                    successor = ((nextState, newCorners), action,  1)
                #去角落的中途
                else:
                    successor = ((nextState, holdCorners), action, 1)
                successors.append(successor)
        self._expanded += 1 
        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999.  This is implemented for you.
        """
        if actions == None: return 999999
        x,y= self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
        return len(actions)


def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

      state:   The current search state
               (a data structure you chose in your search problem)

      problem: The CornersProblem instance for this layout.

    This function should always return a number that is a lower bound on the
    shortest path from the state to a goal of the problem; i.e.  it should be
    admissible (as well as consistent).
    """
    corners = problem.corners # These are the corner coordinates
    walls = problem.walls # These are the walls of the maze, as a Grid (game.py)

    "*** YOUR CODE HERE ***"
    position = state[0]
    stateCorners = state[1]
    corners = problem.corners
    top = problem.walls.height-2
    right = problem.walls.width-2
    node = []
    for c in corners:
        if c == (1,1):
            if not stateCorners[3]:
                node.append(c)
        if c == (1, top):
            if not stateCorners[2]:
                node.append(c)
        if c == (right, top):
            if not stateCorners[1]:
                node.append(c)
        if c == (right, 1):
            if not stateCorners[0]:
                node.append(c)
    cost = 0
    currPosition = position
    while len(node) > 0:
        distArr= []
        for i in range(0, len(node)):
            dist = util.manhattanDistance(currPosition, node[i])
            distArr.append(dist)
        mindist = min(distArr)
        cost += mindist
        minDistI= distArr.index(mindist)
        currPosition = node[minDistI]
        del node[minDistI]
    return cost

class AStarCornersAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem

class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0 # DO NOT CHANGE
        self.heuristicInfo = {} # A dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1 # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x,y= self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost

class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem
        
def foodHeuristic(state, problem):
    """
    状态是一个元组（pacmanPosition，foodGrid）
    其中foodGrid是一个Grid（参见game.py）
    调用foodGrid.asList（）来获取食物坐标列表。

    如果你想存储信息，可以在其他调用中重复使用
    启发式，你可以使用一个名为problem.heuristicInfo的字典
    例如，如果您只想计算一次墙壁并存储它
    尝试：problem.heuristicInfo ['wallCount'] = problem.walls.count（）
    对此启发式的后续调用可以访问
    problem.heuristicInfo [ 'wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"
    hvalue = 0
    food_available = []
    total_distance = 0
    #处理食物的位置，以此构造启发式函数
    for i in range(0,foodGrid.width):
        for j in range(0,foodGrid.height):
            if (foodGrid[i][j] == True):
                food_location = (i,j)
                food_available.append(food_location)
    #没有食物就不用找了
    if (len(food_available) == 0):
            return 0        
    #初始化距离(current_food,select_food,distance)
    max_distance=((0,0),(0,0),0)
    for current_food in food_available:
        for select_food in food_available:
            if(current_food==select_food):
                pass
            else:
                #使用曼哈顿距离构造启发式函数
                distance = util.manhattanDistance(current_food,select_food)
                if(max_distance[2] < distance):
                    max_distance = (current_food,select_food,distance)
    #把起点和第一个搜索的食物连接起来
    #处理只有一个食物的情况
    if(max_distance[0]==(0,0) and max_distance[1]==(0,0)):
        hvalue = util.manhattanDistance(position,food_available[0])
    else: 
        d1 = util.manhattanDistance(position,max_distance[0])
        d2 = util.manhattanDistance(position,max_distance[1])
        hvalue = max_distance[2] + min(d1,d2)
    
    return hvalue

class ClosestDotSearchAgent(SearchAgent):
    "Search for all food using a sequence of searches"
    def registerInitialState(self, state):
        self.actions = []
        currentState = state
        while(currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState) # The missing piece
            self.actions += nextPathSegment
            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    t = (str(action), str(currentState))
                    raise Exception, 'findPathToClosestDot returned an illegal move: %s!\n%s' % t
                currentState = currentState.generateSuccessor(0, action)
        self.actionIndex = 0
        print 'Path found with cost %d.' % len(self.actions)

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition()
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)

        "*** YOUR CODE HERE ***"
        return search.aStarSearch(problem)
        util.raiseNotDefined()

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0
        # DO NOT CHANGE

    def isGoalState(self, state):
        x,y = state
        if self.food[x][y]:
            return True
        else:
            return False
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x,y = state
        "*** YOUR CODE HERE ***"
        foodGrid = self.food
        if (foodGrid[x][y] == True) or (foodGrid.count() == 0):
                return True
        util.raiseNotDefined()

        
##################
# Mini-contest 1 #
##################
class ApproximateSearchAgent(Agent):
    def registerInitialState(self, state):
        self.walls = state.getWalls()
        self.mark = 0
        self.curPos = state.getPacmanPosition()
        self.path = []
        self.path.append(self.curPos)
        self.starttime = time.time()
        self.path = ApproximateSearchAgent.findPath2(self,state)
        self.cost = 0
        self.DisRecord = {}
        self.mark = 0
        self.disTime = 0
        self.pT = 0
        self.m = [[0 for col in range(450)] for row in range(450)]
        ApproximateSearchAgent.initFloyed(self)
      #   print ApproximateSearchAgent.getTotalDistance(self,self.path,state)
    #    print self.path
    #     print len(self.path)
        self.path = ApproximateSearchAgent.TwoOpt(self,state)
    #    print ApproximateSearchAgent.brutalDis(self,self.path,state)
    #     print time.time() - self.starttime
        
    def initFloyed(self):
        size = len(self.path)
        for i in range(0,size):
            x,y=self.path[i]
            if (x+1,y) in self.path:
                self.m[x+y*30][x+1+y*30]=1
                self.m[x+1+y*30][x+y*30]=1
            if (x-1,y) in self.path:
                self.m[x+y*30][x-1+y*30]=1
                self.m[x-1+y*30][x+y*30]=1
            if (x,y+1) in self.path:
                self.m[x+(y+1)*30][x+y*30]=1
                self.m[x+y*30][x+(y+1)*30]=1
            if (x,y-1) in self.path:
                self.m[x+(y-1)*30][x+y*30]=1
                self.m[x+y*30][x+(y-1)*30]=1   
        for k in range(0,size):
            for i in range(0,size):
                if not(i==k):
                    for j in range(0,size):
                        if not(i==j) and not(j==k):
                            tx,ty=self.path[k]
                            pk=tx+ty*30
                            tx,ty=self.path[i]
                            pi=tx+ty*30
                            tx,ty=self.path[j]
                            pj=tx+ty*30
                            if not(self.m[pi][pk]==0) and not(self.m[pk][pj]==0):
                                if self.m[pi][pj]==0 or self.m[pi][pk]+self.m[pk][pj]<self.m[pi][pj]:
                                    self.m[pi][pj]=self.m[pi][pk]+self.m[pk][pj]
                                    self.m[pj][pi]=self.m[pi][pj]
        print(self.m[181][121])

    def findPath(self, state):
        originPath = []
        foodMap = state.getFood()
        foodMap = foodMap.asList()
        curPos = state.getPacmanPosition()
        originPath.append(curPos)
        minDis = 9999999
        nextpos = curPos
        while len(foodMap) > 0:
            minDis = 9999999
            for pos in foodMap:
                t = util.manhattanDistance(curPos,pos)
                if t < minDis:
                    minDis = t
                    nextpos = pos
            originPath.append(nextpos) 
            foodMap.remove(nextpos)
            curPos = nextpos
        return originPath

    # greedy path
    def findPath2(self, state):
        from game import Directions
        s = Directions.SOUTH
        w = Directions.WEST
        n = Directions.NORTH
        e = Directions.EAST
        originPath = []
        foodMap = state.getFood()
        unvisited = foodMap.asList()
        curPos = state.getPacmanPosition()
        originPath.append(curPos)
        while len(unvisited) > 0:
            minDis = 999999
            minMD = 999999
            for pos in unvisited:
             #   print curPos, pos
                t = util.manhattanDistance(curPos,pos)
                if t < minDis:
                    tt = mazeDistance(curPos,pos,state)
                    if tt < minMD:
                        minDis = t
                        minMD = tt
                        nextpos = pos
            
            prob = PositionSearchProblem(state, start=curPos, goal=nextpos, warn=False, visualize=False)
            move = search.bfs(prob)[0]
            x, y = curPos
            if move == s:
                y -= 1
            if move == w:
                x -= 1
            if move == n:
                y += 1
            if move == e:
                x += 1
            curPos = (x,y)
            if curPos in unvisited:
                unvisited.remove(curPos)
                originPath.append(curPos)
        return originPath


    def TwoOpt(self, state):
        size = len(self.path)
        improve = 0
        bestDis = ApproximateSearchAgent.getTotalDistance(self,self.path,state)
        while improve < 20:
            #print bestDis
            for i in range(1,size - 1):
                for k in range (i+1, min(i+size/2,size)):
                    newPath = ApproximateSearchAgent.TwoOptSwap(self,i,k)
                    newDis = ApproximateSearchAgent.newTotalDistance(self,i,k,self.path,state,bestDis)     
                    if newDis <= 285:
                        self.path = newPath
                        return newPath
                    if newDis < bestDis:
                        improve = 0
                        self.path = newPath
                        bestDis = newDis
            improve += 1
        return newPath

    def TwoOptSwap(self,i,k):
        size = len(self.path)
        ansPath = list(self.path[0:i])
        rev = list(self.path[i:k+1])
        rev.reverse()
        end = list(self.path[k+1:size])
        return ansPath + rev + end

    def newTotalDistance(self,i,k,thispath,state,oldDis):
        newDis= oldDis + ApproximateSearchAgent.getDis(self,i-1,k,state,thispath) + ApproximateSearchAgent.getDis(self,i,k+1,state,thispath)- ApproximateSearchAgent.getDis(self,i-1,i,state,thispath)- ApproximateSearchAgent.getDis(self,k,k+1,state,thispath)  
        # paht i-1,k
        return newDis



    def getDis(self,start,end,state,thispath):
        if end >= len(thispath):
            return 0
        tx,ty=thispath[start]
        p1=tx+ty*30
        tx,ty=thispath[end]
        p2=tx+ty*30
        return self.m[p1][p2]

    def getTotalDistance(self,thispath,state):
      #  dt0 = time.time()
        totalDis = 0
        for i in range(len(thispath) - 1):
            tx,ty=thispath[i]
            p1=tx+ty*30
            tx,ty=thispath[i+1]
            p2=tx+ty*30
            totalDis += self.m[p1][p2]
            """
            if self.DisRecord.has_key((thispath[i],thispath[i+1])):
                totalDis += self.DisRecord[(thispath[i],thispath[i+1])]
            else:
                self.DisRecord[(thispath[i],thispath[i+1])] = mazeDistance(thispath[i],thispath[i+1],state)
                totalDis += self.DisRecord[(thispath[i],thispath[i+1])]
            """
            #totalDis += mazeDistance(thispath[i],thispath[i+1],state)
   #     dt = time.time() - dt0
   #     self.disTime += dt
    #    print self.disTime
        return totalDis

    def brutalDis(self,thispath,state):
        totalDis = 0
        for i in range(len(thispath) - 1):
            totalDis += mazeDistance(thispath[i],thispath[i+1],state)
        return totalDis

    def getAction(self, state):
        if self.pT == 0:
          #  print self.disTime
            self.pT = 1
        curPos = state.getPacmanPosition() 
        if self.path[self.mark] == curPos:
            self.mark += 1
        nextpos = self.path[self.mark]
        prob = PositionSearchProblem(state, start=curPos, goal=nextpos, warn=False, visualize=False)
        move = (search.bfs(prob))[0]
        self.cost += 1
        #print self.cost
        #print time.time() - self.starttime 
        return move
        
def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built.  The gameState can be any game state -- Pacman's position
    in that state is ignored.
    Example usage: mazeDistance( (2,4), (5,6), gameState)
    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + point1
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.bfs(prob))
