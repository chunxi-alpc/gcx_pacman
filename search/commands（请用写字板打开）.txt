python2 pacman.py
python2 pacman.py --layout testMaze --pacman GoWestAgent
python2 pacman.py --layout tinyMaze --pacman GoWestAgent
python2 pacman.py -h
python2 pacman.py -l tinyMaze -p SearchAgent -a fn=tinyMazeSearch
python2 pacman.py -l tinyMaze -p SearchAgent
python2 pacman.py -l mediumMaze -p SearchAgent
python2 pacman.py -l bigMaze -z .5 -p SearchAgent
python2 pacman.py -l mediumMaze -p SearchAgent -a fn=ucs
python2 pacman.py -l bigMaze -p SearchAgent -a fn=ucs -z .5
python2 eightpuzzle.py
python2 pacman.py -l mediumMaze -p SearchAgent -a fn=ucs
python2 pacman.py -l mediumDottedMaze -p StayEastSearchAgent
python2 pacman.py -l mediumScaryMaze -p StayWestSearchAgent
python2 pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic 
python2 pacman.py -l tinyCorners -p SearchAgent -a fn=ucs,prob=CornersProblem
python2 pacman.py -l mediumCorners -p SearchAgent -a fn=ucs,prob=CornersProblem
python2 pacman.py -l mediumCorners -p AStarCornersAgent -z 0.5
python2 pacman.py -l testSearch -p AStarFoodSearchAgent
python2 pacman.py -l trickySearch -p AStarFoodSearchAgent
python2 pacman.py -l bigSearch -p ClosestDotSearchAgent -z .5 
python2 pacman.py -l bigSearch -p ApproximateSearchAgent -z .5 -q 
