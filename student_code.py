# Do not change this cell
# When you write your methods correctly this cell will execute
# without problems
import math
class PathPlanner():
    """Construct a PathPlanner Object"""
    def __init__(self, M, start=None, goal=None):
        """ """
        self.map = M
        self.start= start
        self.goal = goal
        self.closedSet = self.create_closedSet() if goal != None and start != None else None
        self.openSet = self.create_openSet() if goal != None and start != None else None
        self.cameFrom = self.create_cameFrom() if goal != None and start != None else None
        self.gScore = self.create_gScore() if goal != None and start != None else None
        self.fScore = self.create_fScore() if goal != None and start != None else None
        self.path = self.run_search() if self.map and self.start != None and self.goal != None else None
        
    def get_path(self):
        """ Reconstructs path after search """
        if self.path:
            return self.path 
        else :
            self.run_search()
            return self.path
    
    def reconstruct_path(self, current):
        """ Reconstructs path after search """
        total_path = [current]
        while current in self.cameFrom.keys():
            current = self.cameFrom[current]
            total_path.append(current)
        return total_path
    
    def _reset(self):
        """Private method used to reset the closedSet, openSet, cameFrom, gScore, fScore, and path attributes"""
        self.closedSet = None
        self.openSet = None
        self.cameFrom = None
        self.gScore = None
        self.fScore = None
        self.path = self.run_search() if self.map and self.start and self.goal else None

    def run_search(self):
        """ """
        if self.map == None:
            raise(ValueError, "Must create map before running search. Try running PathPlanner.set_map(start_node)")
        if self.goal == None:
            raise(ValueError, "Must create goal node before running search. Try running PathPlanner.set_goal(start_node)")
        if self.start == None:
            raise(ValueError, "Must create start node before running search. Try running PathPlanner.set_start(start_node)")

        self.closedSet = self.closedSet if self.closedSet != None else self.create_closedSet()
        self.openSet = self.openSet if self.openSet != None else  self.create_openSet()
        self.cameFrom = self.cameFrom if self.cameFrom != None else  self.create_cameFrom()
        self.gScore = self.gScore if self.gScore != None else  self.create_gScore()
        self.fScore = self.fScore if self.fScore != None else  self.create_fScore()

        while not self.is_open_empty():
            current = self.get_current_node()

            if current == self.goal:
                self.path = [x for x in reversed(self.reconstruct_path(current))]
                return self.path
            else:
#                self.openSet.remove(current)
                self.closedSet.add(current)

            for neighbor in self.get_neighbors(current):
                if neighbor in self.closedSet:
                    continue    # Ignore the neighbor which is already evaluated.

                if not neighbor in self.openSet:    # Discover a new node
                    self.openSet.add(neighbor)
                
                # The distance from start to a neighbor
                #the "dist_between" function may vary as per the solution requirements.
                if self.get_tenative_gScore(current, neighbor) >= self.get_gScore(neighbor):
                    continue        # This is not a better path.

                # This path is the best until now. Record it!
                self.record_best_path_to(current, neighbor)  
            self.openSet.remove(current)
        print("No Path Found")
        self.path = None
        return False
########################################################################################################################################    
    def set_map(self, M):
        """Method used to set map attribute """
        self._reset(self)
        self.start = None
        self.goal = None
        self.M = M
    # TODO: Set map to new value. - DONE DONE DONE
    
    def set_start(self, start):
        """Method used to set start attribute """
        self._reset(self)
        self.goal = None
        self.closedSet = None
        self.openSet = None
        self.cameFrom = None
        self.gScore = None
        self.fScore = None
        self.path = None
        self.start = start
        # TODO: Set start value. Remember to remove goal, closedSet, openSet, cameFrom, gScore, fScore,
        # and path attributes' values.  - DONE DONE DONE
        
    def set_goal(self, goal):
        """Method used to set goal attribute """
        self._reset(self)
        self.goal = goal
        # TODO: Set goal value.  - DONE DONE DONE
        
    def create_closedSet(self):
        """ Creates and returns a data structure suitable to hold the set of nodes already evaluated"""
        # TODO: return a data structure suitable to hold the set of nodes already evaluated  - DONE DONE DONE
        closedSet = set()
        return closedSet
    
    def create_openSet(self):
        """ Creates and returns a data structure suitable to hold the set of currently discovered nodes
        that are not evaluated yet. Initially, only the start node is known."""
        
        
        if self.start != None:
            openSet =set()
            openSet.add(self.start)
            # TODO: return a data structure suitable to hold the set of currently discovered nodes  - DONE DONE DONE
            # that are not evaluated yet. Make sure to include the start node.
            return openSet
        raise(ValueError, "Must create start node before creating an open set. Try running PathPlanner.set_start(start_node)")
        
    def create_cameFrom(self):
        """Creates and returns a data structure that shows which node can most efficiently be reached from another,
        for each node."""
        # TODO: return a data structure that shows which node can most efficiently be reached from another,  - DONE DONE DONE
        # for each node. 
        cameFrom = {}
        #m = self.map
        #cameFrom[self.start] = m.roads[self.start]
        return cameFrom
        
    def create_gScore(self):
        """Creates and returns a data structure that holds the cost of getting from the start node to that node, for each node.
        The cost of going from start to start is zero."""
        # TODO:  a data structure that holds the cost of getting from the start node to that node, for each node.
        # for each node. The cost of going from start to start is zero. The rest of the node's values should be set to infinity.
        gScore = {} 
        m = self.map
        ii = len(m.intersections)
        for i in range(ii):
            gScore[i] = float('inf')
        gScore[self.start] = 0
        return gScore
        
        
    def create_fScore(self):
        """Creates and returns a data structure that holds the total cost of getting from the start node to the goal
        by passing by that node, for each node. That value is partly known, partly heuristic.
        For the first node, that value is completely heuristic."""
        # TODO:  a data structure that holds the total cost of getting from the start node to the goal
        # by passing by that node, for each node. That value is partly known, partly heuristic.
        # For the first node, that value is completely heuristic. The rest of the node's value should be 
        # set to infinity.
        fScore = {} 
        m = self.map
        ii = len(m.intersections)
        for i in range(ii):
            fScore[i] = float('inf')
        fScore[self.start] = 0 
        return fScore
        
    def get_current_node(self):
        """ Returns the node in the open set with the lowest value of f(node)."""
        # TODO: Return the node in the open set with the lowest value of f(node). - DONE DONE DONE
        #self.openSet[]
        OS = self.openSet
        temp = {}
        for i in OS:
            temp[i] = self.calculate_fscore(i)
        node_no = min(temp, key = temp.get)
                
        return node_no
        
    def get_neighbors(self, node):
        """Returns the neighbors of a node"""
        # TODO: Return the neighbors of a node - DONE DONE DONE
        m = self.map
        
        return set(m.roads[node])
        
        
        
    def get_gScore(self, node):
        """Returns the g Score of a node"""
        # TODO: Return the g Score of a node - DOUBTFUL DONE DONE DONE
        g = self.gScore.get(node,0.0)
        return g
        
        
    def get_tenative_gScore(self, current, neighbor):
        """Returns the tenative g Score of a node"""
        # TODO: Return the g Score of the current node - DONE DONE DONE
        # plus distance from the current node to it's neighbors
        g1 = self.get_gScore(current)
        g2 = self.distance(current,neighbor)
        g = g1 + g2
        
        return g
        
        
    def is_open_empty(self):
        """returns True if the open set is empty. False otherwise. """
        # TODO: Return True if the open set is empty. False otherwise.  - DONE DONE DONE
        return len(self.openSet) == 0
              
    def distance(self, node_1, node_2):
        """ Computes the Euclidean L2 Distance"""
        # TODO: Compute and return the Euclidean L2 Distance  - DONE DONE DONE
        m = self.map
        x1,y1 = m.intersections[node_1]
        x2,y2 = m.intersections[node_2]
        d = math.sqrt(math.pow((x1-x2),2)+math.pow((y1-y2),2))
        return d
        
        
    def heuristic_cost_estimate(self, node):
        """ Returns the heuristic cost estimate of a node """
        # TODO: Return the heuristic cost estimate of a node  - DONE DONE DONE
        h = self.distance(node,self.goal)
        return h
        
    def calculate_fscore(self, node):
        """Calculate the f score of a node. """
        # TODO: Calculate and returns the f score of a node. - DONE DONE DONE
        # REMEMBER F = G + H
        g = self.get_gScore(node)
        h = self.heuristic_cost_estimate(node)
        f = g + h
        return f
        
        
    def record_best_path_to(self, current, neighbor):
        """Record the best path to a node """
        # TODO: Record the best path to a node, by updating cameFrom, gScore, and fScore
        m = self.map
        self.cameFrom[neighbor] = current
        self.gScore[neighbor]=self.get_tenative_gScore(current,neighbor)
        self.fScore[current]=self.calculate_fscore(current)
        return current
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        