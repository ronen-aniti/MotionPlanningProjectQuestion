from faulthandler import cancel_dump_traceback_later
import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue

class Map:
    """
    The general map class, from which specific map representations are derived
    """
    def __init__(self, filename, title, safety, global_home, current_local_position, goal_local_position):
        self.filename = filename
        self.title = title
        self.safety = safety
        self.global_home = global_home
        self.current_local_position = current_local_position
        self.current_altitude = self.current_local_position[2]
        self.goal_local_position = goal_local_position
        self.goal_altitude = self.goal_local_position[2]
        # Command the drone to takeoff to the goal altitude.
        self.data = np.loadtxt(self.filename, delimiter=',', skiprows=2)
        self.ned_boundaries, self.map_size = self.take_measurements()
        self.elevation_map = self.compute_elevation_map()
        self.current_grid_location = self.determine_grid_location(self.current_local_position)
        self.goal_grid_location = self.determine_grid_location(self.goal_local_position)

    def determine_grid_location(self, current_local_position):
        """
        Determine current grid location given a local position
        """

        # First, unpack the northing and easting deltas from current local position
        northings, eastings, _ = current_local_position

        # Second, determine the grid north and east coordinates.
        ## Subtract away the grid's northing and easting offsets in order to do so.
        grid_north = int(northings - self.ned_boundaries[0])
        grid_east = int(eastings - self.ned_boundaries[2])
        
        # Return the drone's current grid position
        return (grid_north, grid_east)



    def take_measurements(self):
        """
        Scans the csv map file and returns the NED boundaries relative to global home
        """
        
        # First, calculate NED boundaries
        ned_boundaries = [0, 0, 0, 0, 0, 0]
        ned_boundaries[0] = int(np.floor(np.amin(self.data[:,0] - self.data[:,3])) - self.safety) # North min
        ned_boundaries[1] = int(np.ceil(np.amax(self.data[:,0] + self.data[:,3])) + self.safety) # North max
        ned_boundaries[2] = int(np.floor(np.amin(self.data[:,1] - self.data[:,4])) - self.safety) # East min
        ned_boundaries[3] = int(np.ceil(np.amax(self.data[:,1] + self.data[:,4])) + self.safety) # East max
        ned_boundaries[4] = 0 # Alt min
        ned_boundaries[5] = int(np.ceil(np.amax(self.data[:,2] + self.data[:,5])) + self.safety) # Alt max

        # Second, calculate the size of the map
        map_size = [0, 0, 0]
        map_size[0] = ned_boundaries[1] - ned_boundaries[0]
        map_size[1] = ned_boundaries[3] - ned_boundaries[2]
        map_size[2] = ned_boundaries[5] - ned_boundaries[4]

        # Third, return the calculated quantities
        return ned_boundaries, map_size

    def compute_elevation_map(self):
        """
        Compute a '2.5d' map of the drone's environment
        """

        # First, initialize a grid of zeros
        elevation_map = np.zeros((self.map_size[0], self.map_size[1]))
        
        # Second, build a 2.5d grid representation of the drone's environment
        obstacle_boundaries = [0, 0, 0, 0]
        ## Iterate through the map data file to do this
        for i in range(self.data.shape[0]):
            north, east, down, d_north, d_east, d_down = self.data[i, :]
            height = down + d_down
            obstacle = [
                int(north - self.ned_boundaries[0] - d_north - self.safety),
                int(north - self.ned_boundaries[0] + d_north + self.safety),
                int(east - self.ned_boundaries[2] - d_east - self.safety),
                int(east - self.ned_boundaries[2] + d_east + self.safety)
            ]
            elevation_map[obstacle_boundaries[0] : obstacle_boundaries[1]+1, obstacle_boundaries[2] : obstacle_boundaries[2] : obstacle_boundaries[3]+1] = height - self.ned_boundaries[4]

        # Third, return the 2.5d map
        return elevation_map


class GridMap(Map):
    def __init__(self, filename, title, safety, global_home, current_local_position, goal_local_position):
        super().__init__(filename, title, safety, global_home, current_local_position, goal_local_position)
        self.grid = self.compute_grid()
        

    def compute_grid(self):
        """
        Build a grid to represent the drone's environment
        """
        
        # First, initialize a grid of zeros
        grid = np.zeros((self.map_size[0], self.map_size[1]), dtype='float64')
        
        # Second, build a grid representation of the drone's environment
        obstacle_boundaries = [0, 0, 0, 0]
        ## Iterate through the map data file to do this
        for i in range(self.data.shape[0]):
            north, east, down, d_north, d_east, d_down = self.data[i, :]
            if (down + d_down) > self.goal_altitude:
                obstacle_boundaries = [
                    int(north - self.ned_boundaries[0] - d_north - self.safety),
                    int(north - self.ned_boundaries[0] + d_north + self.safety),
                    int(east - self.ned_boundaries[2] - d_east - self.safety),
                    int(east - self.ned_boundaries[2] + d_east + self.safety)
                ]
                grid[obstacle_boundaries[0] : obstacle_boundaries[1]+1, obstacle_boundaries[2] : obstacle_boundaries[3]+1] = 1.0

        # Third, return the grid map
        return grid
    
    def search_grid(self):
        """
        Generates a position command sequence that will bring the drone from start to goal
        """

        print('Searching for a path...')
        
        current_local_position = self.current_local_position
        current_local_position[2] = self.goal_altitude # The first waypoint will be at the target altitude
        current_north, current_east = self.current_grid_location
        goal_north, goal_east = self.goal_grid_location

        # First, raise an excpetion if the start or end grid cells are occupied
        if self.grid[current_north, current_east] == 1:
            raise Exception('Invalid start node')
        if self.grid[goal_north, goal_east] == 1:
            raise Exception('Invalid goal node')


        # Second, begin an A* search routine
        frontier = PriorityQueue()
        visited = set()
        frontier.put((0.0, self.current_grid_location))
        visited.add(self.current_grid_location)
        travel_cost = 0.0
        grid_sequence = []
        found = False
        pathinfo = {}

        ## Repeat this subroutine until a path from start to goal is found
        while not frontier.empty():
            
            _, gridcell = frontier.get()

            if gridcell == self.goal_grid_location:
                found = True
                print('Found a path.')
                break

            free_neighbors = self.explore_free_neighbors(gridcell)
            for free_neighbor in free_neighbors:
                candidate_cell = free_neighbor[0]
                candidate_north, candidate_east = candidate_cell
                heuristic_cost = np.sqrt((candidate_north - self.goal_grid_location[0])**2 + (candidate_east - self.goal_grid_location[1])**2)
                action_cost = free_neighbor[1] 
                incremental_cost = action_cost + heuristic_cost
                candidate_total_cost = travel_cost + incremental_cost
                if candidate_cell not in visited:
                    frontier.put((candidate_total_cost, candidate_cell))
                    pathinfo[candidate_cell] = (gridcell, action_cost)
                    visited.add(candidate_cell)
        
        # Third, once the goal gridcell has been found, generate a sequence of gricells from star to goal
        if found:
            subgoal = self.goal_grid_location
            origin, action_cost = pathinfo[subgoal]
            grid_sequence.append(subgoal)
            while origin != self.current_grid_location:
                subgoal = origin
                origin, action_cost = pathinfo[subgoal]
                grid_sequence.append(subgoal)
            grid_sequence.append(origin)
            grid_sequence = grid_sequence[::-1]

            
            
            ## Remove collinear gridcells from the waypoint sequence
            grid_sequence = self.remove_collinear(grid_sequence)

            ## Plot the updated grid sequence
            #self.plot_path(grid_sequence)

            ## Covert the grid sequence into a waypoint sequence
            waypoint_commands = [self.grid_to_waypoint(gridcell) for gridcell in grid_sequence]
            print(waypoint_commands)
            ## Return the waypoint sequence
            return waypoint_commands

        else:
            raise Exception('Failed to find a path.')

    
    def remove_collinear(self, grid_sequence):
        """
        Removes collinear gridcells
        """
        i = 0 
        while i+2 < len(grid_sequence):
            x1 = grid_sequence[i][0]
            y1 = grid_sequence[i][1]
            x2 = grid_sequence[i+1][0]
            y2 = grid_sequence[i+1][1]
            x3 = grid_sequence[i+2][0]
            y3 = grid_sequence[i+2][1]

            collinear = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2) == 0

            if collinear:
                del(grid_sequence[i+1])
            else:
                i += 1
        return grid_sequence


    def grid_to_waypoint(self, gridcell):
        """
        Converts a grid cell into a waypoint command
        """
        waypoint = np.array([])
        waypoint = np.append(waypoint, gridcell[0] + self.ned_boundaries[0])
        waypoint = np.append(waypoint, gridcell[1] + self.ned_boundaries[2])
        waypoint = np.append(waypoint, self.goal_altitude)
        waypoint = np.append(waypoint, 0)

        return list(waypoint)




    def explore_free_neighbors(self, gridcell):
        """
        Returns a gridcell's free neighbors along with the cost of getting to each
        """
        
        # First, define the action set
        actions = {}
        actions['LEFT'] = (0, -1, 1)
        actions['RIGHT'] = (0, 1, 1)
        actions['UP'] = (-1, 0, 1)
        actions['DOWN'] = (1, 0, 1)
        actions['NORTHEAST'] = (-1, 1, np.sqrt(2))
        actions['NORTHWEST'] = (1, 1, np.sqrt(2))
        actions['SOUTHEAST'] = (1, 1, np.sqrt(2))
        actions['SOUTHWEST'] = (1, -1, np.sqrt(2))
        
        grid_north, grid_east = gridcell

        free_neighbors = []
        for value in actions.values():
            north_delta, east_delta, action_cost = value
            candidate_cell_north = grid_north + north_delta
            candidate_cell_east = grid_east + east_delta
            candidate_cell = (candidate_cell_north, candidate_cell_east)
            if candidate_cell_north >= 0 and candidate_cell_north < self.map_size[0] and candidate_cell_east >= 0 and candidate_cell_east < self.map_size[1]:
                if self.grid[candidate_cell_north, candidate_cell_east] == 0.0:
                    free_neighbors.append((candidate_cell, action_cost))


        return free_neighbors

    def plot_grid(self):
        plt.imshow(self.grid, origin='lower', cmap='Greys')
        plt.title(self.title)
        plt.xlabel('Eastings (m)')
        plt.ylabel('Northings (m)')
        plt.show()

    def plot_path(self, grid_sequence):
        """
        Plots the path from start to goal
        """

        grid_sequence_north = []
        grid_sequence_east = []
        for grid_cell in grid_sequence:
            grid_sequence_north.append(grid_cell[0])
            grid_sequence_east.append(grid_cell[1])
        
        plt.imshow(self.grid, origin='lower', cmap='Greys')
        plt.plot(grid_sequence_east, grid_sequence_north, color='blue', linestyle='-', marker='o')
        plt.title(self.title)
        plt.xlabel('Eastings (m)')
        plt.ylabel('Northings (m)')
        plt.show()
            

           



class MedialAxisGridMap(Map):
    pass

class VoxelMap(Map):
    pass

class VoronoiMap(Map):
    pass

class PRMMap(Map):
    pass

class RRTMap(Map):
    pass

class PotentialFieldMap(Map):
    pass

