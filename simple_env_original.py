import numpy as np
import random
import networkx as nx
import pickle
import os
import gym
import math
import sys
import rvo2
import matplotlib as mpl
import cv2

from scipy.ndimage import rotate
from matplotlib import pyplot as plt
from PIL import Image

from gibson2.utils.utils import dist
from gibson2.utils.utils import l2_distance
from occupancy_grid.occupancy_grid import inflate_grid, visualize_path, occupancy_grid_to_graph

class Simple_env_original:
    def __init__(self, args, scene_id=None) -> None:
        self.args = args
        self.resolution = 0.1 # map resolution        
        self.num_wps_input = args.num_wps_input
        # Make action space as a gym box
        low = np.array([-1,-1])
        high = np.array([1,1])
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32, seed=123)
        self.robot_linear_velocity = 0.5
        self.robot_angular_velocity = 1.5707963267948966
        self.no_of_collisions = 0
        self.replan = False
        self.ghost_node = None
        #self.robot_linear_velocity = 1 #changed
        #self.robot_angular_velocity = 0.5 #changed
        #self.action_timestep = 0.25 # changed from 0.1
        self.action_timestep = 0.1
        self.waypoints = None
        random.seed(args.train_seed) #train
        #random.seed(12345) #evaluation
        #random.seed(args.seed)
        np.random.seed(args.train_seed) # comment out for train
        self.personal_space_violation_steps = 0

        if args.env_type == 'without_map':
            # Env bounds
            self.x_range = (-15.0, 15.0)
            self.y_range = (-15.0, 15.0)
            # Build graph for pathfinding
            path = f'graph_simple_env_size{self.x_range[1]}.pickle'
            if not os.path.exists(path):
                og_x_size = int((self.x_range[1]-self.x_range[0])/self.resolution)
                og_y_size = int((self.y_range[1]-self.y_range[0])/self.resolution)
                self.graph = nx.grid_graph((og_x_size, og_y_size))
                cost = {}
                for edge in self.graph.edges():
                    cost[edge] = np.linalg.norm(np.array(edge[0]) - np.array(edge[1]))
                nx.set_edge_attributes(self.graph, cost, "cost")
                pickle.dump(self.graph, open(path, 'wb'))
            else:
                self.graph = pickle.load(open(path, 'rb'))
        
        elif args.env_type == 'with_map':            
            # load occupancy grid
            floor_img_path = f'gibson2/data/gibson_challenge_data_2021/ig_dataset/scenes/{scene_id}/layout/floor_trav_0_new.png'
            img = Image.open(floor_img_path)
            occupancy_grid = np.array(img)
            #occupancy_grid[occupancy_grid<=127.5] = 0
            #occupancy_grid[occupancy_grid>127.5] = 1
            inflated_grid = inflate_grid(occupancy_grid, args.inflation_radius, 0, 0)
            self.occupancy_grid = occupancy_grid.astype(np.float64)
            
            self.inflated_grid = inflated_grid.astype(np.float64)
            self.trav_space_inflated_grid = np.where(self.inflated_grid == 1)
            self.inflated_grid_new = occupancy_grid_to_graph(self.inflated_grid)

            path = f'gibson2/data/gibson_challenge_data_2021/ig_dataset/scenes/{scene_id}/layout/trav_graph_inflation_radius_{args.inflation_radius}.pickle'
            # build graph for pathfinding
            if not os.path.exists(path):
                self.graph = nx.grid_graph((inflated_grid.shape[0], inflated_grid.shape[1]))
                nx.set_edge_attributes(self.graph, np.inf, "cost")

                free_space = np.vstack((np.where(inflated_grid==1)[0], np.where(inflated_grid==1)[1])).T
                free_space_nodes = tuple(map(tuple, free_space))
                free_space_nodes = {node:None for node in free_space_nodes}

                cost = {}
                for edge in self.graph.edges():
                    if edge[0] in free_space_nodes and edge[1] in free_space_nodes:
                        cost[edge] = np.linalg.norm(np.array(edge[0])-np.array(edge[1]))
                nx.set_edge_attributes(self.graph, cost, "cost")
                
                pickle.dump(self.graph, open(path, 'wb'))
            else:
                self.graph = pickle.load(open(path, 'rb'))
            
            # needed to compute robot coordinate <-> world coordinate mapping
            height,width = occupancy_grid.shape
            self.trav_map_original_size = height
            self.trav_map_default_resolution = 0.1 
            self.trav_map_size = int(self.trav_map_original_size *
                                        self.trav_map_default_resolution /
                                        self.resolution)
        
        if args.pedestrian_present:
            self.no_episode_pedestrian_collision = 0
            #self.num_pedestrians = args.num_pedestrians
            if self.args.ped_no_variable:
                self.num_pedestrians = random.randint(1, self.args.highest_num_pedestrians)
            else:
                self.num_pedestrians = self.args.num_pedestrians
            self.pedestrian_goal_thresh = self.args.pedestrian_goal_threshold
            #self.pedestrian_goal_thresh = args.orca_radius #changed
            self.closest_visible_pedestrian = None
            self.closest_visible_ped_dist = np.inf
            """
            Initialize a ORCA simulator
            Parameters for ORCA

            timeStep        The time step of the simulation.
                            Must be positive.
            neighborDist    The default maximum distance (center point
                            to center point) to other agents a new agent
                            takes into account in the navigation. The
                            larger this number, the longer the running
                            time of the simulation. If the number is too
                            low, the simulation will not be safe. Must be
                            non-negative.
            maxNeighbors    The default maximum number of other agents a
                            new agent takes into account in the
                            navigation. The larger this number, the
                            longer the running time of the simulation.
                            If the number is too low, the simulation
                            will not be safe.
            timeHorizon     The default minimal amount of time for which
                            a new agent's velocities that are computed
                            by the simulation are safe with respect to
                            other agents. The larger this number, the
                            sooner an agent will respond to the presence
                            of other agents, but the less freedom the
                            agent has in choosing its velocities.
                            Must be positive.
            timeHorizonObst The default minimal amount of time for which
                            a new agent's velocities that are computed
                            by the simulation are safe with respect to
                            obstacles. The larger this number, the
                            sooner an agent will respond to the presence
                            of obstacles, but the less freedom the agent
                            has in choosing its velocities.
                            Must be positive.
            radius          The default radius of a new agent.
                            Must be non-negative.
            maxSpeed        The default maximum speed of a new agent.
                            Must be non-negative.
            """
            #self.neighbor_dist = 3 #changed
            self.neighbor_dist = 5
            self.max_neighbors = self.num_pedestrians
            self.time_horizon = 2.0
            self.time_horizon_obst = 2.0
            self.orca_radius = args.orca_radius
            self.orca_max_speed = 0.5

            self.orca_sim = rvo2.PyRVOSimulator(
                self.action_timestep,
                self.neighbor_dist,
                self.max_neighbors,
                self.time_horizon,
                self.time_horizon_obst,
                self.orca_radius,
                self.orca_max_speed)
            
            self.num_steps_stop = [0] * self.num_pedestrians
            self.neighbor_stop_radius = 1.0
            #self.neighbor_stop_radius = 2 * self.orca_radius #changed
            self.num_steps_stop_thresh = 20
            self.backoff_radian_thresh = np.deg2rad(135.0)
            
            self.orca_pedestrians = self.load_pedestrians()
            #self.load_obstacles()

    
    def load_pedestrians(self):
        '''
        Initialize a number of pedestrians at (0,0) in ORCA simulator.
        Their actual position in initialized at the beginning of the episode.
        '''
        if self.args.robot_visible_to_pedestrians:
            self.robot_orca_ped = self.orca_sim.addAgent((0, 0))
        orca_pedestrians = []
        for i in range(self.num_pedestrians):
            orca_ped = self.orca_sim.addAgent((0, 0))
            orca_pedestrians.append(orca_ped)
        return orca_pedestrians

    def load_obstacles(self):
        # NOT BEING USED
        obstacles = list(map(tuple, np.vstack(np.where(self.occupancy_grid == 0)).T))
        obstacles_wc = list(map(self.map_to_world, map(np.array, obstacles)))
        
        for obstacle in obstacles_wc:
            # Assuming a simple rectangular representation for each obstacle
            obstacle_rectangle = [(obstacle[0], obstacle[1]),
                                        (obstacle[0] + self.resolution, obstacle[1]),
                                        (obstacle[0] + self.resolution, obstacle[1] + self.resolution),
                                        (obstacle[0], obstacle[1] + self.resolution)]
            self.orca_sim.addObstacle(obstacle_rectangle)
        self.orca_sim.processObstacles()

    def initialize_episode(self):
        self.step_number = 0

        flag = True
        dist_threshold = 10        
        if self.args.env_type == 'without_map':
            # Take robot start and goal position in less than 10m
            while flag:
                # Initialize robot starting and goal point in world and map coordinate
                self.robot_pos = (random.uniform(self.x_range[0], self.x_range[1]),\
                                random.uniform(self.y_range[0], self.y_range[1]))
                self.robot_pos_map = self.world_to_map(self.robot_pos)
                #self.robot_orientation_degree = random.uniform(-180, 180)
                self.robot_orientation_degree = random.uniform(0, 360) #changed
                self.robot_orientation_radian = np.radians(self.robot_orientation_degree)

                self.goal_pos = (random.uniform(self.x_range[0], self.x_range[1]),\
                                random.uniform(self.y_range[0], self.y_range[1]))
                self.goal_pos_map = self.world_to_map(self.goal_pos)

                #if l2_distance(self.robot_pos, self.goal_pos) <= dist_threshold:
                #    flag = False
                #else:
                #    continue

                # Compute path and generate waypoints
                path = nx.astar_path(self.graph, self.robot_pos_map, self.goal_pos_map, heuristic=dist, weight="cost")
                path_length = nx.astar_path_length(self.graph, self.robot_pos_map, self.goal_pos_map, heuristic=dist, weight="cost")
                if 1.0 <= path_length * self.resolution <= dist_threshold:
                    flag = False
                else:
                    continue
                
                self.point_interval = self.args.waypoint_interval
                p = path[::self.point_interval][1:] # First point is robot start pos
                
                if tuple(self.goal_pos_map) not in p:
                    p.append(tuple(self.goal_pos_map))
                
                self.waypoints = list(map(self.map_to_world, map(np.array, p)))
                self.goal_pos = self.waypoints[-1]
                #print("robot pos", self.robot_pos)
                #print("robot orientation", self.robot_orientation_degree)
                #print("goal pos", self.goal_pos)
                #print(self.waypoints)
                #wayypints_map = np.ones((int((self.x_range[1]-self.x_range[0])/self.resolution), int((self.y_range[1]-self.y_range[0])/self.resolution)))
                #visualize_path(wayypints_map, p, show=True)
                
            
        elif self.args.env_type == 'with_map':
            # Take robot start and goal position in less than 10m
            while flag:
                # sample robot pos, orientation
                idx = np.random.randint(0, high=self.trav_space_inflated_grid[0].shape[0])
                xy_map = np.array([self.trav_space_inflated_grid[0][idx], self.trav_space_inflated_grid[1][idx]])
                self.robot_pos_map = (xy_map[0], xy_map[1])
                
                axis = 0 if len(xy_map.shape) == 1 else 1
                self.robot_pos = np.flip((xy_map - self.trav_map_size / 2.0) * self.resolution, axis=axis)
                #self.robot_orientation_degree = random.uniform(0, 360) #changed from -180 to 180
                self.robot_orientation_degree = random.uniform(-180, 180) #changed from -180 to 180
                self.robot_orientation_radian = np.radians(self.robot_orientation_degree)
                
                # sample goal pos
                idx = np.random.randint(0, high=self.trav_space_inflated_grid[0].shape[0])
                xy_map = np.array([self.trav_space_inflated_grid[0][idx], self.trav_space_inflated_grid[1][idx]])
                self.goal_pos_map = (xy_map[0], xy_map[1])
                axis = 0 if len(xy_map.shape) == 1 else 1
                self.goal_pos = np.flip((xy_map - self.trav_map_size / 2.0) * self.resolution, axis=axis)

                #if l2_distance(self.robot_pos, self.goal_pos) <= dist_threshold:
                #    flag = False
                #else:
                #    continue

                # Compute path and generate waypoints
                path = nx.astar_path(self.graph, self.robot_pos_map, self.goal_pos_map, heuristic=dist, weight="cost")
                path_length = nx.astar_path_length(self.graph, self.robot_pos_map, self.goal_pos_map, heuristic=dist, weight="cost")
                path_cost = nx.path_weight(self.graph, path, 'cost')
                if path_cost >= np.inf:
                    continue
                # if 1.0 <= path_length * self.resolution <= dist_threshold:
                #     flag = False
                if 5 <= path_length * self.resolution <= 15:
                    flag = False
                else:
                    continue
                
                self.point_interval = self.args.waypoint_interval
                p = path[::self.point_interval][1:] # First point is robot start pos
                
                if tuple(self.goal_pos_map) not in p:
                    p.append(tuple(self.goal_pos_map))
                
                self.waypoints = list(map(self.map_to_world, map(np.array, p)))
                self.goal_pos = self.waypoints[-1]
                #print("robot pos", self.robot_pos)
                #print("robot orientation", self.robot_orientation_degree)
                #print("goal pos", self.goal_pos)
                    
        if self.args.pedestrian_present:
            self.personal_space_violation_steps = 0

            # Set robot position in ORCA simulator
            if self.args.robot_visible_to_pedestrians:
                self.orca_sim.setAgentPosition(self.robot_orca_ped, tuple(self.robot_pos))
            # Set pedestrian position in ORCA simulator
            self.reset_pedestrians()
        '''
        # Visualize waypoints
        if self.args.env_type == 'without_map':
            wayypints_map = np.ones((int((self.x_range[1]-self.x_range[0])/self.resolution), int((self.y_range[1]-self.y_range[0])/self.resolution)))
            visualize_path(wayypints_map, p, show=True)
        elif self.args.env_type == 'with_map':
            #visualize_path(self.occupancy_grid, p, show=True)
            self.visualize_map()
        '''
    def sample_initial_ped_pos(self, ped_id):
        """
        Sample a new initial position for pedestrian with ped_id.
        The inital position is sampled randomly until the position is
        at least |self.orca_radius| away from all other pedestrians' initial
        positions and the robot's initial position.
        """
        must_resample_pos = True
        while must_resample_pos:
            # sample ped pos
            idx = np.random.randint(0, high=self.trav_space_inflated_grid[0].shape[0])
            xy_map = np.array([self.trav_space_inflated_grid[0][idx], self.trav_space_inflated_grid[1][idx]])
            ped_pos_map = (xy_map[0], xy_map[1])
            ped_pos = self.map_to_world(np.array(ped_pos_map))
            
            # sample ped orientaion
            #ped_orientation_degree = random.uniform(0, 360) #changed from -180 to 180
            ped_orientation_degree = random.uniform(-180, 180)
            ped_orientation_radian = np.radians(ped_orientation_degree)
            
            must_resample_pos = False

            # If too close to the robot, resample
            if dist(ped_pos, self.robot_pos) < 2 * self.orca_radius:
                must_resample_pos = True
                continue

            # If too close to the previous pedestrians, resample
            for neighbor_id in range(ped_id):
                neighbor_ped_pos = self.orca_sim.getAgentPosition(self.orca_pedestrians[ped_id])
                if dist(neighbor_ped_pos, ped_pos) < 2 * self.orca_radius:
                    must_resample_pos = True
                    break
        
        return ped_pos, ped_pos_map, ped_orientation_radian
    
    def convert_dense_to_sparse_waypoints(self, path):
        # Convert dense waypoints of the shortest path to coarse waypoints
        # in which the collinear waypoints are merged.
        waypoints = []
        valid_waypoint = None
        prev_waypoint = None
        cached_slope = None
        for waypoint in path:
            waypoint = np.array(waypoint)
            if valid_waypoint is None:
                valid_waypoint = waypoint
            elif cached_slope is None:
                cached_slope = waypoint - valid_waypoint
            else:
                cur_slope = waypoint - prev_waypoint
                cosine_angle = np.dot(cached_slope, cur_slope) / \
                    (np.linalg.norm(cached_slope) * np.linalg.norm(cur_slope))
                if np.abs(cosine_angle - 1.0) > 1e-3:
                    waypoints.append(valid_waypoint)
                    valid_waypoint = prev_waypoint
                    cached_slope = waypoint - valid_waypoint

            prev_waypoint = waypoint

        # Add the last two valid waypoints
        waypoints.append(valid_waypoint)
        waypoints.append(path[-1])

        # Remove the first waypoint because it's the same as the initial pos
        waypoints.pop(0)

        return waypoints
    
    def sample_ped_target_pos(self, ped_pos_map):
        while True:
            idx = np.random.randint(0, high=self.trav_space_inflated_grid[0].shape[0])
            xy_map = np.array([self.trav_space_inflated_grid[0][idx], self.trav_space_inflated_grid[1][idx]])
            ped_target_map = (xy_map[0], xy_map[1])

            path = nx.astar_path(self.graph, ped_pos_map, ped_target_map, heuristic=dist, weight="cost")
            #path_cost = nx.path_weight(self.graph, path, 'cost')
            #if path_cost >= np.inf:
            #    continue
            if len(path) > 1:
                break
        waypoints = self.convert_dense_to_sparse_waypoints(path)
        waypoints = list(map(self.map_to_world, map(np.array, waypoints)))
        return waypoints
    
    def reset_pedestrians(self):
        """
        Reset the poses of pedestrians to have no collisions with the scene or the robot and set waypoints to follow.
        """
        self.pedestrian_waypoints = []
        self.pedestrians = []
        for ped_id, orca_ped in enumerate(self.orca_pedestrians):
            # sample pedestrian pos, orientation. 
            ped_pos, ped_pos_map, ped_orientation_radian = self.sample_initial_ped_pos(ped_id)
            #Get pedestrian waypoints based on sampled target pos
            ped_waypoints = self.sample_ped_target_pos(ped_pos_map)

            self.orca_sim.setAgentPosition(orca_ped, tuple(ped_pos))
            self.pedestrian_waypoints.append(ped_waypoints)
            self.pedestrians.append((ped_pos, ped_orientation_radian))

    def step(self, action):
        self.step_number += 1
        
        if self.args.consider_all_waypoint:
            self.previous_distance_to_waypoints = []
            self.previous_angle_to_waypoints = []
            for waypoint in self.waypoints[:self.args.num_wps_input]:
                self.previous_distance_to_waypoints.append(self.get_relative_pos(waypoint))
                self.previous_angle_to_waypoints.append(abs(self.get_relative_orientation(waypoint)))     

        else:
            if self.ghost_node is not None:
                waypoint = self.ghost_node
            else:
                waypoint = self.waypoints[0]
            self.previous_distance_to_waypoint = self.get_relative_pos(waypoint)
            self.previous_angle_to_waypoint = abs(self.get_relative_orientation(waypoint))

        linear_velocity = action[0] * self.robot_linear_velocity
        angular_velocity = action[1] * self.robot_angular_velocity

        self.robot_orientation_radian += (angular_velocity * self.action_timestep)
        if self.robot_orientation_radian > math.pi:
            self.robot_orientation_radian -= 2 * math.pi
        elif self.robot_orientation_radian < -math.pi:
            self.robot_orientation_radian += 2 * math.pi
        self.robot_orientation_radian = self.robot_orientation_radian % (2*math.pi)
        self.robot_orientation_degree = np.degrees(self.robot_orientation_radian)

        x = linear_velocity * np.cos(self.robot_orientation_radian) * self.action_timestep
        y = linear_velocity * np.sin(self.robot_orientation_radian) * self.action_timestep

        if self.args.env_type == 'without_map':
            collision = False
            self.robot_pos = (self.robot_pos[0]+x, self.robot_pos[1]+y)
            self.robot_pos_map = self.world_to_map(self.robot_pos)
        
        elif self.args.env_type == 'with_map':
            xy = (self.robot_pos[0]+x, self.robot_pos[1]+y)
            xy_map = self.world_to_map(np.array(xy))
            # check for collision. Don't update robot pos if wall collision.
            collision = False

            
            
            if self.occupancy_grid[xy_map[0]][xy_map[1]] == 1: # no wall collision
                self.no_of_collisions = 0
                self.robot_pos = xy
                self.robot_pos_map = xy_map
            else:
                collision = True
                self.no_of_collisions += 1

        if self.args.pedestrian_present:
            if self.args.robot_visible_to_pedestrians:
                self.orca_sim.setAgentPosition(self.robot_orca_ped, tuple(self.robot_pos))
            for i, (ped, orca_ped, waypoints) in enumerate(zip(self.pedestrians, self.orca_pedestrians, self.pedestrian_waypoints)):
                ped_current_pos = np.array(ped[0])
                if len(waypoints) == 0 or self.num_steps_stop[i] >= self.num_steps_stop_thresh:
                    waypoints = self.sample_ped_target_pos(self.world_to_map(ped_current_pos))
                    self.pedestrian_waypoints[i] = waypoints
                    self.num_steps_stop[i] = 0
                next_goal = waypoints[0]
                yaw = np.arctan2(next_goal[1] - ped_current_pos[1],
                            next_goal[0] - ped_current_pos[0])
                self.pedestrians[i] = (ped[0], yaw)
                desired_vel = next_goal - ped_current_pos[0:2]
                with np.errstate(invalid='raise'):
                    try:
                        desired_vel = desired_vel / \
                            np.linalg.norm(desired_vel) * self.orca_max_speed
                    except:
                        print("next goal", next_goal)
                        print("ped current pos", ped_current_pos)
                        print("desired val", desired_vel)
                self.orca_sim.setAgentPrefVelocity(orca_ped, tuple(desired_vel))
            
            self.orca_sim.doStep()
            next_ped_pos, next_ped_stop_flag = self.update_ped_pos_and_stop_flag()

            # Update the pedestrian position in PyBullet if it does not stop
            # Otherwise, revert back the position in RVO2 simulator
            for i, (ped, orca_ped, waypoints) in enumerate(zip(self.pedestrians, self.orca_pedestrians, self.pedestrian_waypoints)):
                ped_pos = next_ped_pos[i]
                if next_ped_stop_flag[i] is True:
                    # revert back ORCA sim pedestrian to the previous time step
                    self.num_steps_stop[i] += 1
                    self.orca_sim.setAgentPosition(orca_ped, tuple(ped_pos))
                else:
                    # advance pybullet pedstrian to the current time step
                    self.num_steps_stop[i] = 0
                    self.pedestrians[i] = (tuple(ped_pos), ped[1])
                    next_goal = waypoints[0]
                    if np.linalg.norm(next_goal - np.array(ped_pos)) \
                            <= self.pedestrian_goal_thresh:
                        waypoints.pop(0)
            
            # Detect robot's personal space violation
            personal_space_violation = False
            for ped in self.pedestrians:
                #if l2_distance(ped[0], self.robot_pos) <= 2 * self.orca_radius + 0.1: #changed
                if l2_distance(ped[0], self.robot_pos) <= self.orca_radius:
                    personal_space_violation = True
                    break
            if personal_space_violation:
                self.personal_space_violation_steps += 1


        reward, reward_type = self.get_rewards(collision)
        # if self.replan and self.ghost_node is not None:
        #     reward_type = "replan"
        #print("reward", reward)
        done, info = self.get_termination()

        # Remove waypoints if passed
        if len(self.waypoints) >= 2:
            if self.get_relative_pos(self.waypoints[1]) < self.get_relative_pos(self.waypoints[0]):
                self.waypoints.pop(0)

        return reward, reward_type, done, info
    
    def detect_collision(self, ped, orca_ped):
        pos_xy = self.orca_sim.getAgentPosition(orca_ped)
        try:
            ped_pos_map = self.world_to_map(np.array([pos_xy[0], pos_xy[1]]))       
            if self.occupancy_grid[ped_pos_map[0]][ped_pos_map[1]] == 0:
                return True
        except:
            print("error in detect_collision")
        return False

    def update_ped_pos_and_stop_flag(self):
        """
        Wrapper function that updates pedestrians' next position and whether
        they should stop for the next time step
        """
        next_ped_pos = {i: np.array(ped[0]) for i, ped in enumerate(self.pedestrians)}
        next_peds_stop_flag = [False for i in range(len(self.pedestrians))]

        for i, (ped, orca_ped, waypoints) in enumerate(zip(self.pedestrians, self.orca_pedestrians, self.pedestrian_waypoints)):
            orca_ped_pos = self.orca_sim.getAgentPosition(orca_ped)
            next_pos = np.array([orca_ped_pos[0], orca_ped_pos[1]])

            if self.detect_backoff(ped, orca_ped):
                self.stop_neighbor_pedestrians(i, next_peds_stop_flag, next_ped_pos)
            # If collides with wall
            if self.detect_collision(ped, orca_ped):
                next_peds_stop_flag[i] = True
                next_ped_pos[i] = np.array(ped[0])
            elif next_peds_stop_flag[i] is False:
                next_ped_pos[i] = next_pos

        return next_ped_pos, next_peds_stop_flag
    
    def detect_backoff(self, ped, orca_ped):
        """
        Detects if the pedestrian is attempting to perform a backoff
        due to some form of imminent collision
        """
        pos_xy = self.orca_sim.getAgentPosition(orca_ped)
        prev_pos = np.array(ped[0])

        yaw = ped[1]

        # Computing the directional vectors from yaw
        normalized_dir = np.array([np.cos(yaw), np.sin(yaw)])

        next_dir = np.array([pos_xy[0] - prev_pos[0],
                             pos_xy[1] - prev_pos[1]])

        if np.linalg.norm(next_dir) == 0.0:
            return False

        next_normalized_dir = next_dir / np.linalg.norm(next_dir)

        angle = np.arccos(np.dot(normalized_dir, next_normalized_dir))
        return angle >= self.backoff_radian_thresh

    def stop_neighbor_pedestrians(self, id, peds_stop_flags, peds_next_pos):
        """
        If the pedestrian whose instance stored in self.pedestrians with
        index |id| is attempting to backoff, all the other neighboring
        pedestrians within |self.neighbor_stop_radius| will stop
        """
        ped = self.pedestrians[id]
        ped_pos = np.array(ped[0])

        for i, neighbor in enumerate(self.pedestrians):
            if id == i:
                continue
            neighbor_pos = np.array(neighbor[0])
            if dist(ped_pos, neighbor_pos) <= self.neighbor_stop_radius:
                peds_stop_flags[i] = True
                peds_next_pos[i] = neighbor_pos
        
        peds_stop_flags[id] = True
        peds_next_pos[id] = ped_pos

    #_______________________________________________Termination Criteria___________________________________#
    def get_termination(self):
        # goal termination condition
        info = {'success':0, 'episode_return':0, 'timeout': 0}
        done = 0

        threshold = 0.1
        #threshold = 0.5 # changed
        #threshold = self.args.orca_radius # changed
        dist_to_goal = l2_distance(self.robot_pos, self.goal_pos)
        if dist_to_goal <= threshold:
            done = 1
            info['success'] = 1

        # max 500 step termination condition
        if self.step_number == self.args.episode_max_num_step:
            done =  1

        if self.args.pedestrian_present:
            info['pedestrian_collision'] = 0
            threshold = self.args.pedestrian_collision_threshold
            #threshold = 2 * self.orca_radius #changed
            for ped in self.pedestrians:
                if l2_distance(self.robot_pos, np.array(ped[0])) <= threshold:
                    done = 1
                    info['success'] = 0
                    info['pedestrian_collision'] = 1
                    self.no_episode_pedestrian_collision += 1
                    break
        if done == 1 and info["success"] == 0:
            info["timeout"] = 1
        
        # out of bound
        #if self.args.env_type == 'without_map':
        #    if not(self.x_range[0] <= self.robot_pos[0] <= self.x_range[1] and\
        #        self.y_range[0] <= self.robot_pos[1] <= self.y_range[1]):
        #        done = 1
        return done, info

    #___________________________________REWARDS____________________________________#
    def get_rewards(self, collision):
        # Goal reward
        reward = 0.0
        reward_type = "other"
        goal_reward, goal_found = self.goal_reward()
        if goal_found :
            reward_type = "goal"
            return goal_reward, reward_type
        
        # Pedestrian collision reward
        if self.args.pedestrian_present:
            ped_reward, ped_collision = self.pedestrian_collision_reward()
            if ped_collision:
                reward_type = "pedestrian collision"
                return ped_reward, reward_type 

        # wall collision
        if collision:
            reward_type = "wall collision"
            return self.collision_reward(), reward_type
        
        # waypoint reward
        waypoint_reward = self.waypoint_reward()
        if waypoint_reward > 0:
            reward += waypoint_reward
        else:
            if self.args.consider_all_waypoint:
            # sort the waypoints based on distance. Needed when all waypoints are considered
                waypoints = self.waypoints[:len(self.previous_distance_to_waypoints)]
                waypoints = sorted(enumerate(waypoints), key=lambda x: l2_distance(self.robot_pos, x[1]))
            
                # Dense potential and orientation reward
                reward += self.orientation_reward(waypoints)
                reward += self.potential_reward(waypoints)
                if self.args.pca_reward:
                    pca_reward, reward_applied = self.pedestrian_collision_avoidance_reward()
                    reward += pca_reward
                    if reward_applied:
                        reward_type = "pca"
            else:
                reward += self.orientation_reward()
                reward += self.potential_reward()
                if self.args.pca_reward:
                    pca_reward, reward_applied = self.pedestrian_collision_avoidance_reward()
                    reward += pca_reward
                    if reward_applied:
                        reward_type = "pca"
        
        # timestep reward
        reward += self.timestep_reward()
    
        return reward, reward_type
    
    def goal_reward(self):
        threshold = 0.1
        #threshold = 0.5 #changed
        #threshold = self.args.orca_radius
        
        goal_reward_weight = self.args.goal_reward

        goal_found = False
        dist_to_goal = l2_distance(self.robot_pos, self.goal_pos)
        if dist_to_goal <= threshold:
            goal_found = True
            return goal_reward_weight, goal_found 
        return 0, goal_found
    
    def pedestrian_collision_reward(self):
        threshold = self.args.pedestrian_collision_threshold
        #threshold = 2 * self.orca_radius #changed

        pedestrian_collision_reward_weight = self.args.pedestrian_collision_reward
        ped_collision = False
        
        for ped in self.pedestrians:
            if l2_distance(self.robot_pos, np.array(ped[0])) <= threshold:
                ped_collision = True
                return pedestrian_collision_reward_weight, ped_collision
        return 0, ped_collision

    def pedestrian_collision_avoidance_reward(self):
        #threshold = 3*self.orca_radius
        #threshold = 0.6
        threshold = self.args.pca_threshold
        pedestrian_collision_threshold = self.args.pedestrian_collision_threshold
        weight = self.args.pca_reward_value
        reward = 0 

        reward_applied = False
        if self.closest_visible_ped_dist <= threshold:
            #print("pca", -weight * self.closest_visible_ped_dist)
            reward_applied = True
            reward = -weight * (threshold - self.closest_visible_ped_dist) / (threshold - pedestrian_collision_threshold)
        return reward, reward_applied

    def orientation_reward(self, waypoints=None):
        orientation_reward_weight = 0.3
        #orientation_reward_weight = 0.5 #changed

        threshold = 0.1
        #threshold = 0.5 #changed

        if self.args.consider_all_waypoint:
            reward = 0
            norm_factor = 0
            for i, (index, waypoint) in enumerate(waypoints):
                angle = abs(self.get_relative_orientation(waypoint))
                weight = (len(waypoints)-i) / len(waypoints)
                #reward += (-angle * weight)
                reward += (self.previous_angle_to_waypoints[index] - angle) * weight
                #norm_factor += math.pi * weight
                norm_factor += (self.robot_angular_velocity * self.action_timestep) * weight
            
            return reward * orientation_reward_weight / norm_factor

        else:           
            if self.ghost_node is None:
                waypoint = self.waypoints[0]
            else:
                waypoint = self.ghost_node
            angle = abs(self.get_relative_orientation(waypoint))

            #return orientation_reward_weight * (np.radians(15)-angle) / math.pi
            #print("orientation reward", orientation_reward_weight * -angle / math.pi)
            return orientation_reward_weight * -angle / math.pi
            #return orientation_reward_weight * (self.previous_angle_to_waypoint - angle) / (self.robot_angular_velocity * self.action_timestep)
        
        

    def potential_reward(self, waypoints=None):
        potential_reward_weight = 0.3
        threshold = 0.1
        #threshold = 0.5 #changed
        reward = 0
        if self.args.consider_all_waypoint:
            
            norm_factor = 0
            for i, (index, waypoint) in enumerate(waypoints):
                dist = l2_distance(self.robot_pos, waypoint)
                weight = (len(waypoints)-i) / len(waypoints)
                reward += (self.previous_distance_to_waypoints[index] - dist) * weight
                norm_factor += (self.robot_linear_velocity * self.action_timestep) * weight
            
            return reward * potential_reward_weight / norm_factor

        else:
            if self.ghost_node is None:
                waypoint = self.waypoints[0]
            else:
                waypoint = self.ghost_node
            dist = l2_distance(self.robot_pos, waypoint)
            #print("potential", potential_reward_weight * (self.previous_distance_to_waypoint - dist) / (self.robot_linear_velocity * self.action_timestep))
            return potential_reward_weight * (self.previous_distance_to_waypoint - dist) / (self.robot_linear_velocity * self.action_timestep)

    def waypoint_reward(self):
        waypoint_reward = 0.8 # changed
        
        threshold = 0.1
        #threshold = 0.5 #changed
        #threshold = self.args.orca_radius
        # if self.ghost_node is not None:
        #     if l2_distance(self.ghost_node, self.robot_pos) <= threshold:
        #         self.waypoints.pop(0)
        #         self.ghost_node = None
        #         return waypoint_reward
        flag = False
        for i, waypoint in enumerate(self.waypoints[:3]):
            if l2_distance(waypoint, self.robot_pos) <= threshold:
                flag = True
                index = i+1
                break
        if flag:
            for i in range(index):
                self.waypoints.pop(0)
            return waypoint_reward
        
        
        #waypoint = self.waypoints[0]
        #if l2_distance(waypoint, self.robot_pos) <= threshold:
        #    self.waypoints.pop(0)
        #    return waypoint_reward
        
        return 0
        
    
    def collision_reward(self):
        if self.args.env_type == 'with_map':
            #return -0.3
            return -10
        
    def timestep_reward(self):
        return -0.001

    def map_to_world(self, point):
        if self.args.env_type == 'without_map':
            x = (point[0] * self.resolution) + self.x_range[0]
            y = (point[1] * self.resolution) + self.y_range[0]

            return (x,y)
        elif self.args.env_type == 'with_map':
            axis = 0 if len(point.shape) == 1 else 1
            xy = np.flip((point - self.trav_map_size / 2.0) * self.resolution, axis=axis)
            return (xy[0],xy[1])
        
    def world_to_map(self, point):
        if self.args.env_type == 'without_map':
            x = int((point[0] - self.x_range[0]) / self.resolution)
            y = int((point[1] - self.y_range[0]) / self.resolution)
        elif self.args.env_type == 'with_map':
            point = np.array(point)
            xy = np.flip((point / self.resolution + (self.trav_map_size / 2.0))).astype(np.int32)
            return (xy[0],xy[1])
        return (x,y)
    
    def get_relative_pos(self, point):
        return np.linalg.norm(np.array(self.robot_pos) - np.array(point))
    
    def get_relative_orientation(self, point):
        p = np.array(point) - np.array(self.robot_pos)
        angle = np.arctan2(p[1], p[0])

        #print("fff")
        #print(point)
        #print(self.robot_pos)
        #print(angle)
        #print("fff")
        #print(self.robot_orientation_radian)
        #print("fff", angle - self.robot_orientation_radian)
        delta_theta = angle - self.robot_orientation_radian #changed
        if delta_theta > math.pi:
            delta_theta -= 2 * math.pi
        elif delta_theta < -math.pi:
            delta_theta += 2 * math.pi
    
        return delta_theta
        #return angle - self.robot_orientation_radian
    
    def get_geodesic_distance(self, point1, point2):
        path_length = nx.astar_path_length(self.graph, point1, point2, heuristic=dist, weight="cost")
        return path_length * self.resolution

    
    def visualize_map(self):
        plt.figure()
        plt.imshow(self.occupancy_grid)
        
        arrow = u'$\u2191$'
        rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
        if self.args.env_type == "without_map":
            marker_rotation_angle = self.robot_orientation_degree-90
        elif self.args.env_type == "with_map":
            marker_rotation_angle = -self.robot_orientation_degree-90

        rotated_marker._transform = rotated_marker.get_transform().rotate_deg(marker_rotation_angle)
        waypoints_in_map = list(map(self.world_to_map, self.waypoints))
        for waypoint in waypoints_in_map:
            plt.plot(waypoint[1], waypoint[0], marker='o')
        plt.scatter((self.robot_pos_map[1]), (self.robot_pos_map[0]), marker=rotated_marker, facecolors='none', edgecolors='b', s = 50)

        
        if self.args.pedestrian_present:
            colors = ['r','g','y']
            for i, ped in enumerate(self.pedestrians):
                ped_pos = self.world_to_map(np.array(ped[0]))
                rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
                marker_rotation_angle = -np.degrees(ped[1])-90
                rotated_marker._transform = rotated_marker.get_transform().rotate_deg(marker_rotation_angle)
                plt.scatter(ped_pos[1], ped_pos[0], marker=rotated_marker, facecolors='none', edgecolors=colors[i], s=50)
                
                waypoints = self.pedestrian_waypoints[i]
                waypoints = list(map(self.world_to_map, waypoints))
                for waypoint in waypoints:
                    plt.plot(waypoint[1], waypoint[0], marker='*', color=colors[i])
                if self.ghost_node is not None:
                    ghost_node_map = self.world_to_map(self.ghost_node)
                    plt.plot(ghost_node_map[1], ghost_node_map[0], marker='^')
                #plt.plot(waypoints[1], waypoints[0], marker="o", markersize=5, alpha=0.8, color=colors[i])
        plt.show()
        



    
