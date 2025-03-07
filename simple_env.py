import numpy as np
import random
import networkx as nx
from gibson2.utils.utils import dist
import gym
import math
from gibson2.utils.utils import l2_distance
import pickle
import os
from matplotlib import pyplot as plt
from occupancy_grid.occupancy_grid import visualize_path
from PIL import Image
from occupancy_grid.occupancy_grid import inflate_grid
import sys

class Simple_env:
    def __init__(self, args, scene_id=None) -> None:
        self.args = args
        self.resolution = 0.1 # map resolution        
        self.num_wps_input = args.num_wps_input
        # Make action space as a gym box
        low = np.array([-1,-1])
        high = np.array([1,1])
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)
        #self.robot_linear_velocity = 0.5
        #self.robot_angular_velocity = 1.5707963267948966
        self.robot_linear_velocity = 1 #changed
        self.robot_angular_velocity = 0.5 #changed
        self.action_timestep = 0.25 # changed from 0.1
        self.waypoints = None
        random.seed(args.train_seed) #train 123
        #random.seed(1234) #evaluation
        #random.seed(args.seed)

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
            occupancy_grid[occupancy_grid!=0] = 1
            inflated_grid = inflate_grid(occupancy_grid, args.inflation_radius, 0, 0)
            self.occupancy_grid = occupancy_grid
            self.inflated_grid = inflated_grid

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

                if l2_distance(self.robot_pos, self.goal_pos) <= dist_threshold:
                    flag = False
                else:
                    continue
                
                # Compute path and generate waypoints
                path = nx.astar_path(self.graph, self.robot_pos_map, self.goal_pos_map, heuristic=dist, weight="cost")
                #point_interval = 5
                point_interval = 10 #changed
                p = path[::point_interval][1:] # First point is robot start pos
                
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
            trav_space = np.where(self.inflated_grid == 1)
            # Take robot start and goal position in less than 10m
            while flag:
                # sample robot pos, orientation
                idx = np.random.randint(0, high=trav_space[0].shape[0])
                xy_map = np.array([trav_space[0][idx], trav_space[1][idx]])
                self.robot_pos_map = (xy_map[0], xy_map[1])
                #print(self.robot_pos_map)
                
                axis = 0 if len(xy_map.shape) == 1 else 1
                self.robot_pos = np.flip((xy_map - self.trav_map_size / 2.0) * self.resolution, axis=axis)
                self.robot_orientation_degree = random.uniform(0, 360) #changed from -180 to 180
                self.robot_orientation_radian = np.radians(self.robot_orientation_degree)
                #print(self.robot_pos)
                #sys.exit()
                # sample goal pos
                idx = np.random.randint(0, high=trav_space[0].shape[0])
                xy_map = np.array([trav_space[0][idx], trav_space[1][idx]])
                self.goal_pos_map = (xy_map[0], xy_map[1])
                axis = 0 if len(xy_map.shape) == 1 else 1
                self.goal_pos = np.flip((xy_map - self.trav_map_size / 2.0) * self.resolution, axis=axis)

                if l2_distance(self.robot_pos, self.goal_pos) <= dist_threshold:
                    flag = False
                else:
                    continue

                # Compute path and generate waypoints
                path = nx.astar_path(self.graph, self.robot_pos_map, self.goal_pos_map, heuristic=dist, weight="cost")
                point_interval = 5
                #point_interval = 10 #changed
                p = path[::point_interval][1:] # First point is robot start pos
                
                if tuple(self.goal_pos_map) not in p:
                    p.append(tuple(self.goal_pos_map))
                
                self.waypoints = list(map(self.map_to_world, map(np.array, p)))
                self.goal_pos = self.waypoints[-1]
                #print("robot pos", self.robot_pos)
                #print("robot orientation", self.robot_orientation_degree)
                #print("goal pos", self.goal_pos)
                #print(self.waypoints)

                # Visualize waypoints
                '''
                if self.args.env_type == 'without_map':
                    wayypints_map = np.ones((int((self.x_range[1]-self.x_range[0])/self.resolution), int((self.y_range[1]-self.y_range[0])/self.resolution)))
                    visualize_path(wayypints_map, p, show=True)
                elif self.args.env_type == 'with_map':
                    visualize_path(self.occupancy_grid, p, show=True)
                '''
    
    def step(self, action):
        self.step_number += 1
        self.previous_distance_to_waypoint = self.get_relative_pos(self.waypoints[0])

        linear_velocity = action[0] * self.robot_linear_velocity
        angular_velocity = action[1] * self.robot_angular_velocity

        x = linear_velocity * np.cos(self.robot_orientation_radian) * self.action_timestep
        y = linear_velocity * np.sin(self.robot_orientation_radian) * self.action_timestep

        if self.args.env_type == 'without_map':
            collision = False
            self.robot_pos = (self.robot_pos[0]+x, self.robot_pos[1]+y)
            self.robot_pos_map = self.world_to_map(self.robot_pos)

            self.robot_orientation_radian += (angular_velocity * self.action_timestep)
            self.robot_orientation_radian = self.robot_orientation_radian % (2*math.pi)
            self.robot_orientation_degree = np.degrees(self.robot_orientation_radian)
        
        elif self.args.env_type == 'with_map':
            xy = (self.robot_pos[0]+x, self.robot_pos[1]+y)
            xy_map = self.world_to_map(np.array(xy))
            # check for collision. Don't update robot pos if collision.
            collision = False
            if self.occupancy_grid[xy_map[0]][xy_map[1]] != 0: # no collision
                self.robot_pos = xy
                self.robot_pos_map = xy_map

                self.robot_orientation_radian += (angular_velocity * self.action_timestep)
                self.robot_orientation_radian = self.robot_orientation_radian % (2*math.pi)
                self.robot_orientation_degree = np.degrees(self.robot_orientation_radian) 
            else:
                collision = True

        reward = self.get_rewards(collision)
        done, info = self.get_termination()

        return reward, done, info
    
    def get_termination(self):
        # goal termination condition
        info = {'success':0, 'episode_return':0}
        done = 0
        #threshold = 0.1
        threshold = 0.5 # changed
        dist_to_goal = l2_distance(self.robot_pos, self.goal_pos)
        if dist_to_goal <= threshold:
            done = 1
            info['success'] = 1
        # max 500 step termination condition
        if self.step_number == 500:
            done =  1
        # out of bound
        #if self.args.env_type == 'without_map':
        #    if not(self.x_range[0] <= self.robot_pos[0] <= self.x_range[1] and\
        #        self.y_range[0] <= self.robot_pos[1] <= self.y_range[1]):
        #        done = 1
        return done, info

    def get_rewards(self, collision):
        # Goal reward
        reward = 0.0
        reward += self.goal_reward()
        if reward > 0:
            return reward
        if collision:
            return self.collision_reward()
        reward += self.orientation_reward()
        reward += self.potential_reward()
        reward += self.waypoint_reward()
        reward += self.timestep_reward()
        
        return reward
    
    def goal_reward(self):
        #threshold = 0.1
        threshold = 0.5 #changed
        dist_to_goal = l2_distance(self.robot_pos, self.goal_pos)
        if dist_to_goal <= threshold:
            #return 10
            return 1 #changed
        return 0

    def orientation_reward(self):
        orientation_reward_weight = -3
        #threshold = 0.1
        threshold = 0.5 #changed
        waypoint = self.waypoints[0]
        angle = abs(self.get_relative_orientation(waypoint))

        if l2_distance(self.robot_pos, waypoint) > threshold:
            return orientation_reward_weight * angle / math.pi
        return 0

    def potential_reward(self):
        potential_reward_weight = 3
        #threshold = 0.1
        threshold = 0.5 #changed
        waypoint = self.waypoints[0]
        dist = l2_distance(self.robot_pos, waypoint)
        if dist > threshold:
            return potential_reward_weight * (self.previous_distance_to_waypoint - dist) / (self.robot_linear_velocity * self.action_timestep)
        return 0

    def waypoint_reward(self):
        #waypoint reward = 0.1
        waypoint_reward = 0.8 # changed
        #threshold = 0.1
        threshold = 0.5 #changed
        waypoint = self.waypoints[0]
        if l2_distance(waypoint, self.robot_pos) <= threshold:
            self.waypoints.pop(0)
            return waypoint_reward
        return 0
    
    def collision_reward(self):
        if self.args.env_type == 'with_map':
            return -0.3

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
            xy = np.flip((point / self.resolution + self.trav_map_size / 2.0)).astype(np.int)
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


    
