import random
import numpy as np
import os
import pickle
from operator import itemgetter
import sys
import time
import torch

class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}".format(env_name)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity

class PrioritizedReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        np.random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.position = 0

    def push(self, state, action, reward, next_state, done, priority):
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            self.priorities.append(None)

        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
        
    
    def push_dict(self, state, action, reward, next_state, done, priority):        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            self.priorities.append(None)

        state_task_obs = state['task_obs']
        state_waypoint_obs = state['waypoints_obs']
        state_local_map = state['local_map']
        state_action = state['action']
        state_pedestrian_map = state['pedestrian_map']
        state_ped_pos_obs = state['ped_pos_obs']

        next_state_task_obs = next_state['task_obs']
        next_state_waypoint_obs = next_state['waypoints_obs']
        next_state_local_map = next_state['local_map']
        next_state_action = next_state['action']
        next_state_pedestrian_map = next_state['pedestrian_map']
        next_state_ped_pos_obs = next_state['ped_pos_obs']

        self.buffer[self.position] = (state_task_obs, state_waypoint_obs, state_local_map, state_action, state_pedestrian_map, state_ped_pos_obs,\
                                      action, reward, 
                                      next_state_task_obs, next_state_waypoint_obs, next_state_local_map, next_state_action, next_state_pedestrian_map, next_state_ped_pos_obs,\
                                        done)
        
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        #s = time.time()
        p = self.priorities / np.sum(self.priorities)
        indices = np.random.choice(len(self.buffer), batch_size, replace=False, p=p)
        #batch = np.take(self.buffer, indices)
        batch = list(itemgetter(*indices)(self.buffer))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        #e = time.time()
        #print(e-s)
        return state, action, reward, next_state, done
    
    def sample_obs_train(self, batch_size):
        #s = time.time()
        p = self.priorities / np.sum(self.priorities)
        indices = np.random.choice(len(self.buffer), batch_size, replace=False, p=p)
        #batch = np.take(self.buffer, indices)
        batch = list(itemgetter(*indices)(self.buffer))
        state_task_obs, state_waypoint_obs, state_local_map, state_action, state_pedestrian_map, state_ped_pos_obs, \
            action, reward, \
                next_state_task_obs, next_state_waypoint_obs, next_state_local_map, next_state_action, next_state_pedestrian_map, next_state_ped_pos_obs, done = map(torch.stack, zip(*batch))
        
        # e = time.time()
        # print(e-s)
        return state_task_obs, state_waypoint_obs, state_local_map, state_action, state_pedestrian_map, state_ped_pos_obs, \
            action, reward, \
                next_state_task_obs, next_state_waypoint_obs, next_state_local_map, next_state_action, next_state_pedestrian_map, next_state_ped_pos_obs, done

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}".format(env_name)
            save_path_priority = f"checkpoints/sac_buffer_priority_{env_name}"
        print('Saving buffer to {}'.format(save_path))
        print('Saving priority to {}'.format(save_path_priority))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)
        with open(save_path_priority, 'wb') as f:
            pickle.dump(self.priorities, f)

    def load_buffer(self, save_path, save_path_priority):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity

        print('Loading priority from {}'.format(save_path_priority))
        with open(save_path_priority, "rb") as f:
            self.priorities = pickle.load(f)