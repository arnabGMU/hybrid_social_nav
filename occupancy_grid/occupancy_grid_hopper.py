import numpy as np
import networkx as nx
import scipy
import math
import copy
import cv2
import sys

from scipy.ndimage import rotate
from matplotlib import pyplot as plt
from frontier.frontier import show_frontiers, Frontier

from collections import defaultdict

def visualize_occupancy_grid(occupancy_grid, robot_position, goal, store=False, show=False, output_path=None):
    plt.plot(robot_position[0], robot_position[1], marker="o", markersize=10, alpha=0.8)
    plt.plot(goal[0], goal[1], marker="o", markersize=10, alpha=0.8)
    plt.imshow(occupancy_grid, cmap='gray') #plt.show()
    if show == True:
        plt.show()
    if store == True:
        plt.savefig(output_path)
    plt.close()

# def get_robot_pos_on_grid(robot_position_wc, min_x, min_y, RESOLUTION=0.05):
#     if robot_position_wc is None:
#         return None
    
#     robot_x = ((robot_position_wc[0] - min_x) / RESOLUTION).astype(int)
#     robot_y = ((robot_position_wc[1] - min_y) / RESOLUTION).astype(int)    
    
#     return (robot_y, robot_x)

def get_turn_angle(angle, yaw):
    yaw *= -1
    angle = angle - yaw
    if -math.pi < angle <= math.pi:
        return -angle
    elif angle > math.pi:
        return math.pi*2 - angle
    elif angle <= -math.pi:
        return -angle-(math.pi*2)
     
def update_map(occupancy_grid, occupancy_grid_prev, robot_pos_prev, prev_robot_pos_current_map):
    if occupancy_grid_prev is None:
        return occupancy_grid
    # Calculate the robot position from the previous occupancy grid to the current occupancy map
    i_current = prev_robot_pos_current_map[0] - robot_pos_prev[0]
    j_current = prev_robot_pos_current_map[1] - robot_pos_prev[1]

    i_current_end = min(occupancy_grid.shape[0], occupancy_grid_prev.shape[0] + i_current)
    j_current_end = min(occupancy_grid.shape[1], occupancy_grid_prev.shape[1] + j_current)

    # store the already occupied and free space indices from the current observation
    free_space = occupancy_grid == 1
    occupied = occupancy_grid == 0

    # Using the previous robot position index in the current map, overlap the previous map to the current.
    occupancy_grid[i_current:i_current_end, j_current:j_current_end] = \
        occupancy_grid_prev[:i_current_end-i_current, :j_current_end-j_current]
    
    # Reapply the current observation values
    occupancy_grid[free_space] = 1
    occupancy_grid[occupied] = 0

    return occupancy_grid

    unknown_current_map = occupancy_grid == 0.5
    mask_map = np.zeros(occupancy_grid.shape)
    mask_map[i_current:i_current_end, j_current:j_current_end] = 1
    relevent_map = np.logical_and(unknown_current_map, mask_map)
    index = np.where(relevent_map==1)
    occupancy_grid[index] = occupancy_grid_prev[index[0]-i_current, index[1]-j_current]

    index = np.where(occupancy_grid_prev==0)
    #a = np.logical_and(index[0]+i_current < occupancy_grid.shape[0], index[1]+j_current < occupancy_grid.shape[1])
    #occupancy_grid[index[0][a]+i_current, index[1][a]+j_current] = occupancy_grid[index[0][a], index[1][a]]    
    
    for i in range(index[0].size):
        try:
            occupancy_grid[index[0][i]+i_current, index[1][i]+j_current] = occupancy_grid_prev[index[0][i], index[1][i]]
        except:
            pass
    
    return occupancy_grid
    i_prev = 0

    while i_current < i_current_end and i_current < occupancy_grid.shape[0]:
        j_curr = j_current
        j_prev = 0
        while j_curr < j_current_end and j_curr < occupancy_grid.shape[1]:
            # If current cell is unknown or prev cell is occupied
            # replace it with whatever prev map has in it
            if occupancy_grid[i_current][j_curr] == 0.5 or occupancy_grid_prev[i_prev][j_prev] == 0:
                occupancy_grid[i_current][j_curr] = occupancy_grid_prev[i_prev][j_prev]
                #print(f'i curr j curr {i_current, j_curr} ijprev {i_prev, j_prev}')
            
            j_curr += 1
            j_prev += 1
        
        i_current += 1
        i_prev += 1
    
    return occupancy_grid
        


# OBSERVATIONS
# ---------------------------------------------------------------------------------------------------------------
def get_lidar(env):
    robot_pos = env.robot_pos_map
    yaw = env.robot_orientation_radian
    fov_angle = 128
    fov = math.radians(fov_angle)
    max_distance = int(env.args.depth/env.resolution)

    #visible_cells = np.ones_like(env.occupancy_grid) * 0.5
    x = robot_pos[1]
    y = robot_pos[0]
    lidar_measurements = [max_distance*env.resolution] * fov_angle
    #l = [5] * 128
    for i, angle in enumerate(np.linspace(-fov/2, fov/2, int(fov_angle))):
        relative_angle = yaw + angle
        dx = math.cos(relative_angle)
        dy = math.sin(relative_angle)
        for r in np.linspace(0, max_distance, int(max_distance)):  # stepping in small increments
            ix = int(x + dx * r)
            iy = int(y + dy * r)
            
            if ix < 0 or iy < 0 or ix >= env.occupancy_grid.shape[1] or iy >= env.occupancy_grid.shape[0]:
                # Out of bounds
                break

            if env.occupancy_grid[iy][ix] == 0:
                # Hit an obstacle
                lidar_measurements[i] = dist(robot_pos, (iy,ix)) * env.resolution
                break
    
    return lidar_measurements, None, None

def get_local_map_raycast(env, validation=None):
    robot_pos = env.robot_pos_map
    yaw = env.robot_orientation_radian
    fov_angle = env.args.fov
    fov = math.radians(fov_angle)
    max_distance = int(env.args.depth/env.resolution)

    # Add pedestrian position in the occupancy grid
    occupancy_grid = env.occupancy_grid.copy()
    if env.args.pedestrian_present:
        for ped in env.pedestrians:
            ped_pos_map = env.world_to_map(np.array(ped[0]))
            occupancy_grid = cv2.circle(occupancy_grid, ped_pos_map[::-1], int(env.orca_radius/env.resolution), 0, -1)

    # Raycast to get the visible cells
    visible_cells = np.ones_like(occupancy_grid) * 0.5
    obstacle_cell_threshold = int(env.orca_radius/env.resolution) * 2 + 2
    for angle in np.linspace(-fov/2, fov/2, int(1.5*fov_angle)):
        ray_casting(visible_cells, robot_pos[1], robot_pos[0], yaw + angle, occupancy_grid, max_distance, obs_cell_threshold=obstacle_cell_threshold)
    
    # Crop grid around the robot. 
    map_dim = 2 * int(env.args.depth/env.resolution)

    t_r = robot_pos[0]-map_dim//2
    top_row = max(0, t_r)

    b_r = robot_pos[0]+map_dim//2
    bottom_row = min(env.occupancy_grid.shape[0], b_r)

    l_c = robot_pos[1]-map_dim//2
    left_col = max(0, l_c)

    r_c = robot_pos[1]+map_dim//2
    right_col = min(env.occupancy_grid.shape[1], r_c)

    map_cut = visible_cells[top_row:bottom_row, left_col:right_col]

    # Cropped size might not be (map_dim,map_dim). 
    # Overlay cropped grid on a grid of size (map_dim, map_dim) of unknown values
    local_map = np.ones((map_dim, map_dim)) * 0.5
    # r = abs(t_r) if t_r < 0 else 0
    # c = abs(l_c) if l_c < 0 else 0
    r = map_dim//2 - (robot_pos[0]-top_row)
    c = map_dim//2 - (robot_pos[1]-l_c)
    local_map[r: r + map_cut.shape[0], c: c + map_cut.shape[1]] = map_cut

    # Rotate grid
    rotated_grid = rotate(local_map, np.degrees(yaw), reshape=False, mode='constant', cval=0.5, prefilter=True)
    # Change all the cell values to either free, occupied or unknown
    rotated_grid[rotated_grid<=0.4] = 0
    rotated_grid[rotated_grid>=0.6] = 1
    uk_mask = np.logical_and(rotated_grid!=0, rotated_grid!=1)
    rotated_grid[uk_mask] = 0.5

    # Create pedestrian map and get the closest pedestrian (if pedestrian is visible then mark those cells)         
    if env.args.obs_pedestrian_map:
        p_map = np.ones((map_dim, map_dim), dtype=float)
        flag = False        
        env.closest_visible_pedestrian = None 
        env.closest_visible_ped_dist = np.inf
        visible_pedestrians = []

        for ped in env.pedestrians:
            ped_pos_map = env.world_to_map(np.array(ped[0]))

            if visible_cells[ped_pos_map[0], ped_pos_map[1]] == 0: # ped is visible to robot
                flag = True # visible pedestrian found
                visible_pedestrians.append(ped)

                # Get the rotated pedestrian position and mark it in the map
                ped_dist = env.get_relative_pos(ped[0])
                ped_angle = env.get_relative_orientation(ped[0])
                # rotated_ped_pos = np.array((env.robot_pos[0] + ped_dist*np.cos(ped_angle), env.robot_pos[1] + ped_dist*np.sin(ped_angle)))
                # rotated_ped_pos_map = env.world_to_map(rotated_ped_pos)
                rotated_ped_pos_map = np.array((map_dim//2 + int((ped_dist*np.sin(ped_angle))/env.resolution), \
                                                map_dim//2 + int((ped_dist*np.cos(ped_angle))/env.resolution)))

                p_map = cv2.circle(p_map, rotated_ped_pos_map[::-1], int(env.orca_radius/env.resolution), 0, -1)

                # Find the closest pedestrian
                if ped_dist < env.closest_visible_ped_dist:
                    env.closest_visible_ped_dist = ped_dist
                    env.closest_visible_pedestrian = ped
                #pedestrian_map = cv2.circle(pedestrian_map, ped_pos_map[::-1], 2, 1, -1)
                #pedestrian_map_ = cv2.circle(pedestrian_map_, ped_pos_map[::-1], 2, 0, -1)

        # Get a (map_dim, map_dim) pedestrian map        
        # p_map_cut = p_map[top_row:bottom_row, left_col:right_col]
        # pedestrian_map = np.ones((map_dim, map_dim))
        # pedestrian_map[r: r + p_map_cut.shape[0], c: c + p_map_cut.shape[1]] = p_map_cut
        pedestrian_map = p_map
    else:
        pedestrian_map = None

    # REPLAN
    path_found = None
    replan_map = None
    replan_map_vis = None
    if env.args.replan_if_collision:            
        slack = 0.2
        replan_needed = False   
        unavoidable_collision_distance_to_waypoint = env.args.pedestrian_collision_threshold #- self.args.waypoint_reach_threshold

        # Check if any visible pedestrian is within unavoidable_collision_distance_to_ waypoints
        for ped in visible_pedestrians:
            for wp in env.waypoints[:env.args.num_wps_input]:
                if dist(ped[0], wp) <= unavoidable_collision_distance_to_waypoint + slack:
                    replan_needed = True
                    break

        if replan_needed:
            replan_map = env.inflated_grid.copy().astype(np.float64) 
            replan_map_vis = env.occupancy_grid.copy().astype(np.float64) 
            if env.args.replan_map == "gaussian":
                for ped in visible_pedestrians:
                    replan_map = get_gaussian_inflated_pedestrian_map(env, ped, replan_map)
                    ped_pos_map = env.world_to_map(np.array(ped[0]))
                    replan_map = cv2.circle(replan_map, ped_pos_map[::-1], \
                                            int((unavoidable_collision_distance_to_waypoint+slack)/env.resolution), 0, -1)      

                    replan_map_vis = get_gaussian_inflated_pedestrian_map(env, ped, replan_map_vis)  
                    # replan_map_vis = cv2.circle(replan_map, ped_pos_map[::-1], \
                    #                         int((unavoidable_collision_distance_to_waypoint+slack)/env.resolution), 0, -1)                
            else:    
                for ped in visible_pedestrians:
                    ped_pos_map = env.world_to_map(np.array(ped[0]))
                    replan_map = cv2.circle(replan_map, ped_pos_map[::-1], int((unavoidable_collision_distance_to_waypoint + slack)/env.resolution), 0, -1)
            env.replan = True

            path, path_found = plan(env, replan_map, visible_pedestrians)
            #env.visualize_map()
            if path_found:
                if path_cost_map(path) * env.resolution * 1.03 < (env.args.episode_max_num_step - env.step_number) * env.robot_linear_velocity * env.action_timestep:
                    point_interval = env.args.waypoint_interval
                    if len(path) > point_interval:
                        p = path[::point_interval][1:]
                    else:
                        p = path
                    env.waypoints = list(map(env.map_to_world, map(np.array, p)))
                    env.goal_pos = env.waypoints[-1]
                    #env.visualize_map()
            else:
                pass
                #print("path not found")

    #plt.figure();plt.imshow(pedestrian_map);plt.show()
    #print("r",np.unique(rotated_grid))
    #print("p",np.unique(pedestrian_map))
    
    # pedestrian_map = np.zeros_like(env.occupancy_grid, dtype=float) 
    # if env.args.obs_pedestrian_map:
    #     flag = False        
    #     env.closest_visible_pedestrian = None 
    #     env.closest_visible_ped_dist = np.inf  
    #     for ped in env.pedestrians:
    #         ped_pos_map = env.world_to_map(np.array(ped[0]))
    #         if visible_cells[ped_pos_map[0], ped_pos_map[1]] == 0: # ped is visible to robot
    #             try:
    #                 flag = True
    #                 ped_dist = env.get_relative_pos(ped[0])
    #                 if ped_dist < env.closest_visible_ped_dist:
    #                     env.closest_visible_ped_dist = ped_dist
    #                     env.closest_visible_pedestrian = ped

    #                 pedestrian_map = cv2.circle(pedestrian_map, ped_pos_map[::-1], 2, 1, -1)
    #             except:
    #                 print(ped_pos_map[::-1])
        
    #     ped_map_cut = pedestrian_map[top_row:bottom_row, left_col:right_col]

    #     ped_map = np.zeros((map_dim, map_dim))
    #     r = abs(t_r) if t_r < 0 else 0
    #     c = abs(l_c) if l_c < 0 else 0
    #     ped_map[r: r + ped_map_cut.shape[0], c: c + ped_map_cut.shape[1]] = ped_map_cut

    #     # Rotate grid
    #     ped_rotated_grid = rotate(ped_map, np.degrees(yaw), reshape=False, mode='constant', cval=0, prefilter=True)

    #     ped_rotated_grid[ped_rotated_grid<0.2] = 0
    #     ped_rotated_grid[ped_rotated_grid>=0.2] = 1

    #     pedestrian_map = ped_rotated_grid
    
    # print("p",np.unique(pedestrian_map))
    # if flag:
    #     plt.figure()
    #     plt.imshow(occupancy_grid)

    #     m = np.zeros_like(env.occupancy_grid, dtype=float)
    #     m[env.occupancy_grid == 1] = 0.5
    #     m[visible_cells == 1] = 1

    #     plt.figure()
    #     plt.imshow(rotated_grid)

    #     plt.figure()
    #     plt.imshow(pedestrian_map)

    #     plt.figure()
    #     plt.imshow(m)
    #     plt.plot(robot_pos[1], robot_pos[0],marker="o", markersize=2, alpha=0.8)
    #     plt.show()
    #occupancy_grid[visible_cells==1] = 5
    return rotated_grid, pedestrian_map, path_found, visible_pedestrians, visible_cells, replan_map, replan_map_vis

def get_simple_local_map(env, first_episode=None, global_map=None, validation=None):  
    # Cut (map_dim x map_dim) matrix from the occupancy grid centered around the robot
    map_dim = 2 * int(env.args.depth/env.resolution) # 100 x 100   
    robot_pos = env.robot_pos_map
    yaw = env.robot_orientation_radian

    t_r = robot_pos[0]-map_dim//2
    top_row = max(0, t_r)

    b_r = robot_pos[0]+map_dim//2
    bottom_row = min(env.occupancy_grid.shape[0], b_r)

    l_c = robot_pos[1]-map_dim//2
    left_col = max(0, l_c)

    r_c = robot_pos[1]+map_dim//2
    right_col = min(env.occupancy_grid.shape[1], r_c)

    occupancy_grid = env.occupancy_grid.copy()
    if env.args.pedestrian_present:
        for ped in env.pedestrians:
            ped_pos_map = env.world_to_map(np.array(ped[0]))
            try:
                occupancy_grid = cv2.circle(occupancy_grid, ped_pos_map[::-1], int(env.orca_radius/env.resolution), 0, -1)
            except:
                print(ped_pos_map[::-1])
    map_cut = occupancy_grid[top_row:bottom_row, left_col:right_col]

    # if global_map:
    #     if first_episode == True:
    #         self.prev_global_map = None
    #     else:
    #         self.prev_global_map = self.global_map
    #     self.global_map[top_row:bottom_row, left_col:right_col] = map_cut
        
    # Overlap the partial map on a (map_dim x map_dim) zero np array
    partial_map = np.ones((map_dim, map_dim)) * 0.5
    # r = abs(t_r) if t_r < 0 else 0
    # c = abs(l_c) if l_c < 0 else 0
    r = map_dim//2 - (robot_pos[0]-top_row)
    c = map_dim//2 - (robot_pos[1]-l_c)
    partial_map[r: r + map_cut.shape[0], c: c + map_cut.shape[1]] = map_cut        
    
    # # Roate the occupancy grid by the robot's orientation (East facing)
    # rotated_grid = rotate(map, np.degrees(env.robot_orientation_radian), reshape=True, mode='nearest')
    # # Rotated grid might be larger than 100x100. So make it 100x100 centered around the robot
    # row_top = rotated_grid.shape[0]//2 - map_dim//2
    # row_bottom = rotated_grid.shape[0]//2 + map_dim//2
    # col_left = rotated_grid.shape[1]//2 - map_dim//2
    # col_right = rotated_grid.shape[1]//2 + map_dim//2
    # rotated_grid = rotated_grid[row_top: row_bottom, col_left:col_right]
    rotated_grid = rotate(partial_map, np.degrees(yaw), reshape=False, mode='constant', cval=0.5, prefilter=True)

    rotated_grid[rotated_grid<=0.4] = 0
    rotated_grid[rotated_grid>=0.6] = 1
    uk_mask = np.logical_and(rotated_grid!=0, rotated_grid!=1)
    rotated_grid[uk_mask] = 0.5

    # Create pedestrian map (if pedestrian is visible then mark those cells)   
    if env.args.obs_pedestrian_map:
        #p_map = np.ones_like(env.occupancy_grid, dtype=float)
        p_map = np.ones((map_dim, map_dim))
        flag = False        
        env.closest_visible_pedestrian = None 
        env.closest_visible_ped_dist = np.inf
        visible_pedestrians = []

        for ped in env.pedestrians:
            ped_pos_map = env.world_to_map(np.array(ped[0]))

            if top_row <= ped_pos_map[0] < bottom_row and left_col <= ped_pos_map[1] < right_col: #ped is visible
                
                visible_pedestrians.append(ped)
            
                ped_dist = env.get_relative_pos(ped[0])
                ped_angle = env.get_relative_orientation(ped[0])
                #rotated_ped_pos = np.array((env.robot_pos[0] + ped_dist*np.cos(ped_angle), env.robot_pos[1] + ped_dist*np.sin(ped_angle)))
                rotated_ped_pos_map = np.array((map_dim//2 + int((ped_dist*np.sin(ped_angle))/env.resolution), \
                                                map_dim//2 + int((ped_dist*np.cos(ped_angle))/env.resolution)))
                #rotated_ped_pos_map = env.world_to_map(rotated_ped_pos)
                p_map = cv2.circle(p_map, rotated_ped_pos_map[::-1], int(env.orca_radius/env.resolution), 0, -1)

                if ped_dist < env.closest_visible_ped_dist:
                    env.closest_visible_ped_dist = ped_dist
                    env.closest_visible_pedestrian = ped
                
        # p_map_cut = p_map[top_row:bottom_row, left_col:right_col]
        # pedestrian_map = np.ones((map_dim, map_dim))
        # pedestrian_map[r: r + p_map_cut.shape[0], c: c + p_map_cut.shape[1]] = p_map_cut
        pedestrian_map = p_map
    else:
        pedestrian_map = None
    
    # REPLAN
    path_found = None
    replan_map = None
    if env.args.replan_if_collision and validation is None:
        slack = 0.2            
        replan_needed = False   
        unavoidable_collision_distance_to_waypoint = env.args.pedestrian_collision_threshold #- self.args.waypoint_reach_threshold

        # Check if any visible pedestrian is within unavoidable_collision_distance_to_ first waypoints
        for ped in visible_pedestrians:
            for wp in env.waypoints[:env.args.num_wps_input]:
                if dist(ped[0], wp) <= unavoidable_collision_distance_to_waypoint + slack:
                    replan_needed = True
                    break

        # # Check if any visible pedestrian is within unavoidable_collision_distance_to_ first 3 waypoints
        # for ped in env.pedestrians:
        #     ped_pos_map = env.world_to_map(np.array(ped[0]))
        #     if top_row <= ped_pos_map[0] < bottom_row and left_col <= ped_pos_map[1] < right_col: #ped is visible
        #         for waypoint in env.waypoints[:1]:                 
        #             if dist(ped[0], waypoint) <= unavoidable_collision_distance_to_waypoint + slack:
        #                 replan_needed = True
        #                 break
        #         if replan_needed:
        #             break
        if replan_needed:
            # Mark visible cells in the inflated grid
            replan_map = env.inflated_grid.copy().astype(np.float64) 
            if env.args.replan_map == "gaussian":
                for ped in visible_pedestrians:
                    replan_map = get_gaussian_inflated_pedestrian_map(env, ped, replan_map)
                    ped_pos_map = env.world_to_map(np.array(ped[0]))
                    replan_map = cv2.circle(replan_map, ped_pos_map[::-1], \
                                            int((unavoidable_collision_distance_to_waypoint+slack-0.1)/env.resolution), 0, -1)                    
            else:    
                for ped in visible_pedestrians:
                    ped_pos_map = env.world_to_map(np.array(ped[0]))
                    replan_map = cv2.circle(replan_map, ped_pos_map[::-1], int((unavoidable_collision_distance_to_waypoint + slack)/env.resolution), 0, -1)
            env.replan = True

            path, path_found = plan(env, replan_map, visible_pedestrians)
            #env.visualize_map()
            if path_found:
                point_interval = env.args.waypoint_interval
                if path_cost_map(path) * env.resolution * 1.03 < (env.args.episode_max_num_step - env.step_number) * env.robot_linear_velocity * env.action_timestep:
                    if len(path) > point_interval:
                        p = path[::point_interval][1:]
                    else:
                        p = path
                    # p = path[int(len(path)/2)]
                    # p = np.array(env.map_to_world(np.array(p)))
                    # env.ghost_node = p

                    #if tuple(env.goal_pos_map) not in p:
                    #    p.append(tuple(env.goal_pos_map))

                    env.waypoints = list(map(env.map_to_world, map(np.array, p)))
                    env.goal_pos = env.waypoints[-1]
                #env.visualize_map()
            else:
                env.ghost_node = None
        else:
            env.replan = False
            env.ghost_node = None
    
    # path_found = None
    # if self.args.replan_if_collision and validation == None:
    #     replan_map = env.inflated_grid.copy() 
    #     replan_needed = False   
    #     unavoidable_collision_distance_to_waypoint = self.args.pedestrian_collision_threshold #- self.args.waypoint_reach_threshold

    #     # Check if any visible pedestrian is within unavoidable_collision_distance_to_ first 3 waypoints
    #     for ped in env.pedestrians:
    #         ped_pos_map = env.world_to_map(np.array(ped[0]))
    #         if top_row <= ped_pos_map[0] < bottom_row and left_col <= ped_pos_map[1] < right_col: #ped is visible
    #             for waypoint in env.waypoints[:1]:                 
    #                 if dist(ped[0], waypoint) <= unavoidable_collision_distance_to_waypoint+0.2:
    #                     replan_needed = True
    #                     break
    #             if replan_needed:
    #                 break
    #     if replan_needed:
    #         # Mark visible cells in the inflated grid
    #         for ped in env.pedestrians:
    #             ped_pos_map = env.world_to_map(np.array(ped[0]))
    #             if top_row <= ped_pos_map[0] < bottom_row and left_col <= ped_pos_map[1] < right_col: #ped is visible
    #                 replan_map = cv2.circle(replan_map, ped_pos_map[::-1], int((unavoidable_collision_distance_to_waypoint + 0.2)/env.resolution), 0, -1)

    #         # Make the robot current position in the inflated grid to be free space.
    #         robot_pos_map = np.zeros(replan_map.shape)
    #         robot_pos_map = cv2.circle(robot_pos_map, robot_pos[::-1], math.ceil(self.args.inflation_radius), 1, -1)
    #         robot_pos_map = np.logical_and(robot_pos_map==1, env.occupancy_grid==1)
    #         # if np.any(robot_pos_map):
    #         #     env.visualize_map()
    #         replan_map[robot_pos_map] = 1

    #         # replan
    #         path, path_found = self.plan(env, replan_map)
    #         #env.visualize_map()
    #         if path_found:
    #             point_interval = 10 #changed
    #             p = path[::point_interval][1:]

    #             if tuple(env.goal_pos_map) not in p:
    #                 p.append(tuple(env.goal_pos_map))

    #             env.waypoints = list(map(env.map_to_world, map(np.array, p)))
    #             env.goal_pos = env.waypoints[-1]
    #             env.visualize_map()
    #         else:
    #             pass

    # Plot grid
    
    # plt.figure()
    # plt.subplot(1,3,1)
    # plt.imshow(occupancy_grid, cmap='gray')
    # plt.xlabel("orientation" + str(-env.robot_orientation_degree))
    # plt.plot(robot_pos[1], robot_pos[0], marker='o')
    # plt.subplot(1,3,2)
    # plt.imshow(rotated_grid)
    # plt.subplot(1,3,3)
    # plt.imshow(pedestrian_map)
    # plt.show()
    
    #t_end = time.time()
    #print("simple local",t_end - t_start)
    return rotated_grid, pedestrian_map, path_found, visible_pedestrians

#---------------------------------------------------------------------------------------------------------


# REPLAN
def plan(env, ped_inflated_grid, visible_pedestrians=None):
    path_found = False
    slack = 0.2
    if len(env.waypoints) == 1: 
        return None, path_found
    
    inflated_grid_new = copy.deepcopy(env.inflated_grid_new)
    # Get the robot current position cells in the grid
    robot_pos_map = np.zeros(env.occupancy_grid.shape)
    robot_pos_map = cv2.circle(robot_pos_map, env.robot_pos_map[::-1], math.ceil(env.args.inflation_radius), 1, -1)
    robot_pos_map = np.logical_and(robot_pos_map==1, env.occupancy_grid==1)
    robot_cells_index = np.vstack((np.where(robot_pos_map==1)[0], np.where(robot_pos_map==1)[1])).T 
    robot_cells_nodes = tuple(map(tuple, robot_cells_index))
    robot_cells_nodes = {node:None for node in robot_cells_nodes}

    # Robot current position might be occupied in the inflated grid. Before path planning mark
    # the robot current position as free in the inflated grid 
    for robot_cell in robot_cells_nodes.keys():
        if (robot_cell[0], robot_cell[1]+1) in robot_cells_nodes:
            inflated_grid_new.add_edge(robot_cell, (robot_cell[0], robot_cell[1]+1), weight=1)
        if (robot_cell[0]+1, robot_cell[1]) in robot_cells_nodes:
            inflated_grid_new.add_edge(robot_cell, (robot_cell[0]+1, robot_cell[1]), weight=1)
        if (robot_cell[0]+1, robot_cell[1]+1) in robot_cells_nodes:
            inflated_grid_new.add_edge(robot_cell, (robot_cell[0]+1, robot_cell[1]+1), weight=1.41)  
        if (robot_cell[0]-1, robot_cell[1]-1) in robot_cells_nodes:
            inflated_grid_new.add_edge(robot_cell, (robot_cell[0]-1, robot_cell[1]-1), weight=1.41)

    # Get the cells that are occupied by visible pedestrians
    ped_cells = np.logical_and(env.inflated_grid==1, ped_inflated_grid==0)
    ped_cells_index = np.vstack((np.where(ped_cells==1)[0], np.where(ped_cells==1)[1])).T 
    ped_cells_nodes = tuple(map(tuple, ped_cells_index))
    ped_cells_nodes = {node:None for node in ped_cells_nodes}

    # Mark the cells as occupied by pedestrians in inflated grid    
    for ped_cell in ped_cells_nodes.keys():
        inflated_grid_new.remove_node(ped_cell)
        
    if env.args.replan_map == "gaussian":
        inflated_ped_cells = np.logical_and(ped_inflated_grid!=0, ped_inflated_grid!=1)
        #plt.figure();plt.imshow(inflated_ped_cells);plt.show()
        inflated_ped_cells_index = np.vstack((np.where(inflated_ped_cells==1)[0], np.where(inflated_ped_cells==1)[1])).T 
        inflated_ped_cells_nodes = tuple(map(tuple, inflated_ped_cells_index))
        inflated_ped_cells_nodes = {node:None for node in inflated_ped_cells_nodes}
        # Mark the cells as occupied by pedestrians in inflated grid
        for inflated_ped_cell in inflated_ped_cells_nodes.keys():
            if (inflated_ped_cell[0], inflated_ped_cell[1]+1) in inflated_ped_cells_nodes:
                # inflated_grid_new.add_node(inflated_ped_cell)
                # inflated_grid_new.add_node((inflated_ped_cell[0], inflated_ped_cell[1]+1))

                w = ped_inflated_grid[inflated_ped_cell] + ped_inflated_grid[(inflated_ped_cell[0], inflated_ped_cell[1]+1)]
                inflated_grid_new.add_edge(inflated_ped_cell, (inflated_ped_cell[0], inflated_ped_cell[1]+1), weight=(1+7*(1-w)))
                #inflated_grid_new.add_edge(inflated_ped_cell, (inflated_ped_cell[0], inflated_ped_cell[1]+1), weight=(2-w))
                #print(2-w)
            
            if (inflated_ped_cell[0]+1, inflated_ped_cell[1]) in inflated_ped_cells_nodes:
                # inflated_grid_new.add_node(inflated_ped_cell)
                # inflated_grid_new.add_node((inflated_ped_cell[0]+1, inflated_ped_cell[1]))
                
                w = ped_inflated_grid[inflated_ped_cell] + ped_inflated_grid[(inflated_ped_cell[0]+1, inflated_ped_cell[1])]
                inflated_grid_new.add_edge(inflated_ped_cell, (inflated_ped_cell[0]+1, inflated_ped_cell[1]), weight=(1+7*(1-w)))
                #inflated_grid_new.add_edge(inflated_ped_cell, (inflated_ped_cell[0]+1, inflated_ped_cell[1]), weight=(2-w))
                #print(2-w)
            if (inflated_ped_cell[0]+1, inflated_ped_cell[1]+1) in inflated_ped_cells_nodes:
                # inflated_grid_new.add_node(inflated_ped_cell)
                # inflated_grid_new.add_node((inflated_ped_cell[0]+1, inflated_ped_cell[1]+1))
                
                w = ped_inflated_grid[inflated_ped_cell] + ped_inflated_grid[(inflated_ped_cell[0]+1, inflated_ped_cell[1]+1)]             
                inflated_grid_new.add_edge(inflated_ped_cell, (inflated_ped_cell[0]+1, inflated_ped_cell[1]+1), weight=(1+7*(1-w)))
                #inflated_grid_new.add_edge(inflated_ped_cell, (inflated_ped_cell[0]+1, inflated_ped_cell[1]+1), weight=(2-w))
                #print(2-w)

            if (inflated_ped_cell[0]-1, inflated_ped_cell[1]-1) in inflated_ped_cells_nodes:
                # inflated_grid_new.add_node(inflated_ped_cell)
                # inflated_grid_new.add_node((inflated_ped_cell[0]-1, inflated_ped_cell[1]-1))
                
                w = ped_inflated_grid[inflated_ped_cell] + ped_inflated_grid[(inflated_ped_cell[0]-1, inflated_ped_cell[1]-1)]             
                inflated_grid_new.add_edge(inflated_ped_cell, (inflated_ped_cell[0]-1, inflated_ped_cell[1]-1), weight=(1+7*(1-w)))
                #inflated_grid_new.add_edge(inflated_ped_cell, (inflated_ped_cell[0]-1, inflated_ped_cell[1]-1), weight=(2-w))

    try:
        #plan_point = env.goal_pos_map
        path = nx.astar_path(inflated_grid_new, env.robot_pos_map, env.goal_pos_map, heuristic=dist, weight="weight")     
        path_found = True
        # plt.figure()
        # plt.imshow(ped_inflated_grid)
        # for p in path:
        #     plt.plot(p[1],p[0],marker='o', markersize=1)
        # plt.show()
    except:
        path = None
        path_found = False

    return path, path_found

def gaussian_inflation(grid, x_o, y_o, theta, amplitude=1, sigma_x=10, sigma_y=7, d=2, inflation_weight=1):
    """
    Apply Gaussian inflation on an occupancy grid around a dynamic obstacle.
    
    :param grid: 2D numpy array representing the occupancy grid.
    :param x_o: x-coordinate of the obstacle's position.
    :param y_o: y-coordinate of the obstacle's position.
    :param theta: orientation of the obstacle in radians.
    :param amplitude: peak value of the Gaussian distribution.
    :param sigma_x: standard deviation of the Gaussian in x-direction.
    :param sigma_y: standard deviation of the Gaussian in y-direction.
    :param d: distance to shift the Gaussian center in the direction of theta.
    """
    #rows, cols = grid.shape
    rows_l = max(0, x_o-15)
    rows_h = min(grid.shape[0], x_o+15)
    cols_l = max(0, y_o-15)
    cols_h = min(grid.shape[1], y_o+15)

    sigma_x = inflation_weight * sigma_x
    sigma_y = inflation_weight * sigma_y
    # Shift the center of the Gaussian in front of the obstacle
    x_0 = y_o + d * np.cos(theta)
    y_0 = x_o + d * np.sin(theta)   
    
    for x in range(cols_l, cols_h):
        for y in range(rows_l, rows_h):
            if grid[y, x] != 0:
                # Calculate the distance of each point in the grid from the center of the Gaussian
                dx = x - x_0
                dy = y - y_0
                # Rotate the grid coordinates by -theta to align the Gaussian elongation with the obstacle's orientation
                dx_rotated = dx * np.cos(-theta) - dy * np.sin(-theta)
                dy_rotated = dx * np.sin(-theta) + dy * np.cos(-theta)
                # Calculate the Gaussian value
                value = amplitude * np.exp(-0.5 * ((dx_rotated ** 2 / sigma_x ** 2) + (dy_rotated ** 2 / sigma_y ** 2)))
                # Update the grid cell if the new value is greater than the current value
                grid[y, x] = grid[y,x] - value
               
    return grid 

def get_gaussian_inflated_pedestrian_map(env, ped, replan_map):
    ped_pos_map = env.world_to_map(ped[0])
    ped_orientation = ped[1]

    ped_distance_to_robot = int(env.get_relative_pos(ped[0]) / env.resolution)    
    min = 10
    max = 40
    inflation_weight = (ped_distance_to_robot-min)/ (max-min)
    inflation_weight = np.clip(inflation_weight, 0, 0.5)
    inflation_weight = 1 - inflation_weight

    inflated_grid = gaussian_inflation(replan_map, ped_pos_map[0], ped_pos_map[1], ped_orientation, inflation_weight=inflation_weight)
    inflated_grid[inflated_grid>0.5] = 1
    # plt.figure()
    # plt.imshow(inflated_grid)
    # plt.xlabel("after gaussian inflation")
    # plt.show()
    return inflated_grid

def occupancy_grid_to_graph(grid):
    G = nx.Graph()

    # Add nodes
    rows = len(grid)
    cols = len(grid[0])
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 1:  # Node is free
                G.add_node((i, j))

    # Add edges
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 1:  # Node is free
                if i < rows and grid[i + 1][j] == 1:
                    G.add_edge((i, j), (i + 1, j), weight=1)
                if j < cols and grid[i][j + 1] == 1:
                    G.add_edge((i, j), (i, j + 1), weight=1)
                if i < rows and j < cols-1 and grid[i+1][j+1] == 1:
                    G.add_edge((i, j), (i+1, j+1), weight=1.41)
                if 0 <= i < rows and 0 <= j < cols and grid[i-1][j-1] == 1:
                    G.add_edge((i,j), (i-1,j-1), weight=1.41)

    return G

def ray_casting(visible_cells, x, y, angle, occupancy_grid, max_distance, obs_cell_threshold=None):
    dx = math.cos(angle)
    dy = math.sin(angle)
    #print(dx,dy)
    
    cell_type = 1 # free space
    obstacle_first_point_found = None
    obstacle_cell_threshold = 0
    for r in np.linspace(0, max_distance, int(max_distance)):  # stepping in small increments
        ix = int(x + dx * r)
        iy = int(y + dy * r)
        #print(ix,iy)
        if ix < 0 or iy < 0 or ix >= occupancy_grid.shape[1] or iy >= occupancy_grid.shape[0]:
            # Out of bounds
            break

        if occupancy_grid[iy][ix] == 0:
            # Hit an obstacle
            obstacle_first_point_found = True
            cell_type = 0 # obstacle
            obstacle_cell_threshold += 1
        if (occupancy_grid[iy][ix] == 1 and obstacle_first_point_found == True) or obstacle_cell_threshold > obs_cell_threshold:
            obstacle_cell_threshold = 0
            break
            
        visible_cells[iy][ix] = cell_type

def dist(a,b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a-b)

def visualize_path(occupancy_grid, path, store=False, show=False, output_path=None):
    plt.figure()
    for i in path:
        plt.plot(i[1], i[0], marker="o", markersize=5, alpha=0.8)
    plt.imshow(occupancy_grid, cmap='gray'); 
    if show == True:
        plt.show()
    if store == True:
        plt.savefig(output_path)
    plt.close()

def inflate_grid(grid,
                 inflation_radius,
                 obstacle_threshold=0,
                 collision_val=0):
    """Inflates obstacles in an occupancy grid
    Creates a mask for all grid cells exceeding the obstacle threshold and
    uses a convolution to compute how much the obstacles should inflate. The
    inflated mask is used to set the cells of a copy of the initial grid to
    occupied. All other cells are unchanged: free space and unnoccupied cells
    outside of the inflation radius are preserved (therefore, frontiers can
    still be computed from an inflated grid).
    Args:
        grid (np.Array): occupancy grid
        inflation_radius (float): radius (in grid units) to inflate obstacles.
            Note that this is a float; a fractional inflation radius can be
            used to determine whether or not corners of a box are included.
        obstacle_threshold (float): value above which a cell is an obstacle.
        collision_val (float): value obstacles are given after inflation.
    Returns:
        inflated_grid (np.Array): grid with inflated obstacles.
    """

    obstacle_grid = np.zeros(grid.shape)
    obstacle_grid[grid == obstacle_threshold] = 1

    kernel_size = int(1 + 2 * math.ceil(inflation_radius))
    cind = int(math.ceil(inflation_radius))
    y, x = np.ogrid[-cind:kernel_size - cind, -cind:kernel_size - cind]
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[y * y + x * x <= inflation_radius * inflation_radius] = 1
    inflated_mask = scipy.ndimage.filters.convolve(obstacle_grid,
                                                   kernel,
                                                   mode='constant',
                                                   cval=0)
    inflated_mask = inflated_mask >= 1.0
    inflated_grid = grid.copy()
    inflated_grid[inflated_mask] = collision_val

    return inflated_grid

def get_cost(path, cost):
    total_cost = 0
    for i in range(len(path)-1):
        try:
            total_cost += cost[(path[i], path[i+1])]
        except:
            total_cost += cost[(path[i+1], path[i])]
    return total_cost

def path_cost_map(path):
    total_cost = 0
    for i in range(len(path)-1):
        total_cost += dist(path[i], path[i+1])
    return total_cost
