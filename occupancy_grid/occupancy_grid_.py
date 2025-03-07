import numpy as np
import scipy
import copy
import sknw
import networkx as nx

from skimage.morphology import skeletonize
from matplotlib import pyplot as plt
from utils.utils import distance
from constants import constants as CONST
from simulator import display_map, convert_points_to_topdown
from point_cloud.point_cloud import get_point_clouds, get_min_max_pcd
from frontier.frontier import Frontier


def nearest_value_og(occupancy_grid, i, j, threshold=4):
    d = {CONST['free_space']:0, CONST['unknown']:0, CONST['occupied']:0}
    d[occupancy_grid[i-1][j]] += 1
    d[occupancy_grid[i+1][j]] += 1
    d[occupancy_grid[i][j-1]] += 1
    d[occupancy_grid[i][j+1]] += 1
      
    for occupancy_value, count in d.items():
      if count >= threshold:
          return occupancy_value, True
    return occupancy_grid[i][j], False

def is_not_on_boundary(size, i, j):
    if i > 0 and j > 0 and i < size[0] - 1 and j < size[1] -1:
        return True
    return False

def remove_isolated_points(occupancy_grid, threshold=2, display=False, show=False, store=False, path=None):
    for i in range(len(occupancy_grid)):
        for j in range(len(occupancy_grid[i])):
            if is_not_on_boundary(occupancy_grid.shape, i, j) == False:
                continue

            # If all of the 4 neighboring cell is of the same type, make the cell to be that type
            nearest_value, found = nearest_value_og(occupancy_grid, i, j, threshold = 4)
            if found == True:
                occupancy_grid[i][j] = nearest_value

            # If the current cell is unknown, replace it with the type of it's 2 neighbors
            elif occupancy_grid[i][j] == CONST['unknown']:
                occupancy_grid[i][j], _ = nearest_value_og(occupancy_grid, i, j, threshold = threshold)
    
    if display:
        display_occupancy_grid(occupancy_grid, show=show, store=store, output_path=path)
    return occupancy_grid

def create_occupancy_grid(current_pcds,current_point,min_x,max_x,min_y,max_y,min_z,max_z, index, \
    RESOLUTION = 0.1, scale=1, display=False, show=False, store=False, path=None):
    x_range = int((max_x - min_x) / RESOLUTION)
    y_range = int((max_y - min_y) / RESOLUTION)
    z_range = int((max_z - min_z) / RESOLUTION)
    
    THRESHOLD_LOW = min_y + 0.5
    #THRESHOLD_HIGH = max_y - (max_y-min_y)*0.35
    #THRESHOLD_HIGH = min_y + (max_y-min_y)*0.65
    THRESHOLD_HIGH = min_y + 1.5
    #print(min_y, max_y, max_y-min_y)
    
    occupancy_grid = np.ones((z_range+1, x_range+1)) * -1 #unexlpored
    l = []
    l1 = []
    l2 = []
    for i,pcd in enumerate(current_pcds):
        pc_points = (np.asarray(pcd.points) * scale) + current_point
        #o3d.visualization.draw_geometries([pcd])

        if pc_points.size == 0:
            continue
        l1.append(pcd)

        x = ((pc_points[:,0] - min_x) / RESOLUTION).astype(int)
        z = ((pc_points[:,2] - min_z) / RESOLUTION).astype(int)
        obj = np.logical_and(THRESHOLD_LOW < pc_points[:,1], pc_points[:,1] < THRESHOLD_HIGH)
        free_space = pc_points[:,1] <= THRESHOLD_LOW

        occupancy_grid[z[free_space], x[free_space]] = CONST['free_space'] #freespace = 0
        occupancy_grid[z[obj],x[obj]] = CONST['occupied'] #occupied = 1

        #o3d.visualization.draw_geometries([get_pcd_from_numpy(pc_points[free_space])])
        #l.extend(pc_points)
        l.extend(pc_points[free_space])
        l2.extend(pc_points[obj])
    
    #o3d.visualization.draw_geometries(current_pcds)    
    #o3d.visualization.draw_geometries([get_pcd_from_numpy(np.array(l))])
    #o3d.visualization.draw_geometries([get_pcd_from_numpy(np.array(l2))])
    if display == True:
        display_occupancy_grid(occupancy_grid, show=show, store=store, output_path=path)
    return occupancy_grid, l

def look_down(sim, occupancy_grid, current_point, min_x, min_z, display=False,\
    show=False, store=False, path=None):
    RESOLUTION = CONST['meters_per_pixel']

    xy_vis_points = convert_points_to_topdown(
         sim.pathfinder, [current_point], meters_per_pixel=CONST['meters_per_pixel']
        )
    #current_x = ((current_point[0] - min_x) / RESOLUTION).astype(int)
    #current_z = ((current_point[2] - min_z) / RESOLUTION).astype(int)
    current_x = int(xy_vis_points[0][0])
    current_z = int(xy_vis_points[0][1])
    display_map(occupancy_grid,[(current_x, current_z)], show=show)
    

    filtered_grid = scipy.ndimage.maximum_filter((occupancy_grid == -1), size=1)
    frontier_point_mask = np.logical_and(filtered_grid,
                                         occupancy_grid == -1)
    # Group the frontier points into connected components
    labels, nb = scipy.ndimage.label(filtered_grid)

    connected_components = []
    ground_cc = None
    if occupancy_grid[current_z][current_x] == 0 or occupancy_grid[current_z][current_x] == 1:
        return occupancy_grid
    flag = False
    for ii in range(nb):
        raw_frontier_indices = np.where(np.logical_and(labels == (ii + 1), frontier_point_mask)) 
        connected_components.append(Frontier(raw_frontier_indices[0], raw_frontier_indices[1]))
        for i in range(len(raw_frontier_indices[1])):
            if current_x == raw_frontier_indices[1][i] and current_z == raw_frontier_indices[0][i]:
                ground_cc = connected_components[-1]
                ground_cc_row = raw_frontier_indices[0]
                ground_cc_col = raw_frontier_indices[1]
                flag = True
                break
        if flag == True:
            break
        
    #show_frontiers(occupancy_grid, connected_components, show=True)
    #show_frontiers(occupancy_grid, [ground_cc], show=True)
    for i in range(len(ground_cc_row)):
        x = ground_cc_row[i]
        y = ground_cc_col[i]
        d = distance((x,y), (current_z, current_x))
        if d <= 1.5 / RESOLUTION:
            occupancy_grid[x][y] = 0
    #show_frontiers(occupancy_grid, [], show=True)
    if display:
        display_occupancy_grid(occupancy_grid, show=show, store=store, output_path=path)
    return occupancy_grid

def inflate_occupancy_grid(occupancy_grid, display=False, show=False, store=False, path=None):
    occupancy_grid_copy = copy.deepcopy(occupancy_grid)
    for i in range(len(occupancy_grid)):
        for j in range(len(occupancy_grid[i])):
            if occupancy_grid[i][j] == 1 and is_not_on_boundary(occupancy_grid.shape, i, j):
                occupancy_grid_copy[i-1][j] = 1
                occupancy_grid_copy[i][j-1] = 1
                occupancy_grid_copy[i+1][j] = 1
                occupancy_grid_copy[i][j+1] = 1
    if display:
        display_occupancy_grid(occupancy_grid_copy, show=show, store=store, output_path=path)
    return occupancy_grid_copy

def display_occupancy_grid(occupancy_grid, show=True, store=False, output_path=None):
    plt.figure(dpi=600)
    plt.imshow(occupancy_grid)
    
    if store == True:
        plt.savefig(output_path)
    if show == True:
        plt.show()
    plt.close()

def align_partial_map_to_gt(sim, gt_map, occupancy_grid, starting_point, x, z, display=False,\
    show=False, store=False, path=None):
    gt_map = copy.deepcopy(gt_map)
    xy_vis_points = convert_points_to_topdown(
         sim.pathfinder, [starting_point], meters_per_pixel=CONST['meters_per_pixel']
        )
    x_difference = int(int(xy_vis_points[0][0]) - x)
    y_difference = int(int(xy_vis_points[0][1]) - z)

    for i in range(occupancy_grid.shape[0]):
        for j in range(occupancy_grid.shape[1]):
            if occupancy_grid[i][j] == CONST['occupied'] or occupancy_grid[i][j] == CONST['free_space']: 
                try:
                    gt_map[i+y_difference][j+x_difference] = occupancy_grid[i][j]
                except:
                    pass
                    #print("obstacle")
                    #print(i,j)
                    #print(f"GT MAP SHAPE {gt_map.shape}, OG shape {occupancy_grid.shape} index: {i+y_difference},{j+x_difference}")
    
    if display:
        display_occupancy_grid(gt_map, show=show, store=store, output_path=path)
    return gt_map

def generate_partial_map(args, sim, occupancy_grid_on_gt, current_position, i, output_folder):
    pcds = get_point_clouds(CONST['rgb_path'], CONST['depth_path'], current_position, current_position)
    min_x,max_x,min_y,max_y,min_z,max_z = get_min_max_pcd(pcds, current_position, scale=1)
    if i == 0:
        fixed_min_y = min_y
        fixed_max_y = max_y
    x = ((current_position[0] - min_x) / CONST['meters_per_pixel']).astype(int)
    z = ((current_position[2] - min_z) / CONST['meters_per_pixel']).astype(int)

    # Create a partial map
    path = f'./{output_folder}/og_{i}'
    occupancy_grid, free_space = \
        create_occupancy_grid(pcds, current_position, min_x, max_x, fixed_min_y, fixed_max_y, min_z, max_z, i,\
            RESOLUTION=CONST['meters_per_pixel'], scale=1, display=args.display, show=args.show, store=args.store, path=path)
    occupancy_grid = remove_isolated_points(occupancy_grid, \
        display=args.display, show=args.show, store=args.store, path=path)
    
    # Align the partial map to the GT map
    path = f'./{output_folder}/og_on_gt{i}'
    occupancy_grid_on_gt = align_partial_map_to_gt(sim, occupancy_grid_on_gt, occupancy_grid, \
        current_position, x, z, display=args.display, show=args.show, store=args.store, path=path)
    
    # Make the space around 1.5m radius closest to the robot as free
    occupancy_grid_on_gt = look_down(sim, occupancy_grid_on_gt, current_position, min_x, min_z, \
        display=args.display, show=args.show, store=args.store, path=path)
    
    # Inflate the occupancy grid
    path = f'./{output_folder}/inflated_og{i}'
    inflated_occupancy_grid_on_gt = inflate_occupancy_grid(occupancy_grid_on_gt, \
        display=args.display, show=args.show, store=args.store, path=path)
    inflated_occupancy_grid_on_gt = remove_isolated_points(inflated_occupancy_grid_on_gt, \
        display=args.display, show=args.show, store=args.store, path=path)

    return occupancy_grid_on_gt, inflated_occupancy_grid_on_gt, occupancy_grid

def display_skeleton(skeleton, graph, show=False, store=False, path=None):
    if show == True:
        plt.imshow(skeleton, cmap='gray')

        # draw edges by pts
        for (s,e) in graph.edges():
            ps = graph[s][e]['pts']
            plt.plot(ps[:,1], ps[:,0], 'green')
            
        # draw node by o
        nodes = graph.nodes()
        ps = np.array([nodes[i]['o'] for i in nodes])
        plt.plot(ps[:,1], ps[:,0], 'r.')

        # title and show
        plt.title('Build Graph')
        plt.show()
    
def skeletonize_map(occupancy_grid, display=False, show=False, store=False):
    skeleton = skeletonize(occupancy_grid)
    
    graph = sknw.build_sknw(skeleton)

    if display == True:
        #display_occupancy_grid(skeleton)
        display_skeleton(skeleton, graph, show, store, path=None)
   
    
    tsp = nx.algorithms.approximation.traveling_salesman_problem
    path = tsp(graph)
    #print(path)

    nodes = list(graph.nodes)
    for i in range(len(path)):
        if not nodes:
            index = i
            break
        if path[i] in nodes:
            nodes.remove(path[i])
    
    d_in = path[:index]
    d_out = path[index-1:]

    cost_din = 0
    for i in range(len(d_in)-1):
        cost_din += graph[d_in[i]][d_in[i+1]]['weight']

    cost_dout = 0
    for i in range(len(d_out)-1):
        cost_dout += graph[d_out[i]][d_out[i+1]]['weight']
    
    return cost_din+cost_dout, cost_din, cost_dout