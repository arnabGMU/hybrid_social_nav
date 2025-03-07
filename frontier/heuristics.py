import numpy as np
from frontier.frontier import Frontier
from constants import constants as CONST
from simulator import get_shortest_path

def get_largest_frontier(frontiers, starting_point, current_point, min_x, min_z, visited_frontiers):
    points = []
    RESOLUTION = 0.1
    max_length = 0
    max_point = np.array([])
    final_path = None
    max_frontier = None
    #print(visited_frontiers)
    for f in frontiers:
        if f in visited_frontiers:
            continue
        
        print("starting_point",starting_point)
        x = (f.centroid[1] * RESOLUTION)+ min_x
        z = (f.centroid[0] * RESOLUTION) + min_z
        point = np.array([x,height,z])
        #print(f.centroid, point)
        #show_frontiers(occupancy_grid, [f], show= True, store=False, output_path=None)

        xy_vis_points = convert_points_to_topdown(
         sim.pathfinder, [point], meters_per_pixel=meters_per_pixel
        )
    
        #path = f'./example4/closest_frontier{i}'
        path = f'./example5/largest_frontier{i}'
        #display_map(top_down_map,xy_vis_points, show=True, store=False, output_path=path)
        #break
        #print(f.frontier)

        shortest_path = get_shortest_path(current_point, point)
        if shortest_path == 0: #Path not found
            print("Shortest path not found")
            #print(current_point, point)
            continue
        if len(f.frontier[0]) > max_length:
            max_length = len(f.frontier[0])
            max_point = point
            max_frontier = f
            final_path = shortest_path
    #points.append(point)
    return max_point, final_path, max_frontier

def get_smallest_frontier(frontiers, current_point, visited_frontiers):
    points = []
    RESOLUTION = 0.1
    min_length = np.inf
    min_point = np.array([])
    final_path = None
    min_frontier = None
    #print(visited_frontiers)
    for f in frontiers:
        if f in visited_frontiers:
            continue
        
        print("starting_point",starting_point)
        x = (f.centroid[1] * RESOLUTION)+ min_x
        z = (f.centroid[0] * RESOLUTION) + min_z
        point = np.array([x,height,z])
        #print(f.centroid, point)
        #show_frontiers(occupancy_grid, [f], show= True, store=False, output_path=None)

        #xy_vis_points = convert_points_to_topdown(
        # sim.pathfinder, [point], meters_per_pixel=meters_per_pixel
        #)
    
        #path = f'./example4/closest_frontier{i}'
        #path = f'./example5/largest_frontier{i}'
        #display_map(top_down_map,xy_vis_points, show=True, store=False, output_path=path)
        #break
        #print(f.frontier)

        shortest_path = get_shortest_path(current_point, point)
        if shortest_path == 0: #Path not found
            print("Shortest path not found")
            #print(current_point, point)
            continue
        if len(f.frontier[0]) < min_length:
            min_length = len(f.frontier[0])
            min_point = point
            min_frontier = f
            final_path = shortest_path
    #points.append(point)
    return min_point, final_path, min_frontier

def get_farthest_frontier(frontiers, current_point, visited_frontiers):
    points = []
    RESOLUTION = 0.1
    max_distance = np.inf * -1
    max_point = np.array([])
    final_path = None
    max_frontier = None
    #print(visited_frontiers)
    for f in frontiers:
        if f in visited_frontiers:
            continue
        #print("centroid", f.centroid)
        #print("curremt_point", current_point)
        x = ((start_z- f.centroid[1]) * RESOLUTION) + current_point[0]
        z = ((f.centroid[0]-start_x) * RESOLUTION) + current_point[2]

        x = (f.centroid[1] * RESOLUTION)+ min_x
        z = (f.centroid[0] * RESOLUTION) + min_z

        point = np.array([x,height,z])
        '''
        print("start_x, start_z", start_x, start_z)
        print("point",point)
        show_frontiers(occupancy_grid, [f], show= True, store=False, output_path=None)
        xy_vis_points = convert_points_to_topdown(
         sim.pathfinder, [point], meters_per_pixel=meters_per_pixel
        )
        display_map(top_down_map,xy_vis_points, show=True, store=False, output_path=path)
        print(top_down_map.shape)
        '''
        shortest_path = get_shortest_path(current_point, point)
        if shortest_path == 0: #Path not found
            print("Shortest path not found")
            #print(current_point, point)
            continue
        if shortest_path.geodesic_distance > max_distance:
            max_distance = shortest_path.geodesic_distance
            max_point = point
            max_frontier = f
            final_path = shortest_path
    #points.append(point)
    return max_point, final_path, max_frontier

def get_closest_frontier(frontiers, current_point, visited_frontiers):
    points = []
    RESOLUTION = 0.1
    min_distance = np.inf
    min_point = np.array([])
    final_path = None
    min_frontier = None
    #print(visited_frontiers)
    for f in frontiers:
        if f in visited_frontiers:
            continue
        #print("centroid", f.centroid)
        #print("curremt_point", current_point)
        #x = ((start_z- f.centroid[1]) * RESOLUTION) + current_point[0]
        #z = ((f.centroid[0]-start_x) * RESOLUTION) + current_point[2]

        x = (f.centroid[1] * RESOLUTION)+ min_x
        z = (f.centroid[0] * RESOLUTION) + min_z

        point = np.array([x,height,z])
        '''
        print("start_x, start_z", start_x, start_z)
        print("point",point)
        show_frontiers(occupancy_grid, [f], show= True, store=False, output_path=None)
        xy_vis_points = convert_points_to_topdown(
         sim.pathfinder, [point], meters_per_pixel=meters_per_pixel
        )
        display_map(top_down_map,xy_vis_points, show=True, store=False, output_path=path)
        print(top_down_map.shape)
        '''
        shortest_path = get_shortest_path(current_point, point)
        if shortest_path == 0: #Path not found
            print("Shortest path not found")
            #print(current_point, point)
            continue
        if shortest_path.geodesic_distance < min_distance:
            min_distance = shortest_path.geodesic_distance
            min_point = point
            min_frontier = f
            final_path = shortest_path
    #points.append(point)
    return min_point, final_path, min_frontier

def get_anticipated_frontier(frontiers, current_point, visited_frontiers):
    points = []
    RESOLUTION = 0.1
    max_count = 0
    max_point = np.array([])
    final_path = None
    max_frontier = None
    max_anticipated_rec = None
    for f in frontiers:
        if f in visited_frontiers:
            continue
        
        #print("starting_point",starting_point)
        x = (f.centroid[1] * RESOLUTION)+ min_x
        z = (f.centroid[0] * RESOLUTION) + min_z
        point = np.array([x,height,z])
        #print(f.centroid, point)
        #show_frontiers(occupancy_grid, [f], show= True, store=False, output_path=None)
        left, right = x - 0.5, x + 0.5
        top, bottom = z - 0.5, z + 0.5
        
        anticipation_count = 0
        anticipation_rec = []
        
        i = top
        while i <= bottom:
            j = left
            while j <= right:
                anticipation_point = np.array((j, height, i))
                if sim.pathfinder.is_navigable(anticipation_point):
                    anticipation_count += 1
                anticipation_rec.append(anticipation_point)
                j += 0.2
            i += 0.2
        #xy_vis_points = convert_points_to_topdown(
        # sim.pathfinder, anticipation_rec, meters_per_pixel=meters_per_pixel
        #)
        #display_map(top_down_map,xy_vis_points, show=True, store=False, output_path=None)
        #path = f'./example4/closest_frontier{i}'
        #path = f'./example5/largest_frontier{i}'
        #display_map(top_down_map,xy_vis_points, show=True, store=False, output_path=path)
        #break
        #print(f.frontier)
        if anticipation_count > max_count:
            max_count = anticipation_count
            shortest_path = get_shortest_path(current_point, point)
            if shortest_path == 0: #Path not found
                print("Shortest path not found")
                #print(current_point, point)
                continue
            else:
                max_point = point
                max_frontier = f
                final_path = shortest_path
                max_anticipated_rec = anticipation_rec
    #points.append(point)
    return max_point, final_path, max_frontier, max_anticipated_rec

def check_nearest_cc(f, connected_components):
    min = np.inf
    for cc in connected_components:
        x = (cc.frontier[0] - f.centroid[0])**2
        y = (cc.frontier[1] - f.centroid[1])**2

        k = np.sqrt(x + y)
        min_distance = np.min(k)
        if min_distance < min:
            min = min_distance
            min_cc = cc
    return min, min_cc

def next_frontier_to_explore(heuristic, frontiers, current_point):
    if heuristic == 'closest':
        return get_closest_frontier(frontiers, current_point, None)
    elif heuristic == 'farthest':
        return get_farthest_frontier(frontiers, current_point, None)
    elif heuristic == 'largest':
        return get_largest_frontier(frontiers, current_point, None)
    elif heuristic == 'anticipation':
        return get_anticipated_frontier(frontiers, current_point, None)
    