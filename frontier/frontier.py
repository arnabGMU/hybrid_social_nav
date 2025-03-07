import numpy as np
import scipy.ndimage
from matplotlib import pyplot as plt


class Frontier:
    def __init__(self,x,y):
        self.frontier = np.vstack((x,y))
        self.centroid = np.mean(self.frontier, axis=1)
    def __eq__(self, __o: object) -> bool:
        if self.centroid[0] == __o.centroid[0] and self.centroid[1] == __o.centroid[1]:
            return True
        else:
            return False

def get_frontiers(occupancy_grid, closest_free_space = None, closest = False, frontier_size=10, args=False, index=None, output_folder=None):
    if closest == False:
        filtered_grid = scipy.ndimage.maximum_filter((occupancy_grid == 1), size=3)
        #print(filtered_grid)
    else:
        filtered_grid = np.zeros(occupancy_grid.shape, dtype=bool)
        filtered_grid[closest_free_space[:,0], closest_free_space[:,1]] = True
        #plt.imshow(filtered_grid);plt.show()
        #print(filtered_grid)
    frontier_point_mask = np.logical_and(filtered_grid,
                                         occupancy_grid == 0.5)
    # Group the frontier points into connected components
    labels, nb = scipy.ndimage.label(frontier_point_mask)

    # Extract the frontiers
    frontiers = []
    for ii in range(nb):
        raw_frontier_indices = np.where(np.logical_and(labels == (ii + 1), frontier_point_mask))
        if len(raw_frontier_indices[0]) < frontier_size:
            continue
        f = Frontier(raw_frontier_indices[0], raw_frontier_indices[1])
        norm = np.linalg.norm(f.frontier - f.centroid[:, None], axis=0)
        ind = np.argmin(norm)
        f.centroid = f.frontier[:, ind]
        frontiers.append(f)
        

    #if args.display == True:
    #    path = f'./{output_folder}/frontier{index}'
    #    show_frontiers(occupancy_grid, frontiers, show=True, store=args.store, output_path=path)
    return frontiers

def show_frontiers(occupancy_grid, frontiers, show=True, store=False, output_path=None):    
    if isinstance(occupancy_grid, list) == False:
        plt.imshow(occupancy_grid)
        for f in frontiers:
            plt.scatter(f.frontier[1],f.frontier[0])
            plt.scatter(f.centroid[1], f.centroid[0])    
    else:
        fig, ax = plt.subplots(2,3, figsize=(20,10))
        fig.dpi = 1200
        p = 0
        for i in range(len(ax)):
            for j in range(len(ax[i])):
                ax[i][j].imshow(occupancy_grid[p])
                for f in frontiers[p]:
                    ax[i][j].scatter(f.frontier[1],f.frontier[0])
                    ax[i][j].scatter(f.centroid[1], f.centroid[0])    
                p+=1    
    if store == True:
        plt.savefig(output_path, figsize=(20,10))
    if show == True:
        plt.show()
    plt.close()

def frontier_path_cost(graph, path):
    path_to_frontier_cost = 0
    for i in range(len(path)-2):
        edge_cost = graph[path[i]][path[i+1]]["cost"]
        if edge_cost == np.inf:
            path_to_frontier_cost = np.inf
            break
            plt.figure()
            plt.imshow(self.global_map_inflated_grid)
            plt.plot(path[i][1], path[i][0], marker='o')
            plt.plot(path[i+1][1], path[i+1][0], marker='o')
        path_to_frontier_cost += graph[path[i]][path[i+1]]["cost"]
    return path_to_frontier_cost