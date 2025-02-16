from heapq import heappush, heappop  # Recommended.
import numpy as np

#from code.sandbox import node_expanded
from flightsim.world import World

from .occupancy_map import OccupancyMap # Recommended.

def collision_detection(map, a, b):
    n = int(np.linalg.norm(np.array(a) - np.array(b)) / np.min(map.resolution))
    for t in np.linspace(0, 1, n):
        interp = (1 - t) * np.array(a) + t * np.array(b)
        if map.is_occupied_metric(interp):
            return True
    return False


def graph_search(world, resolution, margin, start, goal, astar):
    """
    Parameters:
        world,      World object representing the environment obstacles
        resolution, xyz resolution in meters for an occupancy map, shape=(3,)
        margin,     minimum allowed distance in meters from path to obstacles.
        start,      xyz position in meters, shape=(3,)
        goal,       xyz position in meters, shape=(3,)
        astar,      if True use A*, else use Dijkstra
    Output:
        return a tuple (path, nodes_expanded)
        path,       xyz position coordinates along the path in meters with
                    shape=(N,3). These are typically the centers of visited
                    voxels of an occupancy map. The first point must be the
                    start and the last point must be the goal. If no path
                    exists, return None.
        nodes_expanded, the number of nodes that have been expanded
    """

    # While not required, we have provided an occupancy map you may use or modify.
    occ_map = OccupancyMap(world, resolution, margin)
    # Retrieve the index in the occupancy grid matrix corresponding to a position in space.
    start_index = tuple(occ_map.metric_to_index(start))
    goal_index = tuple(occ_map.metric_to_index(goal))

    print("startIndex", occ_map.index_to_metric_negative_corner(start_index))
    print("startIndex", occ_map.index_to_metric_negative_corner(goal_index))

    cost = {start_index: 0}


    aliveNodes = []
    # push index and initial cost
    if astar:
        initial_h = 1.2 * np.linalg.norm(np.array(goal_index) - np.array(start_index))
        heappush(aliveNodes, (initial_h, 0, start_index))
    else:
        heappush(aliveNodes, (0, start_index))
    # distance = np.linalg.norm(np.array(goal_index) - np.array(start_index))
    # heappush(aliveNodes, (distance, start_index))

    nodes_expanded = 0
    reconstructed = []
    parentNodes = {start_index: None}
    # visited = {start_index: False}

    directions = []
    for x in [-1, 0, 1]:
        for y in [-1, 0, 1]:
            for z in [-1, 0, 1]:
                directions.append((x, y, z))
    directions.remove((0, 0, 0))

    while aliveNodes:
        # obtain the current node to be calculated
        # _, currNode = heappop(aliveNodes)
        if astar:
            current_f, current_g, currNode = heappop(aliveNodes)
        else:
            current_g, currNode = heappop(aliveNodes)

        if current_g > cost.get(currNode, float('inf')):
            continue

        nodes_expanded += 1

        # when reaching the goal, reconstruct the path by searching each node's parent
        if currNode == goal_index or not collision_detection(occ_map, currNode, goal_index):
            reconstructed.append(goal)
            while currNode is not None:
                reconstructed.append(occ_map.index_to_metric_center(currNode))
                print(occ_map.index_to_metric_center(currNode))
                currNode = parentNodes.get(currNode)
            reconstructed.append(start)
            reconstructed.reverse()
            return np.array(reconstructed), nodes_expanded

        # if visited.get(currNode, False):
        #     continue
        # visited[currNode] = True

        # obtain the next nodes
        for direction in directions:
            nextNode = tuple(np.array(currNode) + direction)
            # ensure the next node is valid and not occupied
            if not occ_map.is_valid_index(nextNode) or occ_map.is_occupied_index(nextNode):
                continue

            # calculate current running cost
            nextCost = cost[currNode] + np.linalg.norm(direction)
            # maintain the optimal subpath by comparing the running cost
            # if nextNode not in cost or nextCost < cost[nextNode]:
            if nextCost < cost.get(nextNode, float('inf')):
                parentNodes[nextNode] = currNode
                cost[nextNode] = nextCost
                if astar:
                    h = 1.2 * np.linalg.norm(np.array(goal_index) - np.array(nextNode))
                    f = nextCost + h
                    heappush(aliveNodes, (f, nextCost, nextNode))
                    # nextCost += abs(goal_index[0] - nextNode[0]) + abs(goal_index[1] - nextNode[1]) + abs(
                    #     goal_index[2] - nextNode[2])
                else:
                    heappush(aliveNodes, (nextCost, nextNode))

    # Return a tuple (path, nodes_expanded)
    return None, nodes_expanded