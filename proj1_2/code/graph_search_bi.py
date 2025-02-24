from heapq import heappush, heappop
import numpy as np
from flightsim.world import World
from .occupancy_map import OccupancyMap


def collision_detection(occ_map, a, b):
    n = int(np.linalg.norm(np.array(a) - np.array(b)) / np.min(occ_map.resolution))
    for t in np.linspace(0, 1, n):
        interp = (1 - t) * np.array(a) + t * np.array(b)
        if occ_map.is_occupied_metric(interp):
            return True
    return False


def graph_search_bi(world, resolution, margin, start, goal, astar):
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

    occ_map = OccupancyMap(world, resolution, margin)
    start_index = tuple(occ_map.metric_to_index(start))
    goal_index = tuple(occ_map.metric_to_index(goal))

    cost_S = {start_index: 0}
    cost_G = {goal_index: 0}

    aliveNodes_S = []
    aliveNodes_G = []
    distance = np.linalg.norm(np.array(goal_index) - np.array(start_index))
    heappush(aliveNodes_S, (distance, start_index))
    heappush(aliveNodes_G, (distance, goal_index))


    parentNodes_S = {start_index: None}
    parentNodes_G = {goal_index: None}

    nodes_expanded_S = 0
    nodes_expanded_G = 0

    directions = []
    for x in [-1, 0, 1]:
        for y in [-1, 0, 1]:
            for z in [-1, 0, 1]:
                directions.append((x, y, z))
    directions.remove((0, 0, 0))

    reconstructed_S = []
    reconstructed_G = []

    visited_S = {start_index: False}
    visited_G = {goal_index: False}

    while aliveNodes_S and aliveNodes_G:
        _, currNode_S = heappop(aliveNodes_S)
        _, currNode_G = heappop(aliveNodes_G)
        nodes_expanded_S += 1
        nodes_expanded_G += 1

        # dist_S = np.linalg.norm(np.array(goal_index) - np.array(currNode_S))
        # dist_G = np.linalg.norm(np.array(start_index) - np.array(currNode_G))
        # if dist_S > distance / 2 or dist_G > distance / 2:
        if not collision_detection(occ_map, currNode_S, currNode_G):
            while currNode_S is not None:
                reconstructed_S.append(occ_map.index_to_metric_center(currNode_S))
                currNode_S = parentNodes_S.get(currNode_S)
            reconstructed_S.append(start)
            reconstructed_S.reverse()

            while currNode_G is not None:
                reconstructed_G.append(occ_map.index_to_metric_center(currNode_G))
                currNode_G = parentNodes_G.get(currNode_G)
            reconstructed_G.append(goal)

            return np.array(reconstructed_S + reconstructed_G), nodes_expanded_S + nodes_expanded_G
        # else:
        if currNode_S == goal_index:
            reconstructed_S.append(goal)
            while currNode_S is not None:
                reconstructed_S.append(occ_map.index_to_metric_center(currNode_S))
                currNode_S = parentNodes_S.get(currNode_S)
            reconstructed_S.append(start)
            reconstructed_S.reverse()
            return np.array(reconstructed_S), nodes_expanded_S

        if currNode_G == start_index:
            reconstructed_G.append(start)
            while currNode_G is not None:
                reconstructed_G.append(occ_map.index_to_metric_center(currNode_G))
                currNode_G = parentNodes_G.get(currNode_G)
            reconstructed_G.append(goal)
            return np.array(reconstructed_G), nodes_expanded_G

        for direction in directions:
            nextNode_S = tuple(np.array(currNode_S) + np.array(direction))
            nextNode_G = tuple(np.array(currNode_G) + np.array(direction))

            if occ_map.is_valid_index(nextNode_S) and not occ_map.is_occupied_index(nextNode_S):
                nextCost_S = cost_S[currNode_S] + np.linalg.norm(direction)

                if nextCost_S not in cost_S or nextCost_S < cost_S[nextNode_S]:
                    cost_S[nextNode_S] = nextCost_S
                    parentNodes_S[nextNode_S] = currNode_S
                    if astar:
                        nextCost_S += 1.5 * np.linalg.norm(np.array(goal_index) - np.array(nextNode_S))
                    heappush(aliveNodes_S, (nextCost_S, nextNode_S))

            if occ_map.is_valid_index(nextNode_G) and not occ_map.is_occupied_index(nextNode_G):
                nextCost_G = cost_G[currNode_G] + np.linalg.norm(direction)

                if nextCost_G not in cost_G or nextCost_G < cost_G[nextNode_G]:
                    cost_G[nextNode_G] = nextCost_G
                    parentNodes_G[nextNode_G] = currNode_G
                    if astar:
                        nextCost_G += 1.5 * np.linalg.norm(np.array(start_index) - np.array(nextNode_G))
                    heappush(aliveNodes_G, (nextCost_G, nextNode_G))

    return None, nodes_expanded_S + nodes_expanded_G