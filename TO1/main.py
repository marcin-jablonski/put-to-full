import numpy as np
import os
import math

COST_WEIGHT = 5


class Node:
    id = int
    x = float
    y = float
    gain = float

    def __init__(self, id, x, y, gain=0):
        self.id = id
        self.x = x
        self.y = y
        self.gain = gain


def read_positions(path):
    nodes = []
    file = np.loadtxt(path, delimiter=" ", skiprows=6)
    for line in file:
        node = Node(line[0], line[1], line[2])
        nodes.append(node)
    return nodes


def read_gains(nodes, path):
    file = np.loadtxt(path, delimiter=" ", skiprows=6)
    for i, line in enumerate(file.tolist()):
        nodes[i].gain = line[1]
    return nodes


def read_data(path):
    positions_file_path = os.path.join(path, "kroA100.tsp")
    gain_file_path = os.path.join(path, "kroB100.tsp")
    nodes = read_positions(positions_file_path)
    nodes = read_gains(nodes, gain_file_path)
    return nodes


def distance(node1, node2):
    return math.sqrt(pow(node1.x - node2.x, 2) + pow(node1.y - node2.y, 2))


def find_nearest_neighbour(current_node, available_nodes):
    best_node = None
    best_node_result = None
    for node in available_nodes:
        cost = distance(current_node, node) * COST_WEIGHT
        node_result = node.gain - cost
        if best_node is None or node_result > best_node_result:
            best_node = node
            best_node_result = node_result
    return best_node, best_node_result


def nearest_neighbour(nodes, starting_node_index=0):
    current_node = nodes[starting_node_index]
    cycle = [current_node]
    cycle_values = [current_node.gain]
    nodes.remove(current_node)

    while True:
        next_node, next_node_result = find_nearest_neighbour(current_node, nodes)

        if next_node is None or next_node_result < 0:
            break

        nodes.remove(next_node)
        cycle.append(next_node)
        cycle_values.append(next_node_result)
        current_node = next_node

    cycle_values.append(distance(cycle[0], cycle[-1]) * COST_WEIGHT)
    final_value = sum(cycle_values)
    return cycle, final_value


def cycle_expansion(nodes, starting_node_index=0):
    raise NotImplementedError


def cycle_expansion_with_regret(nodes, starting_node_index=0):
    raise NotImplementedError


def main():
    nodes = read_data("./data")
    print(nearest_neighbour(nodes.copy()))
    # cycle_expansion(nodes.copy())
    # cycle_expansion_with_regret(nodes.copy())


main()
