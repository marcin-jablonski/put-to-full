import numpy as np
import os
import math
import matplotlib.pyplot as plt

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
    cycle.append(cycle[0])
    final_value = sum(cycle_values)
    return cycle, final_value


def find_nearest_expansion(available_nodes, cycle):
    best_node = None
    best_node_result = None
    best_edge = None
    for i in range(len(cycle) - 1):
        for node in available_nodes:
            cost = (distance(cycle[i], node) + distance(cycle[i + 1], node)) * COST_WEIGHT
            node_result = node.gain + distance(cycle[i], cycle[i + 1]) * COST_WEIGHT - cost
            if best_node is None or node_result > best_node_result:
                best_node = node
                best_node_result = node_result
                best_edge = (cycle[i], cycle[i + 1])
    return best_node, best_node_result, best_edge


def cycle_expansion(nodes, starting_node_index=0):
    first_node = nodes[starting_node_index]
    nodes.remove(first_node)
    second_node, second_node_result = find_nearest_neighbour(first_node, nodes)
    nodes.remove(second_node)
    cycle = [first_node, second_node, first_node]
    cycle_values = [first_node.gain, second_node_result, second_node_result - second_node.gain]

    while True:
        next_node, next_node_result, edge = find_nearest_expansion(nodes, cycle)
        if next_node is None or next_node_result < 0:
            break

        nodes.remove(next_node)
        cycle_values.append(next_node_result)
        cycle.insert(cycle.index(edge[1]), next_node)

    final_value = sum(cycle_values)
    return cycle, final_value


def cycle_expansion_with_regret(nodes, starting_node_index=0):
    raise NotImplementedError


def print_result(nodes, result_nodes, result, title):
    free_nodes = list(set(nodes) - set(result_nodes))
    result_points = list(map(lambda node: (node.x, node.y), result_nodes))
    free_points = list(map(lambda node: (node.x, node.y), free_nodes))
    node_labels = list(map(lambda node: node.id, result_nodes))
    x, y = zip(*result_points)
    free_x, free_y = zip(*free_points)

    plt.figure()
    plt.plot(x, y, 'r', zorder=1, lw=2)
    plt.scatter(x, y, s=30, zorder=2)
    for i, label in enumerate(node_labels):
        plt.annotate(int(label), (x[i], y[i]))
    plt.scatter(free_x, free_y)
    plt.title(title)
    plt.annotate('Result: ' + str(round(result, 2)), xy=(0, max(y)))
    plt.show()


def main():
    nodes = read_data("./data")
    best_solution = None
    best_result = None
    for starting_index in range(0, len(nodes)):
        # solution = nearest_neighbour(nodes.copy(), starting_index)
        solution = cycle_expansion(nodes.copy(), starting_index)
        if best_solution is None or solution[1] > best_result:
            best_solution = solution[0]
            best_result = solution[1]
        # print(nearest_neighbour(nodes.copy(), starting_index)[1])
        # print(cycle_expansion(nodes.copy(), starting_index)[1])
        # cycle_expansion_with_regret(nodes.copy(), starting_index)
    print_result(nodes, best_solution, best_result, 'Cycle expansion')


main()
