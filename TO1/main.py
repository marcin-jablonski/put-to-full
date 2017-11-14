import numpy as np
import os


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


def nearest_neighbour(nodes, starting_node_index=0):
    raise NotImplementedError


def cycle_expansion(nodes, starting_node_index=0):
    raise NotImplementedError


def cycle_expansion_with_regret(nodes, starting_node_index=0):
    raise NotImplementedError


def main():
    nodes = read_data("./data")
    nearest_neighbour(nodes.copy())
    cycle_expansion(nodes.copy())
    cycle_expansion_with_regret(nodes.copy())


main()
