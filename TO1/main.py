import numpy as np
import os
import math
import matplotlib.pyplot as plt
import time
import random
from random import randint
COST_WEIGHT = 6


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

    cycle_values.append(-distance(cycle[0], cycle[-1]) * COST_WEIGHT)
    cycle.append(cycle[0])
    final_value = sum(cycle_values)
    return cycle, final_value


def find_nearest_expansion(available_nodes, cycle, random_expansion=False):
    best_node = None
    best_node_result = None
    best_edge = None

    if random_expansion:
        i = random.randint(0, len(cycle) - 2)
        j = random.randint(0, len(available_nodes)-1)
        cost = (distance(cycle[i], available_nodes[j]) + distance(cycle[i + 1], available_nodes[j])) * COST_WEIGHT
        node_result = available_nodes[j].gain + distance(cycle[i], cycle[i + 1]) * COST_WEIGHT - cost
        return available_nodes[j], node_result, (cycle[i], cycle[i + 1])

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
        cycle.insert(cycle[1::].index(edge[1]) + 1, next_node)

    final_value = sum(cycle_values)
    return cycle, final_value


def find_best_regret_expansion(nodes, cycle, cycle_values):
    best_node = None
    best_node_regret = None
    broken_edge_index = None

    for node in nodes:
        first_best_edge_index = None
        second_best_edge_index = None
        first_best_edge_score = 0
        second_best_edge_score = 0

        for edge_index in range(len(cycle) - 1):
            new_cycle_values = cycle_values.copy()
            del new_cycle_values[2*edge_index]
            new_cycle_values[2*edge_index:2*edge_index] = [-distance(node, cycle[edge_index]) * COST_WEIGHT, node.gain, -distance(node, cycle[edge_index + 1]) * COST_WEIGHT]
            edge_break_result = sum(new_cycle_values) - sum(cycle_values)

            if first_best_edge_index is None or edge_break_result > first_best_edge_score:
                second_best_edge_index = first_best_edge_index
                second_best_edge_score = first_best_edge_score
                first_best_edge_index = edge_index
                first_best_edge_score = edge_break_result
            elif second_best_edge_index is None or edge_break_result > second_best_edge_score:
                second_best_edge_index = edge_index
                second_best_edge_score = edge_break_result

        if first_best_edge_index is not None and first_best_edge_score > 0:
            node_regret = first_best_edge_score - second_best_edge_score

            if best_node is None or node_regret > best_node_regret:
                best_node = node
                best_node_regret = node_regret
                broken_edge_index = first_best_edge_index

    return best_node, broken_edge_index


def insert_node_with_breaking_edge(node, edge_index, cycle, cycle_values):
    del cycle_values[2 * edge_index]
    cycle_values[2 * edge_index:2 * edge_index] = [-distance(node, cycle[edge_index]) * COST_WEIGHT, node.gain, -distance(node, cycle[edge_index + 1]) * COST_WEIGHT]
    cycle.insert(edge_index + 1, node)
    return cycle, cycle_values


def cycle_expansion_with_regret(nodes, starting_node_index=0):
    first_node = nodes[starting_node_index]
    nodes.remove(first_node)
    second_node, second_node_result = find_nearest_neighbour(first_node, nodes)
    nodes.remove(second_node)
    cycle = [first_node, second_node, first_node]
    cycle_values = [second_node_result - second_node.gain, second_node.gain, second_node_result - second_node.gain, first_node.gain]

    while True:
        node, edge_index = find_best_regret_expansion(nodes, cycle, cycle_values)

        if node is None:
            break

        nodes.remove(node)
        cycle, cycle_values = insert_node_with_breaking_edge(node, edge_index, cycle, cycle_values)

    return cycle, sum(cycle_values)


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


def verify_solution(cycle, result):
    cycle_values = 0
    for i in range(0, len(cycle) - 1):
        cycle_values += cycle[i].gain - distance(cycle[i], cycle[i+1]) * COST_WEIGHT

    if result != cycle_values:
        return result - cycle_values

    return 0

def bestRemoveNode(cycle, random_remove=False):
    starting_node = cycle[0]
    starting_gain = distance(cycle[0], cycle[1]) * COST_WEIGHT + distance(cycle[0], cycle[-2]) * COST_WEIGHT
    starting_result = starting_gain - cycle[0].gain - distance(cycle[-2], cycle[1]) * COST_WEIGHT

    best_node = starting_node
    best_node_result = starting_result

    if random_remove:
        i = random.randint(1, len(cycle) - 2)
        distance_gain = distance(cycle[i - 1], cycle[i]) * COST_WEIGHT + distance(cycle[i + 1], cycle[i]) * COST_WEIGHT
        node_result = distance_gain - cycle[i].gain - distance(cycle[i - 1], cycle[i + 1]) * COST_WEIGHT
        return cycle[i], node_result

    for i in range(1, len(cycle) - 1):
        distance_gain = distance(cycle[i-1], cycle[i]) * COST_WEIGHT + distance(cycle[i+1], cycle[i]) * COST_WEIGHT
        node_result = distance_gain - cycle[i].gain - distance(cycle[i-1], cycle[i+1]) * COST_WEIGHT
        if best_node is None or node_result > best_node_result:
            best_node = cycle[i]
            best_node_result = node_result

    return best_node, best_node_result


def best_edge_swap(cycle):
    best_swap_result = -float("inf")
    best_swapped_cycle = None

    if len(cycle) - 1 <= 3:
        return best_swapped_cycle, best_swap_result

    for i in range(1, len(cycle) - 2):
        for j in range(i+1, len(cycle) - 1):
            if i == 1 and j == len(cycle) - 2:
                continue

            total_change = distance(cycle[i-1], cycle[i]) - distance(cycle[i], cycle[j+1]) + distance(cycle[j + 1], cycle[j]) - distance(cycle[j], cycle[i - 1])
            total_change *= COST_WEIGHT
            swapped_cycle = cycle.copy()
            swapped_cycle[i:j + 1] = list(reversed(swapped_cycle[i:j + 1]))

            if best_swap_result is None or best_swap_result < total_change:
                best_swap_result = total_change
                best_swapped_cycle = swapped_cycle

    return best_swapped_cycle, best_swap_result


def findBestLocal(available_nodes, cycle, times):
    start = time.time()
    next_node, next_node_result, edge = find_nearest_expansion(available_nodes.copy(),cycle.copy())
    node_to_remove, remove_node_result = bestRemoveNode(cycle.copy())
    new_cycle, swap_nodes_result = best_edge_swap(cycle.copy())
    end = time.time()
    times.append(end-start)
    results = [next_node_result, remove_node_result, swap_nodes_result]
    best_local = np.argmax(results)

    if results[best_local] < 0:
        return None, None, None, None
    else:
        if best_local == 0: #add Node
            cycle.insert(cycle[1::].index(edge[1]) + 1, next_node)
            return cycle, next_node_result, next_node, 1
        elif best_local == 1: #remove Node
            if node_to_remove == cycle[0]:
                cycle.remove(node_to_remove)
                cycle.remove(node_to_remove)
                cycle.append(cycle[0])
            else:
                del cycle[cycle[1::].index(node_to_remove)+1]
            return cycle, remove_node_result, node_to_remove, 2
        else: #Swap edges
            return new_cycle, swap_nodes_result, None, 3


def enhanceSolutionWithLocals(cycle, availableNodes, cycle_values):
    enhanced_cycle = cycle
    nodes = availableNodes
    times = []
    while True:
        new_cycle, delta, new_node, local_type = findBestLocal(nodes, enhanced_cycle, times)
        if new_cycle is not None:
            enhanced_cycle = new_cycle
            cycle_values += delta
            if local_type == 1:
                nodes.remove(new_node)
            elif local_type == 2:
                nodes.append(new_node)
        else:
            break

    return enhanced_cycle, cycle_values, times


def generateRandomSolution(nodes):
    no_of_nodes = randint(10, 98)
    cycle = nodes[0:no_of_nodes]
    random.shuffle(cycle)
    cycle.append(cycle[0])
    cycle_values = 0
    for i in range(0, len(cycle) - 1):
        cycle_values += cycle[i].gain - distance(cycle[i], cycle[i+1]) * COST_WEIGHT

    return cycle, cycle_values


def multiple_start_local_search(nodes):
    best_solution = None
    start = time.time()
    for i in range(0, 100):
        print('MS LS completed: ' + str(100*i/100)+" %")
        random_solution = generateRandomSolution(nodes.copy())
        enhanced_solution = enhanceSolutionWithLocals(random_solution[0].copy(), list(set(nodes.copy()) - set(random_solution[0])), random_solution[1])
        if best_solution is None or enhanced_solution[1] > best_solution[1]:
            best_solution = enhanced_solution
    end = time.time()
    duration = end - start
    return best_solution, duration


def get_random_neighbour_solution(nodes, cycle, result):
    cycle_values = result
    decision_made = False

    while not decision_made:
        decision = random.randint(1, 3)
        if decision == 1:
            # swap nodes
            if len(cycle) - 1 > 3:
                cycle, delta = node_swap(cycle, True)
                cycle_values += delta
                decision_made = True
        elif decision == 2:
            # add node
            if len(nodes) > 0:
                next_node, next_node_result, edge = find_nearest_expansion(nodes, cycle, True)
                nodes.remove(next_node)
                cycle_values += next_node_result
                cycle.insert(cycle[1::].index(edge[1]) + 1, next_node)
                decision_made = True
        else:
            # remove node
            if len(cycle) - 1 > 1:
                node_to_remove, remove_node_result = bestRemoveNode(cycle, True)
                del cycle[cycle[1::].index(node_to_remove) + 1]
                cycle_values += remove_node_result
                decision_made = True

    return nodes, cycle, cycle_values


def perturbation(cycle, result):
    #triple random swap nodes
    if len(cycle) - 1 > 3:
        new_cycle, delta = node_swap(cycle, True)
        cycle_values = result + delta
        new_cycle, delta = node_swap(new_cycle, True)
        cycle_values = cycle_values + delta
        new_cycle, delta = node_swap(new_cycle, True)
        cycle_values = cycle_values + delta
        return new_cycle, cycle_values

    return cycle, result


def iterated_local_search(nodes, stop_time):
    best_solution = generateRandomSolution(nodes.copy())
    best_solution = enhanceSolutionWithLocals(best_solution[0].copy(), list(set(nodes.copy()) - set(best_solution[0])), best_solution[1])
    start = time.time()
    while True:
        enhanced_solution = perturbation(best_solution[0].copy(), best_solution[1])
        if enhanced_solution[1] > best_solution[1]:
            best_solution = enhanced_solution
        if time.time() - start >= stop_time:
            break

    return best_solution


def reverse_nodes(nodes, node1, node2):
    n1, n2 = nodes[1:-1].index(node1) + 1, nodes[1:-1].index(node2) + 1
    nodes[n2], nodes[n1] = nodes[n1], nodes[n2]
    return nodes


def node_swap(cycle, random_swap=False):
    if len(cycle) - 1 <= 3:
        return None, -1
    best_swap_result = None
    best_cycle = None

    if random_swap:
        i = random.randint(1, len(cycle) - 4)
        j = random.randint(i+1, len(cycle) - 3)
        swaped_cycle, swap_result = calculate_node_swap(cycle, i, j)
        return swaped_cycle, swap_result

    for i in range(1, len(cycle) - 3):
        for j in range(i + 1, len(cycle) - 2):
            swaped_cycle, swap_result = calculate_node_swap(cycle, i,j)

            if best_swap_result is None or best_swap_result < swap_result:
                best_swap_result = swap_result
                best_cycle = swaped_cycle

    return best_cycle, best_swap_result


def calculate_node_swap(cycle, i, j):
    before_delta_around_node1 = distance(cycle[i - 1], cycle[i]) + distance(cycle[i], cycle[i + 1])
    before_delta_around_node2 = distance(cycle[j - 1], cycle[j]) + distance(cycle[j], cycle[j + 1])
    before_delta_gain = (before_delta_around_node1 + before_delta_around_node2) * COST_WEIGHT
    swaped_cycle = reverse_nodes(cycle.copy(), cycle[i], cycle[j])
    after_delta_around_node1 = distance(swaped_cycle[i - 1], swaped_cycle[i]) + distance(swaped_cycle[i],
                                                                                         swaped_cycle[i + 1])
    after_delta_around_node2 = distance(swaped_cycle[j - 1], swaped_cycle[j]) + distance(swaped_cycle[j],
                                                                                         swaped_cycle[j + 1])
    after_delta_gain = (after_delta_around_node1 + after_delta_around_node2) * COST_WEIGHT
    swap_result = before_delta_gain - after_delta_gain

    return swaped_cycle, swap_result


def simulated_annealing(nodes):
    start = time.time()
    L = 1000
    T0 = 75
    Tk = 1
    alpha = 0.98

    random_solution = generateRandomSolution(nodes.copy())
    nodes = list(set(nodes) - set(random_solution[0]))
    best_solution = random_solution[0]
    best_result = random_solution[1]
    best_global_solution = best_solution
    best_global_result = best_result

    T = T0
    while T > Tk:
        for i in range(0, L):
            new_nodes, new_solution, new_result = get_random_neighbour_solution(nodes.copy(), best_solution.copy(), best_result)
            if new_result > best_result:
                best_solution = new_solution
                nodes = new_nodes
                best_result = new_result
            elif math.exp((new_result - best_result) / T) > random.uniform(0, 1):
                best_solution = new_solution
                nodes = new_nodes
                best_result = new_result

            if best_result > best_global_result:
                best_global_result = best_result
                best_global_solution = best_solution

        T = T * alpha
        #print('temp: ' + str(T))

    duration = time.time() - start
    return best_global_solution, best_global_result, duration

flatten = lambda l: [item for sublist in l for item in sublist]


def rewrite_cycle_to_start_at(cycle, starting_node):
    index = cycle.index(starting_node)

    if index == 0:
        return cycle

    cycle_first_half = cycle[index:]
    cycle_second_half = cycle[1:index]
    new_cycle = cycle_first_half + cycle_second_half + [cycle_first_half[0]]
    return new_cycle


def unify_cycles(cycle_1, cycle_2):
    for node in cycle_1:
        if node in cycle_2:
            new_cycle_1 = rewrite_cycle_to_start_at(cycle_1, node)
            new_cycle_2 = rewrite_cycle_to_start_at(cycle_2, node)
            return new_cycle_1, new_cycle_2
    #TODO this might crash
    return None


def is_sublist(list, sublist):
    return ''.join(map(str, [x.id for x in sublist])) in ''.join(map(str, [x.id for x in list])) or ''.join(map(str, [x.id for x in sublist]))[::-1] in ''.join(map(str, [x.id for x in list]))


def range_is_already_covered(found_ranges, new_range):
    for found_range in found_ranges:
        if new_range[0] >= found_range[0] and new_range[1] <= found_range[1]:
            return True
    return False


def find_common_paths(cycle_1, cycle_2):
    cycle_1, cycle_2 = unify_cycles(cycle_1, cycle_2)
    common_parts_ranges = []

    for starting_index in range(len(cycle_1) - 1):
        for end_index in range(len(cycle_1) - 1, starting_index, -1):
            cycle_1_part = cycle_1[starting_index:end_index]
            new_range = (starting_index, end_index)
            if is_sublist(cycle_2, cycle_1_part) and not range_is_already_covered(common_parts_ranges, new_range):
                common_parts_ranges.append(new_range)
                break

    return [cycle_1[found_range[0]:found_range[1]] for found_range in common_parts_ranges]


def get_unused_nodes(cycle_1, cycle_2, common_paths):
    common_paths_nodes = set(flatten(common_paths))

    all_nodes = set(cycle_1) | set(cycle_2)

    return [[x] for x in list(all_nodes - common_paths_nodes)]


def recombine(cycle_1, cycle_2):
    common_paths = find_common_paths(cycle_1, cycle_2)

    additional_nodes_count = random.randint(min(len(cycle_1) - 1, len(cycle_2) - 1),
                                            max(len(cycle_1) - 1, len(cycle_2) - 1)) - len(flatten(common_paths))

    unused = get_unused_nodes(cycle_1, cycle_2, common_paths)
    random.shuffle(unused)

    unused_to_add = unused[:additional_nodes_count]

    new_cycle_parts = common_paths + unused_to_add
    random.shuffle(new_cycle_parts)

    for i in range(len(new_cycle_parts)):
        if len(new_cycle_parts[i]) > 1 and random.choice([True, False]):
            reversed = new_cycle_parts[i][::-1]
            new_cycle_parts[i] = reversed

    new_cycle = flatten(new_cycle_parts)
    new_cycle = new_cycle + [new_cycle[0]]

    return new_cycle


def lab_3_results():
    nodes = read_data("./data")
    multiple_start_times = []
    multiple_start_results = []
    best_multiple_start_solution = None
    best_multiple_start_result = None
    iterated_ls_results = []
    best_iterated_ls_solution = None
    best_iterated_ls_result = None
    simulated_annealing_results = []
    best_simulated_annealing_solution = None
    best_simulated_annealing_result = None
    simulated_annealing_times = []

    for i in range(0, 10):
        print('MultipleStart LS')
        solution, duration = multiple_start_local_search(nodes.copy())
        if verify_solution(solution[0], solution[1]) > 1:
            raise ValueError("Node path verification failed")
        multiple_start_times.append(duration)
        multiple_start_results.append(solution[1])

        if best_multiple_start_solution is None or solution[1] > best_multiple_start_result:
            best_multiple_start_solution = solution[0]
            best_multiple_start_result = solution[1]

    stop_time = np.mean(multiple_start_times)
    #stop_time = 6

    for i in range(0, 10):
        print('Iterated LS')
        solution = iterated_local_search(nodes.copy(), stop_time)
        if verify_solution(solution[0], solution[1]) > 1:
            raise ValueError("Node path verification failed")
        iterated_ls_results.append(solution[1])
        if best_iterated_ls_solution is None or solution[1] > best_iterated_ls_result:
            best_iterated_ls_solution = solution[0]
            best_iterated_ls_result = solution[1]

    for i in range(0, 10):
        print('Simulated annealing LS')
        solution = simulated_annealing(nodes.copy())
        if verify_solution(solution[0], solution[1]) > 1:
            raise ValueError("Node path verification failed")
        simulated_annealing_results.append(solution[1])
        simulated_annealing_times.append(solution[2])
        if best_simulated_annealing_solution is None or solution[1] > best_simulated_annealing_result:
            best_simulated_annealing_solution = solution[0]
            best_simulated_annealing_result = solution[1]

    print_result(nodes, best_multiple_start_solution, best_multiple_start_result, 'MultipleStart LS')
    print('MultipleStart LS - best: {}, worst: {}, average: {}. Times: min {}, max {}, avg {}'.format(best_multiple_start_result, min(multiple_start_results), np.mean(multiple_start_results), min(multiple_start_times), max(multiple_start_times), np.mean(multiple_start_times)))
    print(list(map(lambda node: int(node.id), best_multiple_start_solution)))

    print_result(nodes, best_iterated_ls_solution, best_iterated_ls_result, 'Iterated LS')
    print('Iterated LS - best: {}, worst: {}, average: {}. Stop time: {}'.format(best_iterated_ls_result, min(iterated_ls_results), np.mean(iterated_ls_results), stop_time))
    print(list(map(lambda node: int(node.id), best_iterated_ls_solution)))

    print_result(nodes, best_simulated_annealing_solution, best_simulated_annealing_result, 'Simulated annealing')
    print('Simulated annealing - best: {}, worst: {}, average: {}. Times: min {}, max {}, avg {}'.format(best_simulated_annealing_result, min(simulated_annealing_results), np.mean(simulated_annealing_results), min(simulated_annealing_times), max(simulated_annealing_times), np.mean(simulated_annealing_times)))
    print(list(map(lambda node: int(node.id), best_simulated_annealing_solution)))


def main():
    # node_0 = Node(0)
    # node_1 = Node(1)
    # node_2 = Node(2)
    # node_3 = Node(3)
    # node_4 = Node(4)
    # cycle_1 = [node_0, node_1, node_2, node_4, node_0]
    # cycle_2 = [node_1, node_4, node_3, node_2, node_1]
    # 
    # result = recombine(cycle_1, cycle_2)
    # print([x.id for x in result])
    
    # lab_3_results()

    # nodes = read_data("./data")
    # best_nearest_neighbour_solution = None
    # best_nearest_neighbour_result = None
    # nearest_neighbour_times = []
    # nearest_neighbour_results = []
    # best_cycle_expansion_solution = None
    # best_cycle_expansion_result = None
    # cycle_expansion_times = []
    # cycle_expansion_results = []
    # best_cycle_expansion_with_regret_solution = None
    # best_cycle_expansion_with_regret_result = None
    # cycle_expansion_with_regret_times = []
    # cycle_expansion_with_regret_results = []
    # best_random_solution = None
    # best_random_result = None
    # random_times = []
    # random_results = []

    # for starting_index in range(0, len(nodes)):
    #     print(starting_index)
    #     print('NN')
    #     solution = nearest_neighbour(nodes.copy(), starting_index)
    #     locals_solution = enhanceSolutionWithLocals(solution[0], list(set(nodes.copy()) - set(solution[0])), solution[1])
    #     nearest_neighbour_results.append(locals_solution[1])
    #     nearest_neighbour_times.append(sum(locals_solution[2]))
    #     if best_nearest_neighbour_solution is None or locals_solution[1] > best_nearest_neighbour_result:
    #         best_nearest_neighbour_solution = locals_solution[0]
    #         best_nearest_neighbour_result = locals_solution[1]
    #
    #     print('CE')
    #     solution = cycle_expansion(nodes.copy(), starting_index)
    #     locals_solution = enhanceSolutionWithLocals(solution[0], list(set(nodes.copy()) - set(solution[0])), solution[1])
    #     cycle_expansion_results.append(locals_solution[1])
    #     cycle_expansion_times.append(sum(locals_solution[2]))
    #     if best_cycle_expansion_solution is None or locals_solution[1] > best_cycle_expansion_result:
    #         best_cycle_expansion_solution = locals_solution[0]
    #         best_cycle_expansion_result = locals_solution[1]
    #
    #     print('CE+R')
    #     solution = cycle_expansion_with_regret(nodes.copy(), starting_index)
    #     locals_solution = enhanceSolutionWithLocals(solution[0], list(set(nodes.copy()) - set(solution[0])), solution[1])
    #     cycle_expansion_with_regret_results.append(locals_solution[1])
    #     cycle_expansion_with_regret_times.append(sum(locals_solution[2]))
    #     if best_cycle_expansion_with_regret_solution is None or locals_solution[1] > best_cycle_expansion_with_regret_result:
    #         best_cycle_expansion_with_regret_solution = locals_solution[0]
    #         best_cycle_expansion_with_regret_result = locals_solution[1]
    #
    #     print('RAND')
    #     solution = generateRandomSolution(nodes.copy())
    #     locals_solution = enhanceSolutionWithLocals(solution[0], list(set(nodes.copy()) - set(solution[0])), solution[1])
    #     random_results.append(locals_solution[1])
    #     random_times.append(sum(locals_solution[2]))
    #     if best_random_solution is None or locals_solution[1] > best_random_result:
    #         best_random_solution = locals_solution[0]
    #         best_random_result = locals_solution[1]
    #
    # print_result(nodes, best_nearest_neighbour_solution, best_nearest_neighbour_result, 'Nearest neighbour')
    # print('Nearest neigbour - best: {}, worst: {}, average: {}. Times: min {}, max {}, avg {}'.format(best_nearest_neighbour_result, min(nearest_neighbour_results), np.mean(nearest_neighbour_results), min(nearest_neighbour_times), max(nearest_neighbour_times), np.mean(nearest_neighbour_times)))
    # print(list(map(lambda node: int(node.id), best_nearest_neighbour_solution)))
    # print_result(nodes, best_cycle_expansion_solution, best_cycle_expansion_result, 'Cycle expansion')
    # print('Cycle expansion - best: {}, worst: {}, average: {}. Times: min {}, max {}, avg {}'.format(best_cycle_expansion_result, min(cycle_expansion_results), np.mean(cycle_expansion_results), min(cycle_expansion_times), max(cycle_expansion_times), np.mean(cycle_expansion_times)))
    # print(list(map(lambda node: int(node.id), best_cycle_expansion_solution)))
    # print_result(nodes, best_cycle_expansion_with_regret_solution, best_cycle_expansion_with_regret_result, 'Cycle expansion with regret')
    # print('Cycle expansion with regret - best: {}, worst: {}, average: {}. Times: min {}, max {}, avg {}'.format(best_cycle_expansion_with_regret_result, min(cycle_expansion_with_regret_results), np.mean(cycle_expansion_with_regret_results), min(cycle_expansion_with_regret_times), max(cycle_expansion_with_regret_times), np.mean(cycle_expansion_with_regret_times)))
    # print(list(map(lambda node: int(node.id) , best_cycle_expansion_with_regret_solution)))
    # print_result(nodes, best_cycle_expansion_with_regret_solution, best_cycle_expansion_with_regret_result,'Random')
    # print('Random - best: {}, worst: {}, average: {}. Times: min {}, max {}, avg {}'.format(best_random_result, min(random_results),np.mean(random_results), min(random_times),max(random_times), np.mean(random_times)))
    # print(list(map(lambda node: int(node.id), best_random_solution)))



main()
