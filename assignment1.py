import pygad
import numpy as np
import time
from mazes import *
import random

# Conversion from array of strings to integer numpy matrix
char2num = {"#": 0, ".": 1, "S": 2, "E": 3, "T": 4}
def convert(mazeChars):
    return np.array([[char2num[char] for char in line] for line in mazeChars], dtype=np.int32)

move_letters = ["U", "D", "L", "R"]

# Choose the maze
original_maze = maze8
maze = convert(original_maze)

# Get the coordinates of 'S' and 'E': (row, column)
start = list(zip(*np.where(maze == 2)))[0]
end = list(zip(*np.where(maze == 3)))[0]

# Runs after every generation of GA
last_fitness = 0
def on_generation(ga_instance):
    generation = ga_instance.generations_completed
    if generation % 1 == 0:
        global last_fitness
        fitness = ga_instance.best_solution()[1]
        change = ga_instance.best_solution()[1] - last_fitness
        print("Generation: %-10d Fitness: %-10d Change: %-10d" % (generation, fitness, change))
    last_fitness = ga_instance.best_solution()[1]

# calculates points where we can decide where to go
# returns array representing next point after certain move
# and moves to those points
def preprocess_maze(maze):
    not_walls = maze != 0
    dots = maze == 1
    neighbour_fields = np.zeros_like(maze)
    neighbour_fields[:-1,:] += not_walls[1:,:]
    neighbour_fields[1:,:] += not_walls[:-1,:]
    neighbour_fields[:,:-1] += not_walls[:,1:]
    neighbour_fields[:,1:] += not_walls[:,:-1]
    neighbour_fields *= not_walls
    decision_points = np.array(np.logical_or(maze >= 2, neighbour_fields >= 3), dtype=np.int32)
    points_row, points_col = np.nonzero(decision_points)
    num_points = points_row.size
    decision_points[:,:] = -1
    decision_points[points_row, points_col] = np.arange(num_points)

    # next points in all 4 directions from all decision points
    # -1 where there is no path in given direction
    next_points = np.zeros((num_points, 4), dtype=np.int32) - 1
    # list of lists shaped like next_points containing
    # np.arrays with moves to next points for each point and direction
    moves_to_next_points = []
    for i in range(num_points):
        moves_to_next_points.append([])
        row = points_row[i]
        col = points_col[i]
        for move in range(0, 4):
            point, moves = next_point(maze, decision_points, row, col, move)
            next_points[i, move] = point
            moves_to_next_points[i].append(moves)

    start_point = decision_points[maze == 2][0]
    end_point = decision_points[maze == 3][0]
    treasures = decision_points[maze == 4]

    """
    print(maze)
    #print(next_point(maze, decision_points, points_row[5], points_col[5], 1))   
    print()
    print(decision_points)
    print(points_col)
    print(next_points)
    print(moves_to_next_points)
    """
    return start_point, end_point, treasures, decision_points, next_points, moves_to_next_points

# returns first possible next decision point in given direction and moves to it
def next_point(maze, decision_points, row, col, move):
    moves = [move]
    if move == 0:
        row -= 1
    elif move == 1:
        row += 1
    elif move == 2:
        col -= 1
    elif move == 3:
        col += 1
    if row < 0 or row >= maze.shape[0] or col < 0 or col >= maze.shape[1] \
        or maze[row, col] == 0:
        return -1, np.zeros(0)
    previous_move = move
    while decision_points[row, col] == -1:
        found_move = False
        for move in range(4):
            # going back
            if move != previous_move and move // 2 == previous_move // 2:
                continue
            # new field
            if move == 0:
                row -= 1
            elif move == 1:
                row += 1
            elif move == 2:
                col -= 1
            elif move == 3:
                col += 1
            # wall
            if row < 0 or row >= maze.shape[0] or col < 0 or col >= maze.shape[1] \
                or maze[row, col] == 0:
                # reverse move
                if move == 0:
                    row += 1
                elif move == 1:
                    row -= 1
                elif move == 2:
                    col += 1
                elif move == 3:
                    col -= 1
            # valid move
            else:
                moves.append(move)
                previous_move = move
                found_move = True
                break
        # no valid moves indicating dead end
        if not found_move:
            return -1, np.zeros(0)
    return decision_points[row, col], np.array(moves)

start_point, end_point, treasures, decision_points,  next_points, moves_to_next_points = preprocess_maze(maze)

# fitness
def fitness_points(solution, solution_idx):
    length = 0
    found_treasures = set()
    current_point = start_point
    for i in range(solution.size):
        current_move = int(solution[i])
        length += moves_to_next_points[current_point][current_move].size
        current_point = next_points[current_point, current_move]
        if current_point == end_point:
            return 1000000 + len(found_treasures) * 10000 - length
        elif np.any(treasures == current_point):
            found_treasures.add(current_point)
            pass
    return len(found_treasures) * 10000 + length

# mutation
# changes one move and randomly selects new valid sequence of moves after that
def mutation_points(offspring, ga_instance):
    for i in range(offspring.shape[0]):
        mutation_locaion = random.randint(0, offspring.shape[1] - 1)
        current_point = start_point
        for j in range(mutation_locaion):
            current_point = next_points[current_point, int(offspring[i, j])]
        for j in range(mutation_locaion, offspring.shape[1]):
            move = random.randint(0, 3)
            point = next_points[current_point, move]
            while point == -1:
                move = random.randint(0, 3)
                point = next_points[current_point, move]
            offspring[i, j] = move
            current_point = point
    return offspring

# returns a sequence of visited decision points
def decision_points_visited(solution):
    visited = np.zeros(solution.size)
    current_point = start_point
    for i in range(solution.size):
        visited[i] = next_points[current_point, int(solution[i])]
        current_point = next_points[current_point, int(solution[i])]
    return visited

# adds new valid moves to the end
def extend(child, new_moves):
    new_child = np.zeros(child.size + new_moves)
    new_child[:child.size] = child
    current_point = start_point
    for i in range(child.size):
        current_point = next_points[current_point, int(child[i])]
    for i in range(new_moves):
        move = random.randint(0, 3)
        point = next_points[current_point, move]
        while point == -1:
            move = random.randint(0, 3)
            point = next_points[current_point, move]
        new_child[child.size + i] = move
        current_point = point
    return new_child

# crossover
# finds mutual point and combines path from start to that poin 
# from first parent and to the end from other
def crossover_points(parents, offspring_size, ga_instance):
    height, width =  maze.shape
    offspring = []
    id_parent = 0
    while len(offspring) != offspring_size[0]:
        # choose two parents
        parent1 = parents[id_parent % parents.shape[0], :].copy()
        parent2 = parents[(id_parent + 1) % parents.shape[0], :].copy()

        # get all decision points for both parents
        points1 = decision_points_visited(parent1)[:-1]
        points2 = decision_points_visited(parent2)[:-1]

        # calculates same points and randomly cooses one of them
        same_points = set(points1).intersection(set(points2))
        same_point = random.sample(same_points, 1)

        # chooses one of the indexes of same_point for both parents
        idx1_same_point = random.sample(set(np.where(points1 == same_point)[0]), 1)[0] + 1
        idx2_same_point = random.sample(set(np.where(points2 == same_point)[0]), 1)[0] + 1

        # splits parents in firt and second part
        if idx1_same_point < idx2_same_point:
            short_first_part = parent1[:idx1_same_point]
            long_first_part = parent2[:idx2_same_point]
            long_second_part = parent1[idx1_same_point:]
            short_second_part = parent2[idx2_same_point:]
        else:
            short_first_part = parent2[:idx2_same_point]
            long_first_part = parent1[:idx1_same_point]
            long_second_part = parent2[idx2_same_point:]
            short_second_part = parent1[idx1_same_point:]

        # size difference
        difference = abs(idx1_same_point - idx2_same_point)

        # creates and adds the first child
        long_child = np.concatenate((long_first_part, long_second_part))
        if difference > 0:
            long_child = long_child[:-difference]
        offspring.append(long_child)

        # creates and adds the second child if there is not enough offspring yet
        if (len(offspring) < offspring_size[0]):
            short_child = np.concatenate((short_first_part, short_second_part))
            short_child = extend(short_child, difference)
            offspring.append(short_child)
        
        """
        print(parent1)
        print(parent2)
        print(long_child)
        print(short_child)
        print(idx1_same_point, idx2_same_point, same_point)
        print(decision_points_visited( parent1))
        print(decision_points_visited( parent2))
        print(decision_points_visited( long_child))
        print(decision_points_visited( short_child))
        print()
        """
        id_parent += 1
    #print(np.array(offspring))
    return np.array(offspring)

# population
# generates sol_per_pop solutions represented as sequence of moves after decision points.
# length of solutions is length_factor times the number of different
# decision points in the maze
def population_points(maze, sol_per_pop, length_factor):
    num_points = next_points.shape[0]
    num_genes = num_points * length_factor
    population = np.zeros((sol_per_pop, num_genes), dtype=np.int32)
    for i in range(sol_per_pop):
        #population[i, 0] = start_point
        current_point = start_point
        # generates a valid sequence of decision points as one solution
        for j in range(0, num_genes):
            move = random.randint(0, 3)
            point = next_points[current_point, move]
            while point == -1:
                move = random.randint(0, 3)
                point = next_points[current_point, move]
            population[i, j] = move
            current_point = point
    """
    for line in original_maze:
        print(line)
    print(decision_points)
    print(population[0,:])
    print(decision_points_visited(population[0,:]))
    """
    return population

def on_stop_points(ga_instance, fitness):
    solution_points = np.array(ga_instance.best_solution()[0], dtype=np.int32)
    print(solution_points)
    solution = []

    # converts solution containing only moves after decision points to list of all moves
    current_point = start_point
    for i in range(solution_points.size):
        for move in moves_to_next_points[current_point][solution_points[i]]:
            solution.append(move)
        current_point = next_points[current_point, solution_points[i]]
        if current_point == end_point:
            break

    # same as before

    num2char = ['U','D','L','R']
    index = len(solution)
    solution = [4] + solution # So 'X' starts on 'S'
    row = start[0]
    col = start[1]
    height, width = maze.shape

    print()
    [print(i) for i in original_maze]
    time.sleep(0.8)

    # Set x or y
    for i,move in enumerate(solution):
        if move == 0:
            row -= 1
        elif move == 1:
            row += 1
        elif move == 2:
            col -= 1
        elif move == 3:
            col += 1
        
        # Erasing previous maze
        for j in range(height):
            print('\033[1A', end='\x1b[2K') # line_up='\033[1A' and line_clear='\x1b[2K'

        # Write out the new maze
        if (0 <= row and row < height) and (0 <= col and col < width):
            new_maze = np.copy(original_maze)
            new_maze[row] = new_maze[row][:col] + 'X' + new_maze[row][col+1:]
            [print(j) for j in new_maze]
        else:
            [print(j) for j in original_maze]

        time.sleep(0.2)
        if (0 <= row and row < height) and (0 <= col and col < width) and maze[row,col] == 3:
            time.sleep(0.2)
            index = i
            break

    # Print solution
    solution = solution[1:]
    print('\nSolution: ', end='')
    for i in solution[:index]:
        print(num2char[i], end ="")
    print('\n')


sol_per_pop = 50
keep_elitism = 10
num_parents_maiting = 20
length_factor = 3
population_p = population_points(maze, sol_per_pop, length_factor)


# Creating an instance of the GA class inside the ga module.
ga_instance = pygad.GA(num_generations=2000,
                       num_parents_mating=num_parents_maiting,
                       #sol_per_pop=sol_per_pop,
                       initial_population=population_p,
                       gene_space=[0, 1, 2, 3],
                       keep_elitism=keep_elitism,
                       #random_seed=2,
                       stop_criteria="saturate_200", # stop evolution, if the fitness does not change for 200 consecutive generations.
                       #crossover_type=None,
                       crossover_type=crossover_points,
                       mutation_type=mutation_points,
                       fitness_func=fitness_points,
                       on_generation=on_generation,
                       on_stop=on_stop_points)

# Running the GA to optimize the parameters of the function.
ga_instance.run()

# After the generations complete, some plots are showed that summarize the how the outputs/fitenss values evolve over generations.
# ga_instance.plot_fitness()