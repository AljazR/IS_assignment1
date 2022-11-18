import pygad
import numpy as np
import time
import math
from mazes import *
import random

# Conversion from array of strings to integer numpy matrix
char2num = {"#": 0, ".": 1, "S": 2, "E": 3}
def convert(mazeChars):
    return np.array([[char2num[char] for char in line] for line in mazeChars], dtype=np.int32)

move_letters = ["U", "D", "L", "R"]

# Choose the maze
original_maze = maze4
maze = convert(original_maze)

# Get the coordinates of 'S' and 'E': (row, column)
start = list(zip(*np.where(maze == 2)))[0]
end = list(zip(*np.where(maze == 3)))[0]

# Moves before hitting wall or to the end
def valid_solution_part(moves: np.ndarray, maze: np.ndarray):
    a, b = start
    rows, cols = indexes_visited(moves, a, b)
    out_of_maze1 = np.nonzero(np.logical_or(rows < 0, rows >= maze.shape[0]))[0]
    out_of_maze2 = np.nonzero(np.logical_or(cols < 0, cols >= maze.shape[1]))[0]
    n_valid_moves = moves.size
    if out_of_maze1.size > 0 :
        n_valid_moves = min(out_of_maze1[0], n_valid_moves)
    if out_of_maze2.size > 0:
        n_valid_moves = min(out_of_maze2[0], n_valid_moves)
    field_values = maze[rows[:n_valid_moves], cols[:n_valid_moves]]
    walls = np.nonzero(field_values == 0)[0]
    if walls.size > 0:
        n_valid_moves = walls[0] # index of first move, that hit the wall
    finished = False
    end = np.nonzero(field_values[:n_valid_moves] == char2num["E"])[0]
    if end.size > 0:
        finished = True
        n_valid_moves = end[0] + 1
    different_fields = np.zeros_like(maze)
    different_fields[rows[:n_valid_moves], cols[:n_valid_moves]] = 1
    dots_travelled = different_fields.sum()
    return moves[:n_valid_moves], finished, dots_travelled

def fitness_func_klemen(solution, solution_idx):
    moves, finished, dots_travelled = valid_solution_part(solution, maze)
    n_moves = moves.size
    if finished:
        return 1000 - n_moves
    else:
        return dots_travelled - n_moves / 1000


def moves_before_end(solution):
    # indexes of rows and columns
    rows, cols = indexes_visited(solution, start[0], start[1])

    # indexes of moves, not rows an columns
    out_of_maze_rows = np.nonzero(np.logical_or(rows < 0, rows >= maze.shape[0]))[0]
    out_of_maze_cols = np.nonzero(np.logical_or(cols < 0, cols >= maze.shape[1]))[0]

    # indexes of moves that go out of maze
    out = np.unique(np.concatenate((out_of_maze_cols, out_of_maze_rows)))

    # indexes of moves that stay in maze
    in_maze = np.arange(solution.shape[0])
    in_maze[out] = -1
    in_maze = in_maze[np.nonzero(in_maze != -1)]

    # values of fields visited to the end (excluding thos out of maze)
    fields = maze[rows[in_maze], cols[in_maze]]

    # indexes of moves, that go to end
    ends = np.nonzero(fields == 3)[0]
    
    # number of all moves before first end
    moves_to_end = solution.size if ends.size == 0 else ends[0]
    return moves_to_end


def fitness_func_klemen2(solution, solution_idx):
    moves, finished, dots_travelled = valid_solution_part(solution, maze)

    # indexes of rows and columns
    rows, cols = indexes_visited(solution, start[0], start[1])

    # indexes of moves, not rows an columns
    out_of_maze_rows = np.nonzero(np.logical_or(rows < 0, rows >= maze.shape[0]))[0]
    out_of_maze_cols = np.nonzero(np.logical_or(cols < 0, cols >= maze.shape[1]))[0]

    # indexes of moves that go out of maze
    out = np.unique(np.concatenate((out_of_maze_cols, out_of_maze_rows)))

    # indexes of moves that stay in maze
    in_maze = np.arange(solution.shape[0])
    in_maze[out] = -1
    in_maze = in_maze[np.nonzero(in_maze != -1)]

    # values of fields visited to the end (excluding thos out of maze)
    fields = maze[rows[in_maze], cols[in_maze]]

    # indexes of moves, that go to end
    ends = np.nonzero(fields == 3)[0]
    
    # number of all moves before first end
    moves_to_end = solution.size if ends.size == 0 else ends[0]
    
    # new values calculated only on moves before end
    out = out[out < moves_to_end]
    in_maze = np.arange(moves_to_end)
    in_maze[out] = -1
    in_maze = in_maze[np.nonzero(in_maze != -1)]
    fields = maze[rows[in_maze], cols[in_maze]]

    # number of moves before end going into the wall
    walls = np.sum(fields == 0)

    # number of moves before end going on valid field
    dots = np.sum(fields == 1)

    # bit mask of all fields (including walls) before end
    fields_travelled = np.zeros_like(maze)
    fields_travelled[rows[in_maze], cols[in_maze]] = 1

    # number of different dots visited before end
    different_dots = ((maze == 1) * fields_travelled).sum()

    if finished:
        return 1000000 - moves.size
    else:
        return out.size * (-10000) + walls * (-100) + different_dots * 20



# Define fitness function
def fitness_func_aljaz(solution, solution_idx):
    score = 0
    row = start[0]
    col = start[1]
    height, width = maze.shape 
    last_move = -1
    number_of_moves = 0

    # loop over every step of solution and penalize for moving back and forth
    for move in solution:
        number_of_moves += 1
        if move == 0:
            row -= 1
            if last_move == 1:
                score -= 200
        elif move == 1:
            row += 1
            if last_move == 0:
                score -= 200
        elif move == 2:
            col -= 1
            if last_move == 3:
                score -= 200
        elif move == 3:
            col += 1
            if last_move == 2:
                score -= 200

        last_move = move
        if (row >= height or row < 0) or (col >= width or col < 0) or maze[row,col] == 0: # out of bounds or on wall
            return score - 1000

        elif maze[row,col] == 1: # on field
            distance = math.sqrt((col - end[1])**2 + (row - end[0])**2) # distance between current point and end
            max_distance = math.sqrt(2 * (height * width)) # maximal distance in maze
            score += 10 * (max_distance - distance)

        elif maze[row,col] == 3: # on end
            score += 100000 - number_of_moves * 1000
            return score

    return score

# Define crossover function
def crossover_func(parents, offspring_size, ga_instance):
    offspring = []
    id_parent = 0
    while len(offspring) != offspring_size[0]:
        parent1 = parents[id_parent % parents.shape[0], :].copy()
        parent2 = parents[(id_parent + 1) % parents.shape[0], :].copy()

        # Example for sigle-point crossover
        random_split_point = np.random.choice(range(offspring_size[1]))
        parent1[random_split_point:] = parent2[random_split_point:]

        offspring.append(parent1)

        id_parent += 1

    return np.array(offspring)

# Initial population with all solutions arriving to end after aproximately half of possible moves,
# but going through walls and out of maze
# num_genes mut be at least two times bigger than distance between start and end
def initial_to_end_through_walls(sol_per_pop, num_genes, start, end, maze_shape):
    col, row = start
    population = np.zeros((sol_per_pop, num_genes), dtype=np.int32)
    for i in range(sol_per_pop):
        distance_row = end[0] - start[0]
        distance_col = end[1] - start[1]
        free_moves = num_genes // 2 - abs(distance_col) - abs(distance_row)
        if distance_row > 0:
            up = free_moves // 4
            down = free_moves // 4 + distance_row
        else:
            up = free_moves // 4 - distance_row
            down = free_moves // 4
        if distance_col > 0:
            left = free_moves // 4
            right = free_moves // 4 + distance_col
        else:
            left = free_moves // 4 - distance_col
            right = free_moves // 4
        moves = np.zeros(num_genes)
        moves[:up + down + left + right] = np.concatenate((np.zeros(up), 
            np.ones(down), np.ones(left) * 2, np.ones(right) * 3))
        np.random.shuffle(moves[:up + down + left + right])
        for j in range(free_moves % 4):
            moves[-1-j] = random.randint(0, 3)
        population[i,:] = moves
    return population

# calculates row and column indexes after all moves
def indexes_visited(moves, start_col, start_row):
    moves_sum = np.array([moves == 0, moves == 1, moves == 2, moves == 3]).cumsum(axis=1)
    rows = start_col - moves_sum[0,:] + moves_sum[1,:]
    cols = start_row - moves_sum[2,:] + moves_sum[3,:]
    return rows, cols

"""
        for j in range(num_genes):
            # 0 1 2 3 ~ U D L R
            axis = random.randint(0, 1) # 0: U/D, 1: L/R
            if distance > num_genes - j:
                direction = random.randint(0, 1) # 0: U/L, 1: D/R
            else:
                direction = 0 

            move = axis * 2 + direction
            col += (-1) ** (1 - direction) * (1 - axis)
            row += (-1) ** (1 - direction) * axis
            distance = abs(end[0] - col) + abs(end[1] - row)
            population[i,j] = move
"""





# Randomly select a different move
def random_new(move):
    return (move + random.randint(1, 3)) % 4

# Randomly selects a new perpendicular move
def random_perpendicular(move):
    return (move + (-1) ** random.randint(1, 2)) % 4

# Define mutatoion function
def mutation_func_two_adjacent_genes(offspring, ga_instance):
    for chromosome_idx in range(offspring.shape[0]):
        offspring[chromosome_idx, :] = mutate_two_adjacent_genes(offspring[chromosome_idx, :])
    return offspring

# Mutate two adjacent genese like so, that moves behind mutation stay on tha same path
def mutate_two_adjacent_genes(chromosome: np.ndarray):
    #valid_part, _, _ = valid_solution_part(chromosome, maze)
    #mutation_location = random.randint(0, chromosome.size - 2)
    mutation_location = random.randint(0, max(moves_before_end(chromosome) - 2, 0))
    #mutation_location = valid_part.size
    gene1, gene2 = chromosome[mutation_location], chromosome[mutation_location + 1]

    # eliminates a back and forth move, adds 2 new moves to the end
    if ((gene1 + 2) % 4) == gene2:
        chromosome[mutation_location:-2] = chromosome[mutation_location + 2:]
        chromosome[-2] = random.randint(0, 4)
        chromosome[-1] = random.randint(0, 4)

    # "Rotates" a corner: LU to UL, DL to LD ...
    elif gene1 % 2 != gene2 % 2:
        chromosome[mutation_location] = gene2
        chromosome[mutation_location + 1] = gene1
    
    # Goes around: LL -> ULLD / DLLU, UU -> RUUL / LUUR ...
    else:
        # To close to the end, randomly choose new move
        if mutation_location > chromosome.size - 4:
            chromosome[mutation_location] = random_new(gene1)
        # Goes around, deleting last two moves
        else:
            r_p = random_perpendicular(gene1)
            # Shifts moves to right
            chromosome[mutation_location + 4 :] = chromosome[mutation_location + 2 : -2]
            # Inserts moves to go around
            chromosome[mutation_location : mutation_location + 4] = np.array((r_p, gene1, gene2, (r_p + 2) % 4))
    return chromosome


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

# Runs at the end of GA
def on_stop(ga_instance, fitness):
    solution = np.asarray(ga_instance.best_solution()[0], dtype=int).tolist()
    # solution = [2,2,2,0,0,3,3,3,0,0,2,1,2,3,3] # Example of solution for maze2

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


population = initial_to_end_through_walls(100, np.where(maze == 1)[0].size * 4, start, end, maze.shape)

# Creating an instance of the GA class inside the ga module.
ga_instance = pygad.GA(num_generations=2000,
                       num_parents_mating=10, 
                       #sol_per_pop=50,
                       initial_population=population,
                       #num_genes=np.where(maze == 1)[0].size * 2,
                       gene_space=[0, 1, 2, 3],
                       keep_elitism=5,
                       #random_seed=2,
                       stop_criteria="saturate_200", # stop evolution, if the fitness does not change for 100 consecutive generations.
                       crossover_type=None,
                    #    crossover_type=crossover_func,
                       mutation_type=mutation_func_two_adjacent_genes,
                       fitness_func=fitness_func_klemen2,
                       on_generation=on_generation,
                       on_stop=on_stop)

# Running the GA to optimize the parameters of the function.
ga_instance.run()

# After the generations complete, some plots are showed that summarize the how the outputs/fitenss values evolve over generations.
# ga_instance.plot_fitness()