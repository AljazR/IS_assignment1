import pygad
import numpy as np
import time
import math
from mazes import *

# Conversion from array of strings to integer numpy matrix
char2num = {"#": 0, ".": 1, "S": 2, "E": 3}
def convert(mazeChars):
    return np.array([[char2num[char] for char in line] for line in mazeChars], dtype=np.int8)

# Choose the maze
original_maze = maze7
maze = convert(original_maze)

# Get the coordinates of 'S' and 'E': (row, column)
start = list(zip(*np.where(maze == 2)))[0]
end = list(zip(*np.where(maze == 3)))[0]

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
    

# Define mutatoion function
def mutation_func():
    a = 1

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

# Creating an instance of the GA class inside the ga module.
ga_instance = pygad.GA(num_generations=2000,
                       num_parents_mating=10, 
                       sol_per_pop=50, 
                       num_genes=np.where(maze == 1)[0].size * 2,
                       gene_space=[0, 1, 2, 3],
                       keep_elitism=5,
                       random_seed=2,
                       stop_criteria="saturate_500", # stop evolution, if the fitness does not change for 100 consecutive generations.
                       crossover_type="uniform",
                    #    crossover_type=crossover_func,
                    #    mutation_type=mutation_func,
                       fitness_func=fitness_func_aljaz,
                       on_generation=on_generation,
                       on_stop=on_stop)

# Running the GA to optimize the parameters of the function.
ga_instance.run()

# After the generations complete, some plots are showed that summarize the how the outputs/fitenss values evolve over generations.
# ga_instance.plot_fitness()