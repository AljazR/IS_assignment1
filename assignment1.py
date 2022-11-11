import pygad
import numpy as np
import time
from mazes import *

# Conversion from array of strings to integer numpy matrix
char2num = {"#": 0, ".": 1, "S": 2, "E": 3}
def convert(mazeChars):
    return np.array([[char2num[char] for char in line] for line in mazeChars], dtype=np.int8)

# Choose the maze
original_new = maze2
maze = convert(original_new)

# Get the coordinates of 'S' and 'E'
start = list(zip(*np.where(maze == 2)))[0]
end = list(zip(*np.where(maze == 3)))[0]

# Set parameters for GA
num_generations = 1000 # Number of generations.
num_parents_mating = 7 # Number of solutions to be selected as parents in the mating pool.
sol_per_pop = 50 # Number of solutions in the population.
gene_space = [0, 1, 2, 3] # [U, D, L, R]
num_genes = np.where(maze == 1)[0].size # Set to number of dots in the maze.

def fitness_func(solution, solution_idx):
    score = 0
    return score

last_fitness = 0
def on_generation(ga_instance):
    global last_fitness
    # print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    # print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
    # print("Change     = {change}".format(change=ga_instance.best_solution()[1] - last_fitness))
    generation = ga_instance.generations_completed
    fitness = ga_instance.best_solution()[1]
    change = ga_instance.best_solution()[1] - last_fitness
    print("Generation: %-10d Fitness: %-10d Change: %-10d" % (generation, fitness, change))
    last_fitness = ga_instance.best_solution()[1]

def on_stop(ga_instance, fitness):
    solution = np.asarray(ga_instance.best_solution()[0], dtype=int).tolist()
    # solution = [2,2,2,0,0,3,3,3,0,0] # Example of solution for maze2
    
    num2char = ['U','D','L','R']
    index_end = 0
    solution = [5] + solution # So 'X' starts on 'S'
    x = start[0]
    y = start[1]

    print()
    [print(i) for i in original_new]
    time.sleep(0.7)

    # Set x or y
    for i in solution:
        index_end += 1

        if i == 0:
            x -= 1
        elif i == 1:
            x += 1
        elif i == 2:
            y -= 1
        elif i == 3:
            y += 1
        
        # Erasing previous maze
        for j in range(maze.shape[1]):
            print('\033[1A', end='\x1b[2K') # line_up='\033[1A' and line_clear='\x1b[2K'

        # Write out the new maze
        if x < maze.shape[0] and y < maze.shape[1]:
            new_maze = np.copy(original_new)
            new_maze[x] = new_maze[x][:y] + 'X' + new_maze[x][y+1:]
            [print(i) for i in new_maze]
            
            if maze[x,y] == 3:
                solution = solution[1:index_end]
                print('\nSolution: ', end='')
                [print(num2char[i], end ="") for i in solution]
                print('\n')
                return
        else:
            [print(i) for i in original_new]
        
        time.sleep(0.5)

    # Print solution
    solution = solution[1:]
    print('\nSolution:', end='')
    [print(num2char[i], end ="") for i in solution]
    print('\n')
    return

# Creating an instance of the GA class inside the ga module.
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating, 
                       fitness_func=fitness_func,
                       sol_per_pop=sol_per_pop, 
                       num_genes=num_genes,
                       gene_space=gene_space,
                       on_generation=on_generation,
                       on_stop=on_stop)

# Running the GA to optimize the parameters of the function.
ga_instance.run()

# After the generations complete, some plots are showed that summarize the how the outputs/fitenss values evolve over generations.
ga_instance.plot_fitness()
