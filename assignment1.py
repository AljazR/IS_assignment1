import pygad
import numpy as np
from mazes import *

# ga_instance = pygad.GA(num_generations=3,
#                        num_parents_mating=5,
#                        fitness_func=fitness_function,
#                        sol_per_pop=10,
#                        num_genes=len(function_inputs),
#                        on_start=on_start,
#                        on_fitness=on_fitness,
#                        on_parents=on_parents,
#                        on_crossover=on_crossover,
#                        on_mutation=on_mutation,
#                        on_generation=on_generation,
#                        on_stop=on_stop)


def fitness_func(solution, solution_idx):
    # TO-DO
    return 

fitness_function = fitness_func

num_generations = 100 # Number of generations.
num_parents_mating = 7 # Number of solutions to be selected as parents in the mating pool.
sol_per_pop = 50 # Number of solutions in the population.
num_genes = 0 # TO-DO: Number of dots in the maze
gene_space = [0, 1, 2, 3] # [U, D, L, R]

last_fitness = 0
def callback_generation(ga_instance):
    global last_fitness
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
    print("Change     = {change}".format(change=ga_instance.best_solution()[1] - last_fitness))
    last_fitness = ga_instance.best_solution()[1]

def on_stop():
    # TO-DO: presentcija resitve
    return

# Creating an instance of the GA class inside the ga module. Some parameters are initialized within the constructor.
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating, 
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop, 
                       num_genes=num_genes,
                       gene_space=gene_space,
                       on_generation=callback_generation,
                       on_stop=on_stop)

# Running the GA to optimize the parameters of the function.
ga_instance.run()

# After the generations complete, some plots are showed that summarize the how the outputs/fitenss values evolve over generations.
ga_instance.plot_fitness()
