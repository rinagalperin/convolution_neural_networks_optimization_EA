import random
import multiprocessing
import time
import ujson
import numpy as np

from multiprocessing import Process
from deap import creator, base, tools, algorithms
from experiment import Experiment

# our implementation for pair-wise crossover operator
def pair_wise_crossover(return_dict, num_of_process, population_size, gene):
    new_population = []
    for v1, v2 in zip(return_dict[range(0, population_size//2, 2)], return_dict[range(1, population_size//2, 2)]):
        v1 = eval(v1)
        v2 = eval(v2)
        th = np.random.choice(range(1, gene), 2, replace=True)
        v3 = v1[:th[0]] + v2[th[0]:]
        v1, v2 = v2, v1
        v4 = v1[:th[1]] + v2[th[1]:]
        new_population += [v1, v2, v3, v4]

    return np.array(new_population).reshape(num_of_process, -1, gene)

# creates a process according to the number of processes and starts an experiment
def process_run(process_num, population, shear_pram, return_dict):
    prev_results = shear_pram['prev_results']
    for net_param in population:
        if net_param.tolist().__str__() in prev_results:
            return_dict[net_param.tolist().__str__()] = prev_results[net_param.tolist().__str__()]
        else:
            train_lost, val_lost = shear_pram['runner'].run(net_param, epoch=shear_pram['epoch'])
            return_dict[net_param.tolist().__str__()] = [train_lost, val_lost]

# divides the tasks (parameter vectors) according to the number of processes
# ie. if we have 16 vectors and 4 processes each process will work on 4 vectors.
def spared_to_processes(population):
    return np.array(population).reshape(num_of_process, -1, gene)

# initializes a population with random numbers (each one from the correct value space)
def set_random_population(population, gene):
    for i in range(len(population)):
        for j in range(1, gene, 2):
            population[i][j] = np.random.randint(8, 33)

# creates the processes in the operating system.
# each process runs the process_run function
def start_processes(population, shear_pram):
    process_list = []
    return_dict = manager.dict()

    for i in range(num_of_process):
        p = Process(target=process_run, args=(i, population[i], shear_pram, return_dict))
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()
    return return_dict

# defines the parameters for the evolutionary algorithm
def get_toolbox():
    '''
    crossover: OnePoint
    mutate: Uniform
    select: Best
    :return:
    '''
    toolbox = base.Toolbox()
    # initializes a random population with value space of [1,8]
    toolbox.register("attr_bool", random.randint, 1, 8)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=gene)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # defines the chosen crossover operator
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutUniformInt, low=1, up=33, indpb=0.05)
    toolbox.register("select", tools.selBest)
    return toolbox

# our impementation for cross validation
def run_cross_val(population):
    task_list = spared_to_processes(population)
    return_dict = start_processes(task_list, shear_pram)
    with open('corss_val.json', 'w') as f:
        result = sorted(return_dict.items(), key=lambda x: x[1][1])
        ujson.dump(result, f, indent=4)

# runs the evolutionary algorithm
def run_EA(population, generation):
    for gen in range(generation):
        print('generation', gen)
        task_list = spared_to_processes(population)
        return_dict = start_processes(task_list, shear_pram)
        shear_pram['prev_results'] = return_dict

        with open('gen_'+str(gen)+'.json', 'w') as f:
            result = sorted(return_dict.items(), key=lambda x: x[1][0])
            ujson.dump(result, f, indent=4)

        for ind in population:
            ind.fitness.values = (return_dict[str(ind)][0],)

        best_population = toolbox.select(population, k=population_size // 2)
        offspring = algorithms.varAnd(best_population, toolbox, cxpb=0.5, mutpb=0.05)
        population = best_population + offspring
        population_unique = np.unique(population, axis=0)
        population = [population[population.index(p.tolist())] for p in population_unique]

        if population_size > len(population):
            print('add random', population_size - len(population))
            random_population = toolbox.population(n=(population_size - len(population)))
            set_random_population(random_population, gene)
            population += random_population
            print(population)

    top4 = tools.selBest(population, k=4)
    return top4

if __name__ == '__main__':
    t0 = time.time()
    multiprocessing.freeze_support()
    manager = multiprocessing.Manager()

    # the EA parameters
    shear_pram = manager.dict()
    shear_pram['runner'] = Experiment(batch_size=300)
    shear_pram['prev_results'] = []
    shear_pram['epoch'] = 5
    # number of generations
    generation = 20
    num_of_process = 4
    population_size = 16
    num_of_conv_layers = 3
    gene = num_of_conv_layers * 2
    AE = True

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = get_toolbox()
    if AE:
        population = toolbox.population(n=population_size)
        set_random_population(population, gene)
        run_EA(population, generation)
    else:
        population = toolbox.population(n=population_size*generation)
        set_random_population(population, population_size*generation, gene)
        run_cross_val(population)

