import copy
from random import sample, random
from classes import Individual
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
from random import randint


mean = []
best = []

graph = {}
times = []
stations = 32
k = 18
operations = 0


def gen_indv(cross_type, mut_type):

    global operations, stations

    code = [randint(0, stations - 1) for i in range(operations)]
    return Individual(operations, stations, code, cross_type=cross_type, mut_type=mut_type)


def create_population(pop_size, cross_type, mut_type):
    """
    Creates the initial population.

    Args:
        pop_size (int): the size of the population
        cross_type (string): crossing technique
        mut_type (string): mutation technique

    Returns:
        (list(Individual)): initial population
    """
    pop = [gen_indv(cross_type, mut_type) for i in range(pop_size)]

    return pop


def rank_population(pop, gen):
    """
    Return the sorted population.

    Args:
        pop (list(Individual)): Population
        gen (int): current generation

    Returns:
        (list(Individual)): Sorted population
    """
    global graph, times, k
    return sorted(pop, key=lambda x:x.calc_fitness(gen, graph, times, k=k*10, scalling_factor=0) if x.fitness is 0 else x.fitness, reverse=False)


def select_by_tournament_(population):

    i = randint(0, len(population) - 1)
    j = randint(0, len(population) - 1)

    return population[i] if (population[i].fitness > population[j].fitness)else population[j]


def select_by_tournament(population, num=2):

    return [select_by_tournament_(population) for i in range(num)]


def select_by_rank(population, num=2):

    total = len(population)*(1 + len(population))/2
    rank = [i/total for i in range(1, len(population) + 1)]
    return np.random.choice(a=population, size=num, p=rank)


def select_by_roulette(population, num=2):
    total = reduce(lambda x, y: x + y.fitness, population, 0)
    roulette = [indiv.fitness / total for indiv in population]

    return np.random.choice(a=population, size=num, p=roulette)


def engine(size=100, iterations=100,
           perc_elitism=0.1, perc_mat=0.1, sel_type='roulette', cross_type='SP',
           mutation_rate=0.05, mut_type='random'
           ):

    population = rank_population(create_population(size, cross_type, mut_type), 0)

    select = eval('select_by_' + sel_type)

    for i in range(iterations):

        best.append(population[0].fitness)
        mean.append(reduce(lambda x, y: x + y.fitness, population, 0)/size)
        if population[0].gen < i - 50:
            break

        # Elitism
        new_generation = population[:int(perc_elitism*size)]

        # Selection
        the_chosen_ones = select(population[:int(size * perc_mat)], num=(size - int(perc_elitism*size)))

        mut = 2
        # Crossover
        for j in range(0, len(the_chosen_ones), 2):

            if j == len(the_chosen_ones) - 1:
                new_generation.append(Individual(operations, stations, copy.deepcopy(the_chosen_ones[j].code),
                                                 cross_type=cross_type, mut_type=mut_type))
                mut = 1
            else:
                new_generation.extend(the_chosen_ones[j].crossover(the_chosen_ones[j+1]))

            # Mutation
            for indv in new_generation[-mut:]:
                if random() < mutation_rate:
                    if mut_type == 'heur':
                        indv.mutate(graph)
                    else:
                        indv.mutate()

        #Evaluation
        population = rank_population(new_generation, i)

    if (population[0].calc_violations(graph)) > 0:
        print("SOLUCION NO VALIDA: ", population[0].calc_violations(graph))

    print(f"Parameters of the best solution : {[i+1 for i in population[0].code]}")
    print(f"Best solution reached after {population[0].gen} generations.")
    print(f"Fitness of the best solution : {population[0].fitness}")

    return population[0]


def read_file(file_name):
    """
    Read file to get times and graph

    Args:
        file_name(string): file name
    """
    global operations, graph, times, k, operations
    with open(file_name) as fd:
        operations = int(fd.readline()[:-1])
        for k in range(operations):
            times.append(int(fd.readline()))
            graph[k] = []
        while (True):
            line = fd.readline()
            if not line:
                break
            ij = line.split(',')
            graph[int(ij[0]) - 1].append(int(ij[1][:-1]) - 1)

    k = max(times)
    operations = len(times)


def compare_crossover():
    '''
    Compares crossover type
    '''
    global mean, best

    bests = []

    plt.figure(figsize=(6.8, 10))
    plt.title("Crossover")
    plt.xlabel('Generation', fontsize='large')
    plt.ylabel('Fitness', fontsize='large')

    engine(size=200, iterations=100,
           perc_elitism=1 / 200, perc_mat=0.8, sel_type='roulette', cross_type='SP',
           mutation_rate=0.50, mut_type='heur')

    plt.plot(mean, color='blue', linestyle=':')
    plt.plot(best, label='Single Point', color='blue')
    bests.append(best[-1])
    best = []
    mean = []

    engine(size=200, iterations=100,
           perc_elitism=1 / 200, perc_mat=0.8, sel_type='roulette', cross_type='DP',
           mutation_rate=0.50, mut_type='heur')

    plt.plot(mean, color='red', linestyle=':')
    plt.plot(best, label='Double Point', color='red')
    bests.append(best[-1])
    best = []
    mean = []

    engine(size=200, iterations=100,
           perc_elitism=1 / 200, perc_mat=0.8, sel_type='roulette', cross_type='UX',
           mutation_rate=0.50, mut_type='heur')

    plt.plot(mean, color='green', linestyle=':')
    plt.plot(best, label='Uniform', color='green')
    bests.append(best[-1])
    ticks = sorted(list(plt.yticks()[0]) + bests)

    i = 0
    while i < len(ticks):
        if ticks[i - 1] < ticks[i] <= ticks[i - 1] + 50:
            if not ticks[i-1] in bests:
                del ticks[i-1]
            else:
                del ticks[i]
        elif ticks[i] <= ticks[i - 1] <= ticks[i] + 50:
            if not ticks[i] in bests:
                del ticks[i]
            else:
                del ticks[i-1]
        else:
            i += 1

    plt.yticks(ticks)
    plt.legend(frameon=False, fontsize='large')
    plt.show()

    best = []
    mean = []


def compare_selection():
    '''
    Compares selection type
    '''
    global mean, best

    bests = []
    plt.figure(figsize=(6.8, 10))
    plt.title("Generation vs Fitness")
    plt.xlabel('Generation', fontsize='large')
    plt.ylabel('Fitness', fontsize='large')

    engine(size=200, iterations=100,
           perc_elitism=1 / 200, perc_mat=0.8, sel_type='roulette', cross_type='SP',
           mutation_rate=0.50, mut_type='heur')

    plt.plot(mean, color='blue', linestyle=':')
    plt.plot(best, label='Roulette', color='blue')
    bests.append(best[-1])
    best = []
    mean = []

    engine(size=200, iterations=100,
           perc_elitism=1 / 200, perc_mat=0.8, sel_type='rank', cross_type='SP',
           mutation_rate=0.50, mut_type='heur')

    plt.plot(mean, color='red', linestyle=':')
    plt.plot(best, label='Rank', color='red')
    bests.append(best[-1])
    best = []
    mean = []

    engine(size=200, iterations=100,
           perc_elitism=1 / 200, perc_mat=0.8, sel_type='tournament', cross_type='SP',
           mutation_rate=0.50, mut_type='heur')

    plt.plot(mean, color='green', linestyle=':')
    plt.plot(best, label='Tournament', color='green')
    bests.append(best[-1])
    ticks = sorted(list(plt.yticks()[0]) + bests)

    i = 0
    while i < len(ticks):
        if ticks[i - 1] < ticks[i] <= ticks[i - 1] + 50:
            if not ticks[i - 1] in bests:
                del ticks[i - 1]
            else:
                del ticks[i]
        elif ticks[i] <= ticks[i - 1] <= ticks[i] + 50:
            if not ticks[i] in bests:
                del ticks[i]
            else:
                del ticks[i - 1]
        else:
            i += 1

    plt.yticks(ticks)

    plt.legend(frameon=False, fontsize='large')
    plt.show()

    best = []
    mean = []


def compare_mutation():
    '''
    Compares mutation type
    '''
    global mean, best

    bests = []

    plt.figure(figsize=(6.8, 10))
    plt.title("Mutations types")
    plt.xlabel('Generation', fontsize='large')
    plt.ylabel('Fitness', fontsize='large')

    engine(size=200, iterations=200,
           perc_elitism=1 / 200, perc_mat=1, sel_type='roulette', cross_type='SP',
           mutation_rate=0.50, mut_type='heur')

    plt.plot(mean, color='blue', linestyle=':')
    plt.plot(best, label='Heur', color='blue')
    bests.append(best[-1])
    best = []
    mean = []

    engine(size=200, iterations=200,
           perc_elitism=1 / 200, perc_mat=1, sel_type='roulette', cross_type='SP',
           mutation_rate=0.50, mut_type='random')

    plt.plot(mean, color='red', linestyle=':')
    plt.plot(best, label='Random', color='red')
    bests.append(best[-1])
    best = []
    mean = []

    engine(size=200, iterations=200,
           perc_elitism=1 / 200, perc_mat=1, sel_type='roulette', cross_type='SP',
           mutation_rate=0.50, mut_type='swap')

    plt.plot(mean, color='green', linestyle=':')
    plt.plot(best, label='Swap', color='green')
    bests.append(best[-1])
    best = []
    mean = []

    engine(size=200, iterations=200,
           perc_elitism=1 / 200, perc_mat=1, sel_type='roulette', cross_type='SP',
           mutation_rate=0.50, mut_type='inversion')

    plt.plot(mean, color='orange', linestyle=':')
    plt.plot(best, label='Inversion', color='orange')
    bests.append(best[-1])
    best = []
    mean = []

    engine(size=200, iterations=200,
           perc_elitism=1 / 200, perc_mat=1, sel_type='roulette', cross_type='SP',
           mutation_rate=0.50, mut_type='gaussian')

    plt.plot(mean, color='black', linestyle=':')
    plt.plot(best, label='Gaussian', color='black')
    bests.append(best[-1])
    best = []
    mean = []

    engine(size=200, iterations=200,
           perc_elitism=1 / 200, perc_mat=1, sel_type='roulette', cross_type='SP',
           mutation_rate=0.50, mut_type='scramble')

    plt.plot(mean, color='purple', linestyle=':')
    plt.plot(best, label='Scramble', color='purple')
    bests.append(best[-1])
    ticks = sorted(list(plt.yticks()[0]) + bests)

    i = 0
    while i < len(ticks):
        if ticks[i - 1] < ticks[i] <= ticks[i - 1] + 50:
            if not ticks[i - 1] in bests:
                del ticks[i - 1]
            else:
                del ticks[i]
        elif ticks[i] <= ticks[i - 1] <= ticks[i] + 50:
            if not ticks[i] in bests:
                del ticks[i]
            else:
                del ticks[i - 1]
        else:
            i += 1

    plt.yticks(ticks)

    plt.legend(frameon=False, fontsize='large')
    plt.show()
    best = []
    mean = []


if __name__ == '__main__':
    read_file('data32.txt')
    stations = 5
    plt.figure(figsize=(6.8, 10))
    plt.title("Generation vs Fitness")
    plt.xlabel('Generation', fontsize='large')
    plt.ylabel('Fitness', fontsize='large')

    while (True):

        best = []
        mean = []

        my_best = engine(size=200, iterations=200,
                  perc_elitism=10 / 200, perc_mat=0.5, sel_type='roulette', cross_type='SP',
                  mutation_rate=0.20, mut_type='heur')

        print('aqui sigo')
        if my_best.fitness < 175:
            break


    plt.plot(mean, color='blue', linestyle=':')
    plt.plot(best, color='blue')

    plt.yscale('log')
    #plt.yticks(list(set([i if i > 0 else 0 for i in list(plt.yticks()[0]) + [my_best]])))
    plt.yticks([my_best.fitness, 1000, 10000], [str(my_best.fitness), r'$10^3$', r'$10^4$'])

    plt.show()
    print(reduce(lambda x, y: x + y, times, 1)/stations)

