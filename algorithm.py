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
k = 0
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
    return sorted(pop, key=lambda x:x.calc_fitness(gen, graph, times, k=k*10) if x.fitness is 0 else x.fitness, reverse=False)


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


def engine(size=100, iterations=200,
           perc_elitism=0.1, perc_mat=0.1, sel_type='roulette', cross_type='SP',
           mutation_rate=0.05, mut_type='random'
           ):

    population = rank_population(create_population(size, cross_type, mut_type), 0)

    select = eval('select_by_' + sel_type)

    for i in range(iterations):

        best.append(population[0].fitness)
        mean.append(reduce(lambda x, y: x + y.fitness, population, 0)/size)

        #Elitism
        new_generation = population[:int(perc_elitism*size)]

        #Selection and crossover
        for _ in range(int((size - int(perc_elitism*size))/2)):
            # the_chosen_ones = [select_by_tournament(population), select_by_tournament(population)]
            # the_chosen_ones = select_by_rank(population[:int(size*perc_mat)])
            the_chosen_ones = select(population[:int(size * perc_mat)])
            new_generation.extend(the_chosen_ones[0].crossover(the_chosen_ones[1]))

            # Mutation
            for indv in new_generation[-2:]:
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


if __name__ == '__main__':

    read_file('data32.txt')
    stations = 3

    engine(size=200, iterations=200,
           perc_elitism=1/200, perc_mat=1, sel_type='roulette', cross_type='UX',
           mutation_rate=0.50, mut_type='heur')

    best_heur = best
    mean_heur = mean
    best = []
    mean = []

    engine(size=200, iterations=200,
           perc_elitism=1 / 200, perc_mat=1, sel_type='roulette', cross_type='UX',
           mutation_rate=0.50, mut_type='random')

    best_random = best
    mean_random = mean
    best = []
    mean = []

    engine(size=200, iterations=200,
           perc_elitism=1 / 200, perc_mat=1, sel_type='roulette', cross_type='UX',
           mutation_rate=0.50, mut_type='swap')

    best_swap = best
    mean_swap = mean
    best = []
    mean = []

    engine(size=200, iterations=200,
           perc_elitism=1 / 200, perc_mat=1, sel_type='roulette', cross_type='UX',
           mutation_rate=0.50, mut_type='inversion')

    best_inverse = best
    mean_inverse = mean
    best = []
    mean = []

    engine(size=200, iterations=200,
           perc_elitism=1 / 200, perc_mat=1, sel_type='roulette', cross_type='UX',
           mutation_rate=0.50, mut_type='gaussian')

    best_gaussian = best
    mean_gaussian = mean
    best = []
    mean = []

    engine(size=200, iterations=200,
           perc_elitism=1 / 200, perc_mat=1, sel_type='roulette', cross_type='UX',
           mutation_rate=0.50, mut_type='scramble')

    plt.title("Generation vs Fitness")
    plt.xlabel('Generation')
    plt.ylabel('Fitness')

    plt.plot(mean_heur, color='blue', linestyle=':')
    plt.plot(best_heur, label='Heur', color='blue')

    plt.plot(mean_random, color='red', linestyle=':')
    plt.plot(best_random, label='Random', color='red')

    plt.plot(mean_swap, color='green', linestyle=':')
    plt.plot(best_swap, label='Swap', color='green')

    plt.plot(mean_inverse, color='orange', linestyle=':')
    plt.plot(best_inverse, label='Inversion', color='orange')

    plt.plot(mean_gaussian, color='black', linestyle=':')
    plt.plot(best_gaussian, label='Gaussian', color='black')

    plt.plot(mean, color='purple', linestyle=':')
    plt.plot(best, label='Scramble', color='purple')

    plt.legend(frameon=False)
    plt.show()


