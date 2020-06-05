import copy
from math import exp
from random import randint, random
from numpy.random import permutation


class Individual:
    """
    Each individual is coded as a list of integers. The nth-element of the list corresponds to station asigned
    to the operation n.

    The fitness of a route is given by the sum of the time of the slowest station and the number of precedence
    violations multiplied by a constant k (equal to the slowest operation).

    SP -> Single point crossover
    DP -> Double point crossover
    UX -> Uniform crossover

    random -> Random mutation (gives a random value to a random element)
    heur -> Heuristic mutation
    swap -> Swap mutation (select 2 elemtents and swaps them)
    scramble -> scramble subset
    inverse -> Inverse subset

    """
    _fitness = 0
    code = list()
    operations = 0
    stations = 0
    gen = 0

    def __init__(self, operations, stations, code, cross_type='SP', mut_type='random'):
        self.operations = operations
        self.stations = stations
        self.code = code
        self.crossover = eval('self.crossover_' + cross_type)
        self.mutate = eval('self.mutate_' + mut_type)

    def __repr__(self):
        return str(self.code)

    @property
    def fitness(self):
        return self._fitness

    @fitness.setter
    def fitness(self, fitness):
        self._fitness = fitness

    def calc_violations(self, graph):
        """
        Calculates the number of precedence violations

        Args:
            graph(dict): precedence graph of the problem
        Return:
            (int)
        """
        violations = 0
        for op in range(self.operations):
            for neighbor in graph[op]:
                if self.code[neighbor] < self.code[op]:
                    violations += 1
        return violations

    def calc_fitness(self, gen, graph, times, k=1, scalling_factor=0):
        """
        Calculates fitness of the code.

        Args:
            graph(dict): precedence graph of the problem
            k(float): time of slowest operations
        Returns:
            (float)
        """
        time_op = [0] * self.stations
        for op in range(self.operations):
            self.code[op] = int(self.code[op]) % self.stations
            time_op[int(self.code[op])] += times[op]

        # Scalling factor...
        # self.fitness = -exp(scalling_factor * (max(time_op) + k * calc_violations(indv)))
        # No scalling factor
        # self.fitness = max(time_op) + (k * self.calc_violations(graph)) if (scalling_factor is 0) else
        # exp(-scalling_factor * (max(time_op) + k * self.calc_violations(graph)))
        self.fitness = max(time_op) + (k * self.calc_violations(graph))
        self.gen = gen

        return self.fitness

    def mutate_random(self):
        self.code[randint(0, self.operations - 1)] = randint(0, self.stations - 1)
        self.fitness = 0

    def mutate_heur(self, graph):

        has_changed = False
        for op in range(self.operations):
            for neighbor in graph[op]:
                if self.code[neighbor] < self.code[op]:
                    self.code[op] = randint(0, self.stations - 1)
                    has_changed = True

        if not has_changed:
            self.mutate_random()
        else:
            self.fitness = 0

    def mutate_swap(self):

        i = randint(0, self.operations - 1)
        j = randint(0, self.operations - 1)

        self.code[i], self.code[j] = self.code[j], self.code[i]
        self.fitness = 0

    def mutate_scramble(self):

        i = randint(0, self.operations)
        j = randint(0, self.operations)

        if i > j:
            i, j = j, i

        self.code[i:j] = copy.deepcopy(permutation(self.code[i:j]))
        self.fitness = 0

    def mutate_inversion(self):

        i = randint(0, self.operations)
        j = randint(0, self.operations)

        if i > j:
            i, j = j, i

        self.code[i:j].reverse()
        self.fitness = 0


    def mutate_gaussian(self):

        i = randint(0, self.operations - 1)
        self.code[i] += randint(0, self.stations - 1 - self.code[i])
        self.fitness = 0


    def crossover_SP(self, indv):

        p = randint(0, self.operations)

        code_c1 = self.code[:p] + indv.code[p:]
        code_c2 = indv.code[:p] + self.code[p:]

        ch1 = Individual(self.operations, self.stations, code=copy.deepcopy(code_c1),
                         cross_type=self.crossover.__name__[-2:],
                         mut_type=self.mutate.__name__.split('_')[-1])

        ch2 = Individual(self.operations, self.stations, code=copy.deepcopy(code_c2),
                         cross_type=self.crossover.__name__[-2:],
                         mut_type=self.mutate.__name__.split('_')[-1])
        return ch1, ch2


    def crossover_DP(self, indv):

        i = randint(0, self.operations)
        j = randint(0, self.operations)
        if i > j:
            j, i = i, j

        code_c1 = copy.deepcopy(self.code[:i] + indv.code[i:j] + self.code[j:])
        code_c2 = copy.deepcopy(indv.code[:i] + self.code[i:j] + indv.code[j:])

        ch1 = Individual(self.operations, self.stations, code=code_c1,
                         cross_type=self.crossover.__name__[-2:],
                         mut_type=self.mutate.__name__.split('_')[-1])

        ch2 = Individual(self.operations, self.stations, code=code_c2,
                         cross_type=self.crossover.__name__[-2:],
                         mut_type=self.mutate.__name__.split('_')[-1])
        return ch1, ch2


    def crossover_UX(self, indv):

        code_c1 = [0] * self.operations
        code_c2 = [0] * self.operations

        for i in range(self.operations):
            if random() < 0.5:
                code_c1[i] = self.code[i]
                code_c2[i] = indv.code[i]
            else:
                code_c2[i] = self.code[i]
                code_c1[i] = indv.code[i]

        ch1 = Individual(self.operations, self.stations, code=copy.deepcopy(code_c1),
                         cross_type=self.crossover.__name__[-2:],
                         mut_type=self.mutate.__name__.split('_')[-1])

        ch2 = Individual(self.operations, self.stations, code=copy.deepcopy(code_c2),
                         cross_type=self.crossover.__name__[-2:],
                         mut_type=self.mutate.__name__.split('_')[-1])

        return ch1, ch2
