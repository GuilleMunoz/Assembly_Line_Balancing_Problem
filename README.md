# Assembly Line Balancing Problem
Genetic algorithm for the Assembly Line Balancing Problem.

## Description

The ALBP consists of assigning tasks to an ordered sequence of stations such that the precedence relations among the tasks are satisfied and some performance measure is optimized. There are some algorithms designed for this problem. The genetic algorithm approach is easy to program and can be easaly adapted for other problems such as the travelling salesman problem.
To find a better solution faster we defined an heuritic mutation operator. It assign a random station to a task that violates the precedence relations. 
We also include multiple selection methods (roulette, tournament, rank), mutation operators (random, swap, scramble, inverse), crossover operators (Double Point, Single Point, Uniform).
To compare the performance, we define comparations methods.

---

## Usage

How to run the algorithm

```python
from algorithm import engine, read_file

if __name__ == '__main__':
    k, num_op, graph, times = read_file('data75.txt')
    num_stations = 10

    engine(k, num_op, graph, times, num_stations=num_stations,
           pop_size=200, iterations=200,
           perc_elitism=10 / 200, perc_mat=0.5, sel_type='roulette', cross_type='SP',
           mutation_rate=0.20, mut_type='heur')
```

How to compare different selection methods

```python
from algorithm import compare_selection, read_file

# Comparaciones
if __name__ == '__main__':
    k, num_op, graph, times = read_file('data75.txt')
    stations = 10

    compare_selection(k, num_op, graph, times)
```

## Documentation

