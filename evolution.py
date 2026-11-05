import numpy as np 
from src.Q_learning import Q_Learning
from src.environment import Environment
from src.version_3 import AgentTrainer3
import random 

# mutation rates for alpha, gamma, epsilon, T, t, c, d
mutation_rates = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]

# bounds for parameters. both ends inclusive 
params_bounds = [(0, 1000), (0, 999), (0, 990), (-100000, 100000), (-100000, 100000), (-100000, 100000), (-100000, 100000)]

# conversion factors for each parameter. when a random is selected within the bounds, it is multiplied by this number to get the actual value 
conv_facts = [0.001, 0.001, 0.001, 0.01, 0.01, 0.01, 0.01]

def reproduce(params):
    new_params = [None] * len(params)
    for i in range(len(params)):
        if random.random() < mutation_rates[i]:
            new_params[i] = random.randint(params_bounds[i][0], params_bounds[i][1]) * conv_facts[i]
        else:
            new_params[i] = params[i]

    return new_params

def output_results(l):
    f = open("data/best_params.csv", "w")
    f.write("alpha,gamma,epsilon,T,t,c,d,excess steps\n")
    for excess_steps, params in l:
        alpha, gamma, eps, T, t, c, d = params 
        f.write(f"{alpha},{gamma},{eps},{T},{t},{c},{d},{excess_steps}\n")

episodes = 100
params_list = [
    [1.0, 0.2, 0.15, 1000, -1, 0, -9],
    [1.0, 0.2, 0.15, 1000, -1, 0, -9],
    [1.0, 0.2, 0.15, 1000, -1, 0, -9],
    [1.0, 0.2, 0.15, 1000, -1, 0, -9],
    [1.0, 0.2, 0.15, 1000, -1, 0, -9],
    [1.0, 0.2, 0.15, 1000, -1, 0, -9],
    [1.0, 0.2, 0.15, 1000, -1, 0, -9],
    [1.0, 0.2, 0.15, 1000, -1, 0, -9],
    [1.0, 0.2, 0.15, 1000, -1, 0, -9],
    [1.0, 0.2, 0.15, 1000, -1, 0, -9],

    [0.5, 0.5, 0.5, 10, -1, 0, 0],
    [0.5, 0.5, 0.5, 10, -1, 0, 0],
    [0.5, 0.5, 0.5, 10, -1, 0, 0],
    [0.5, 0.5, 0.5, 10, -1, 0, 0],
    [0.5, 0.5, 0.5, 10, -1, 0, 0],
    [0.5, 0.5, 0.5, 10, -1, 0, 0],
    [0.5, 0.5, 0.5, 10, -1, 0, 0],
    [0.5, 0.5, 0.5, 10, -1, 0, 0],
    [0.5, 0.5, 0.5, 10, -1, 0, 0],
    [0.5, 0.5, 0.5, 10, -1, 0, 0]
]

limit = 5
offsprings_per = 3

print("\rProgress: 0%", end='')
at3 = AgentTrainer3()

for k in range(episodes):
    excess_steps = [None] * len(params_list)

    for i, params in enumerate(params_list):
        agent = at3.set_parameters(params).create_agent()
        agent.learn(100)
        init_loc, steps, found = agent.search_source(Environment(), learn=True, explore=False, return_steps_list=False)
        excess_steps[i] = np.inf if not found else steps - np.sum(np.abs(init_loc))

    new_params_list = sorted([(excess_steps[i], params_list[i]) for i in range(len(params_list))])[:5]
    if k == episodes-1:
        output_results(new_params_list)
        break

    new_params_list = [l[1] for l in new_params_list]


    for i in range(limit):
        for j in range(offsprings_per):
            new_params_list.append(reproduce(new_params_list[i]))

    params_list = new_params_list

    if (k+1) * 100 % episodes == 0:
        print(f"\rProgress: {(k+1) * 100 // episodes}%", end='')

print("\rProgress: 100%")