from Q_learning import Q_Learning
from version_1 import get_agent
from version_2 import AgentTrainer2
from version_3 import AgentTrainer3
from environment import Environment
import turtle
from turtle import Turtle
import time
import numpy as np
from random import sample

def print_results(agents_steps):
    for i, steps in enumerate(agents_steps):
        print(f"Agent {i+1}:")
        print(f"Initial location: {steps[0]}")
        print(f"Reached source: {np.array_equal(np.array([0, 0]), steps[-1])}")
        print(f"Steps: {len(steps)-1}")
        print()

    
def visualize(agents_steps, colors, scale=75):
    N = len(colors) # number of agents 

    food_source = Turtle("circle")
    food_source.color("green")

    turtles = [None] * N
    for i in range(N):
        turtles[i] = Turtle("turtle")
        t = turtles[i]
        t.color(colors[i])
        t.speed(0)
        t.penup()
        t.goto(scale * agents_steps[i][0])
        t.speed(1)

    indices = [1] * N

    cont = True
    time.sleep(4) 
    while cont:
        time.sleep(1)
        cont = False
        for i in range(N):
            if indices[i] < len(agents_steps[i]):
                cont = True
                turtles[i].goto(scale * agents_steps[i][indices[i]])
                indices[i] += 1

    turtle.exitonclick()

at2 = AgentTrainer2()
at3 = AgentTrainer3()

def get_excess_avg(agent):
    agent.learn(100)

    N = 100
    dif_sum = 0
    failed_attempts = 0

    locs_sample = sample(at3.all_locs, counts=[N] * len(at3.all_locs), k=len(at3.all_locs)*N)

    for i, (x, y) in enumerate(locs_sample):
        agent_steps = agent.search_source(Environment(x, y), learn=True, explore=False, MAX_STEPS=100)
        steps = len(agent_steps)-1

        if not np.array_equal(Environment.food_source_loc, agent_steps[-1]):
            failed_attempts += 1

        dif_sum += steps - (np.abs(x) + np.abs(y))

    return (dif_sum / len(locs_sample), failed_attempts / len(locs_sample))


at3.set_parameters([9.77751848e-01, 1.90289042e-01, 1.50162023e-01, 1.02678976e+03, -1.02492501e+00, -9.53983561e-05, -8.73070488e+00])

N = 100
f1 = open("data/version_1.csv", "w")
# f2 = open("data/version_2.csv", "w")
# f3 = open("data/version_3.csv", "w")

print("Progress: 0%", end='')
for i in range(N):
    a1 = get_agent()
    # a2 = at2.create_agent()
    # a3 = at3.create_agent()
    excess, fail = get_excess_avg(a1)
    f1.write(f"{excess}, {fail}\n")
    # f2.write(f"{get_excess_avg(a2)}\n")
    # f3.write(f"{get_excess_avg(a3)}\n")

    print(f"\rProgress: {i+1}%", end='')

f1.close()
# f2.close()
# f3.close()

# at3.auto_tune([0.96, 0.19, 0.15, 1001.29, -1.01, -9.43e-05, -8.97])
# print()
"""
Auto Tune results: 
Success: True
Optimal parameters: [ 9.77751848e-01  1.90289042e-01  1.50162023e-01  1.02678976e+03
 -1.02492501e+00 -9.53983561e-05 -8.73070488e+00]
Min Value of Function: 2.2012345679012344
Message: Optimization terminated successfully.
Iterations: 239
Function evaluations: 666
Initial objective: 2.3519753086419755
Final objective: 2.2012345679012344
"""


# print("Version 3:")
# print(at3.get_excess_avg([9.77751848e-01, 1.90289042e-01, 1.50162023e-01, 1.02678976e+03, -1.02492501e+00, -9.53983561e-05, -8.73070488e+00]))
# print("Version 2:")
# print(at3.get_excess_avg([1, 0.2, 0.15, 1000, -1, 0, -9]))
# print()

# a3.learn(100)
# a2.learn(100)

# x = np.random.choice(Environment.x_range)
# y = np.random.choice(Environment.y_range)

# agents_steps = [a3.search_source(Environment(x, y), learn=True, explore=False), a2.search_source(Environment(x, y), learn=True, explore=False)]
# print_results(agents_steps)
# visualize(agents_steps, colors=["blue", "red"])