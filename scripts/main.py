from src.Q_learning import Q_Learning
from src.version_1 import get_agent
from src.version_2 import AgentTrainer2
from src.version_3 import AgentTrainer3
from src.environment import Environment
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

at3 = AgentTrainer3()

def get_excess_avg(agent):
    agent.learn(100)

    N = 100
    dif_sum = 0
    failed_attempts = 0

    locs_sample = sample(at3.all_locs, counts=[N] * len(at3.all_locs), k=len(at3.all_locs)*N)

    for i, (x, y) in enumerate(locs_sample):
        init_loc, steps, found = agent.search_source(Environment(x, y), learn=True, explore=False, MAX_STEPS=100)

        if not found:
            failed_attempts += 1

        dif_sum += steps - (np.abs(x) + np.abs(y))

    return (dif_sum / len(locs_sample), failed_attempts / len(locs_sample))

"""
Auto tune 1 results:
Success: True
Optimal parameters: [ 9.57847753e-01  2.03602251e-01  1.48061516e-01  1.02635859e+03
 -1.01601256e+00  1.84233436e-04 -8.77201507e+00]
Min Value of Function: 2.217037037037037
Message: Optimization terminated successfully.
Iterations: 183
Function evaluations: 533
Initial objective: 2.3851851851851853
Final objective: 2.217037037037037
Total duration: 242.87s

Auto tune 2 results:
Success: True
Optimal parameters: [ 9.62699775e-01  2.04958511e-01  1.49322619e-01  1.04428666e+03
 -9.90434907e-01  1.85460731e-04 -8.83047074e+00]
Min Value of Function: 2.189753086419753
Message: Optimization terminated successfully.
Iterations: 192
Function evaluations: 575
Initial objective: 2.289506172839506
Final objective: 2.189753086419753
Total duration: 268.01s
"""

N = 100
# f = open("data/version_5.csv", "w")
print("\rProgress: 0%", end='')
for i in range(N):
    agent = at3.set_parameters([0.165,0.491,0.864,970.82,-836.5,390.94,-620.67]).create_agent()
    steps, failure = get_excess_avg(agent)
    # f.write(f"{steps},{failure}\n")

    if (i+1) * 100 % N == 0:
        print(f"\rProgress: {(i+1) * 100 // N}%", end='')
    
print("\rProgress: 100%")

# f.close()