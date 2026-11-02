from src.Q_learning import Q_Learning
from version_2 import AgentTrainer2
from version_3 import AgentTrainer3
from environment import Environment
import turtle
from turtle import Turtle
import time
import numpy as np

# locs = list of 
def visualize(agents_steps, colors, scale=75):
    N = len(colors) # number of agents 

    for i in range(N):
        print(f"Agent {i+1}:")
        print(f"Initial location: {agents_steps[i][0]}")
        print(f"Reached source: {np.array_equal(np.array([0, 0]), agents_steps[i][-1])}")
        print(f"Steps: {len(agents_steps[i])}")
        print()


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

# at3.auto_tune()
"""
Auto Tune results: 
Success: False
Optimal parameters: [ 9.5e-01  2.0e-01  1.5e-01  1.0e+03 -1.0e+00  0.0e+00 -9.0e+00]
Min Value of Function: 160.70000000000005
"""
a3 = at3.create_agent()
a2 = at2.create_agent()

a3.learn(100)
a2.learn(100)

x = np.random.choice(Environment.x_range)
y = np.random.choice(Environment.y_range)
visualize([a3.search_source(Environment(x, y), learn=False, explore=False), a2.search_source(Environment(x, y), learn=False, explore=False)], colors=["blue", "red"])
