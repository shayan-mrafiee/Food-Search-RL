import numpy as np 
from environment import Environment 
import turtle
from turtle import Turtle
import time

class Q_Learning:

    def __init__(self, S, actions, R, get_init_state, get_next_state, alpha, gamma, epsilon):
        self.S = S # number of states 
        self.A = len(actions)
        self.actions = actions 
        self.R = R # reward function 
        self.get_init_state = get_init_state
        self.get_next_state = get_next_state
        self.alpha = alpha # learning rate 
        self.gamma = gamma # discount factor 
        self.epsilon = epsilon # exploration rate 
        self.Q_table = np.zeros((S, self.A))


    # Q learning: 
    def learn(self, episodes = 100000, env = None):
        MAX_STEPS = 10000
        use_diff_env = (env is None)

        for k in range(episodes):
            # print(f"Episode {k+1}:")
            if use_diff_env:
                env = Environment()
            else:
                env.reset()
            
            s = self.get_init_state(env)
            # print(f"Initial location: {env.loc}")
            steps = 0 
            while not env.found_target() and steps < MAX_STEPS:
                if np.random.random() <= self.epsilon:
                    a = np.random.randint(0, self.A)
                else:
                    a = np.argmax(self.Q_table[s])

                next_s = self.get_next_state(a, env)
                while next_s is False: # invalid move
                    a = np.random.randint(0, self.A)
                    next_s = self.get_next_state(a, env)

                done = env.found_target()
                
                r = self.R(next_s, env)

                target = r if done else r + self.gamma * np.max(self.Q_table[next_s])

                self.Q_table[s, a] += self.alpha * (target - self.Q_table[s, a])

                s = next_s
                steps += 1

            # print(f"Steps: {env.steps}")
            # print(f"Reached source: {env.found_target()}\n")

    def search_source(self, env, learn=True, explore=True, visualize=False):
        if visualize:
            t = Turtle("turtle")
            t.penup()
            t.color("blue")
            t.speed(0)

            food = Turtle("circle")
            food.color("green")

            time.sleep(5)

        MAX_STEPS = 10000
        s = self.get_init_state(env)

        print(f"Initial location: {env.loc}")
        steps = 0
        while not env.found_target() and steps < MAX_STEPS:
            if visualize:
                t.goto(env.loc * 75)
                t.speed(1)
                time.sleep(1)

            if explore and np.random.random() <= self.epsilon:
                a = np.random.randint(0, self.A)
            else:
                a = np.argmax(self.Q_table[s])

            next_s = self.get_next_state(a, env)
            while next_s is False: # invalid move
                a = np.random.randint(0, self.A)
                next_s = self.get_next_state(a, env)

            if learn:
                done = env.found_target()
                r = self.R(next_s, env)
                target = r if done else r + self.gamma * np.max(self.Q_table[next_s])
                self.Q_table[s, a] += self.alpha * (target - self.Q_table[s, a])

            s = next_s
            steps += 1
        
        if visualize:
            t.goto(env.loc * 75)
            turtle.exitonclick()
        print(f"Steps: {env.steps}")
        print(f"Reached source: {env.found_target()}\n")
