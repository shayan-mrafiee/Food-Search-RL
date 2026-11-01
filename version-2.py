from src.Q_learning import Q_Learning
from environment import Environment
import numpy as np

actions = [
    np.array([0, 1]), # up 
    np.array([0, -1]), # down 
    np.array([1, 0]), # right 
    np.array([-1, 0]), # left 
]
A = len(actions) # number of actions 

states = {} # tuple (c_cur, c_prev, a_prev) as key, state number as value 
for x in Environment.x_range:
    for y in Environment.y_range:
        loc = np.array([x, y])
        c_cur = Environment.static_concentration((x, y))
        if (c_cur, None, None) not in states:
            states[(c_cur, None, None)] = len(states)

        for a, action in enumerate(actions):
            loc_prev = loc - action 
            if Environment.static_valid_loc(loc_prev):
                c_prev = Environment.static_concentration(loc_prev)
                if (c_cur, c_prev, a) not in states:
                    states[(c_cur, c_prev, a)] = len(states)



S = len(states) # number of states

alpha = 1.0 # learning rate
gamma = 0.2 # discount factor 
exploration_rate = 0.15

def get_next_state(a : int, env : Environment):
    c_prev = env.get_concentration()
    c_cur = env.move(actions[a])
    if c_cur == -1: # invalid move: 
        return False
    
    return states[(c_cur, c_prev, a)]

def get_init_state(env : Environment):
    c = env.get_concentration()
    return states[(c, None, None)]

def reward(s, env : Environment):
    if env.found_target():
        return 1000
    
    c_cur, c_prev, _ = list(states)[s]
    
    # -10 if concentration decreased, otherwise -1
    return -10 if c_cur < c_prev else -1

agent = Q_Learning(S, actions, reward, get_init_state, get_next_state, alpha, gamma, exploration_rate)
env = Environment(1, -3)
# agent.learn(7, Environment(-4, -4))
# agent.learn(7, Environment(-4, 4))
# agent.learn(7, Environment(4, -4))
# agent.learn(7, Environment(4, 4))
agent.learn(100)
env.reset()
agent.search_source(env, learn=True, explore=False, visualize=True)
