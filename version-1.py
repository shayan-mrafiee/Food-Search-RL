from src.Q_learning import Q_Learning
from environment import Environment
import numpy as np

S = 9 # number of states 
A = 4 # number of actions 

actions = [
    np.array([0, 1]), # up 
    np.array([0, -1]), # down 
    np.array([1, 0]), # right 
    np.array([-1, 0]), # left 
]

alpha = 1.0 # learning rate
gamma = 0.2 # discount factor 
exploration_rate = 0.15

def get_next_state(a, env : Environment):
    c_cur = env.get_concentration()
    c_next = env.move(actions[a])
    if c_next == -1: # invalid move: 
        return False
    
    return (c_next > c_cur) * A + a

def reward(s, env : Environment):
    if env.found_target():
        return 1000
    # -1 if concentration increased, otherwise -2
    return -10 if s // A == 0 else -1

agent = Q_Learning(S, actions, reward, get_next_state, alpha, gamma, exploration_rate)
env = Environment()
agent.learn(1000)
env.reset()
agent.search_source(env, learn=True, explore=False)
