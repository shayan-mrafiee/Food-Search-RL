from Q_learning import Q_Learning
from environment import Environment
import numpy as np

class AgentTrainer2:

    def __init__(self):

        self.actions = [
            np.array([0, 1]), # up 
            np.array([0, -1]), # down 
            np.array([1, 0]), # right 
            np.array([-1, 0]), # left 
        ]
        self.A = len(self.actions) # number of actions 

        self.states = {} # tuple (c_cur, c_prev, a_prev) as key, state number as value 
        for x in Environment.x_range:
            for y in Environment.y_range:
                loc = np.array([x, y])
                c_cur = Environment.static_concentration((x, y))
                if (c_cur, None, None) not in self.states:
                    self.states[(c_cur, None, None)] = len(self.states)

                for a, action in enumerate(self.actions):
                    loc_prev = loc - action 
                    if Environment.static_valid_loc(loc_prev):
                        c_prev = Environment.static_concentration(loc_prev)
                        if (c_cur, c_prev, a) not in self.states:
                            self.states[(c_cur, c_prev, a)] = len(self.states)



        self.S = len(self.states) # number of states

        self.alpha = 1.0 # learning rate
        self.gamma = 0.2 # discount factor 
        self.exploration_rate = 0.15

    def get_next_state(self, a : int, env : Environment):
        c_prev = env.get_concentration()
        c_cur = env.move(self.actions[a])
        if c_cur == -1: # invalid move: 
            return False
        
        return self.states[(c_cur, c_prev, a)]

    def get_init_state(self, env : Environment):
        c = env.get_concentration()
        return self.states[(c, None, None)]

    def reward(self, s, env : Environment):
        if env.found_target():
            return 1000
        
        c_cur, c_prev, _ = list(self.states)[s]
        
        # -10 if concentration decreased, otherwise -1
        return -10 if c_cur < c_prev else -1

    def create_agent(self):
        return Q_Learning(self.S, self.actions, self.reward, self.get_init_state, self.get_next_state, self.alpha, self.gamma, self.exploration_rate)
