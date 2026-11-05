from src.Q_learning import Q_Learning
from src.environment import Environment
from scipy.optimize import minimize
import numpy as np
from random import sample

class AgentTrainer3:

    def __init__(self):

        self.actions = [
            np.array([0, 1]), # up 
            np.array([0, -1]), # down 
            np.array([1, 0]), # right 
            np.array([-1, 0]), # left 
        ]
        self.A = len(self.actions) # number of actions 

        self.all_locs = []

        self.states = {} # tuple (c_cur, c_prev, a_prev) as key, state number as value 
        for x in Environment.x_range:
            for y in Environment.y_range:
                loc = np.array([x, y])
                self.all_locs.append(loc)

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
        self.states_list = list(self.states)

        self.alpha = 1.0 # learning rate
        self.gamma = 0.2 # discount factor 
        self.epsilon = 0.15
        self.T = 1000
        self.t = -1 # time penalty 
        self.c = 0 # concentration increase reward 
        self.d = -9 # concentration decrease reward 


    def set_parameters(self, params):
        self.alpha, self.gamma, self.epsilon, self.T, self.t, self.c, self.d = params
        return self


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
            return self.T
        
        c_cur, c_prev, _ = self.states_list[s]
        
        return self.t + (self.c if c_cur > c_prev else self.d)
    
    
    def get_excess_avg(self, params):
        self.set_parameters(params)

        agent = self.create_agent()
        agent.learn(100)

        N = 100
        dif_sum = 0

        locs_sample = sample(self.all_locs, counts=[N] * len(self.all_locs), k=len(self.all_locs)*N)

        for x, y in locs_sample:
            agent_steps = agent.search_source(Environment(x, y), learn=True, explore=False)
            steps = len(agent_steps)-1

            if not np.array_equal(Environment.food_source_loc, agent_steps[-1]):
                steps += 1000
                

            dif_sum += steps - (np.abs(x) + np.abs(y))

        return dif_sum / len(locs_sample)
    
    
    def auto_tune(self, x0 = None):
        if x0 is None:
            x0 = [
                1.0, # alpha 
                0.2, # gamma 
                0.15, # epsilon 
                1000, # T
                -1, # t
                0, # c
                -9 # d
            ]

        bounds = [
            (0.0, 1.0), # alpha
            (0.0, 1.0), # gamma 
            (0.0, 1.0), # epsilon 
            (None, None), # T
            (None, None), # t
            (None, None), # c
            (None, None) # d
        ]

        result = minimize(
            self.get_excess_avg, 
            x0=x0, 
            bounds=bounds, 
            method='Nelder-Mead',
            options={
                "adaptive" : True,
                "xatol" : 0.01,
                "fatol" : 0.1
            }
        )
        self.alpha, self.gamma, self.epsilon, self.T, self.t, self.c, self.d = result.x

        print("Success:", result.success)
        print(f"Optimal parameters: {result.x}")
        print(f"Min Value of Function: {result.fun}")
        print("Message:", result.message)
        print("Iterations:", result.nit)
        print("Function evaluations:", result.nfev)
        print("Initial objective:", self.get_excess_avg(x0))
        print("Final objective:", result.fun)


    def create_agent(self):
        return Q_Learning(self.S, self.actions, self.reward, self.get_init_state, self.get_next_state, self.alpha, self.gamma, self.epsilon)
