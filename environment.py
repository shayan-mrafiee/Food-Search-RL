import numpy as np 

class Environment: 

    x_range = range(-4, 5)
    y_range = range(-4, 5)
    food_source_loc = np.array([0, 0])

    def __init__(self, x = None, y = None):
        x = np.random.choice(Environment.x_range) if x is None else x
        y = np.random.choice(Environment.y_range) if y is None else y
        self.loc = np.array([x, y])
        self.init_loc = self.loc.copy()
        self.steps = 0 


    def valid_loc(self):
        x, y = self.loc
        return x in Environment.x_range and y in Environment.y_range
    
    def static_valid_loc(loc):
        x, y = loc
        return x in Environment.x_range and y in Environment.y_range

    def move(self, a):
        self.loc += a 
        if not self.valid_loc():
            self.loc -= a
            return -1
        
        self.steps += 1
        return self.get_concentration()
    
    def get_concentration(self, loc = None):
        if loc is None:
            loc = self.loc

        x, y = loc

        r = np.sqrt(x**2 + y**2) 
        return 1/(1 + r)
    
    def static_concentration(loc):
        x, y = loc
        r = np.sqrt(x**2 + y**2)
        return 1/(1 + r)
    

    def found_target(self):
        return np.array_equal(self.loc, Environment.food_source_loc)
    
    def reset(self):
        self.loc = self.init_loc.copy()
        self.steps = 0