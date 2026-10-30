import numpy as np 

class Environment: 

    x_range = (-4, 4)
    y_range = (-4, 4)
    food_source_loc = np.array([0, 0])

    def __init__(self, x = None, y = None):
        x = np.random.randint(Environment.x_range[0], Environment.x_range[1]+1) if x is None else x
        y = np.random.randint(Environment.y_range[0], Environment.y_range[1]+1) if y is None else y
        self.loc = np.array([x, y])
        self.init_loc = self.loc.copy()
        self.steps = 0 


    def valid_loc(self):
        x, y = self.loc[0], self.loc[1]
        return x >= Environment.x_range[0] and x <= Environment.x_range[1] and y >= Environment.y_range[0] and y <= Environment.y_range[1]

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

        x, y = loc[0], loc[1]

        r = np.sqrt(x**2 + y**2) 
        return 1/(1 + r)
    

    def found_target(self):
        return np.array_equal(self.loc, Environment.food_source_loc)
    
    def reset(self):
        self.loc = self.init_loc.copy()
        self.steps = 0