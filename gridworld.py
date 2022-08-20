import numpy as np

# the grid world class
class GridWorld:
    def __init__(self) -> None:
        # world width
        self.WORLD_WIDTH = 15

        # world height
        self.WORLD_HEIGHT = 10

        # reward for each step
        self.REWARD = -1

        # all possible actions
        self.ACTION_UP = 0
        self.ACTION_DOWN = 1
        self.ACTION_LEFT = 2
        self.ACTION_RIGHT = 3
        self.ACTIONS = [self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT]

        # start state (y, x)
        self.START = [6, 1]

        # goal states (y, x)
        self.GOAL = [8, 11]

        # all obstacles
        #self.obstacles = []
        self.obstacles = [[2,3], [3,3], [7,3], [8,3], [9,3],
                            [7,8], [8,8], [9,8]]
        
        # step count
        self.STEP_CNT = 0

        # wind strength for each row
        self.WIND = [0, 0, 0, 0, 1, 2, 1, 0, 0, 0]
        
        # probability of wind striking, this is aimed at simulating turbulence
        self.WIND_PROB = 0.8

        # max steps
        self.max_steps = float('inf')

    # the step function
    def step(self, state, action):
        """
        Takes a step in the grid world while following the constraints and simulating wind

        Arguments:
            state: a tuple specifying the current state, i.e. coordiantes in [y,x] format
            action: an integer denoting either of ther four possible actions

        Returns:
            a tuple of the following -
            next_state: a tuple specifying the next state of the agent
            reward: the reward incured for taking the step
        """

        # simulates wind effect 80% of the times
        wind = np.zeros(len(self.WIND))
        if np.random.binomial(1, self.WIND_PROB) == 1:
            wind = self.WIND

        y, x = state
        if action == self.ACTION_UP:
            y = max(y - 1, 0)
            x = int(max(x - wind[y], 0))
        elif action == self.ACTION_DOWN:
            y = min(y + 1, self.WORLD_HEIGHT - 1)
            x = int(max(x - wind[y], 0))
        elif action == self.ACTION_LEFT:
            x = int(max(x - 1 - wind[y], 0))
        elif action == self.ACTION_RIGHT:
            x = int(max(min(x + 1 - wind[y], self.WORLD_WIDTH - 1),0))
        else:
            raise ValueError(f'action passed is {action}, but only 0, 1, 2, 3 are accepted')

        if [y, x] in self.obstacles:
            y, x = state
        if [y, x] == self.GOAL:
            reward = 0.0
        else:
            reward = self.REWARD

        return [y, x], reward
