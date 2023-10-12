import gridworld
import numpy as np

# create a world object
world = gridworld.GridWorld()
rewards = np.full((world.WORLD_HEIGHT, world.WORLD_WIDTH), world.REWARD)
for i in world.obstacles:
    rewards[i[0]][i[1]] = -100

rewards[world.GOAL[0], world.GOAL[1]] = 100

q_values = np.zeros((world.WORLD_HEIGHT, world.WORLD_WIDTH, 4))

def is_terminal_state(current_row_index, current_column_index):
    # Return True if the current state is an obstacle, otherwise False
    return [current_row_index, current_column_index] in world.obstacles


#finding a random location to train the bot and store all different Q value outcome
def random_starting_location():
    # get a random row and column index
    current_column_index = np.random.randint(world.WORLD_WIDTH)
    current_row_index = np.random.randint(world.WORLD_HEIGHT)
    # continue choosing random row and column indexes until a non-terminal state is identified
    # (i.e., until the chosen state is a 'white square').
    while is_terminal_state(current_row_index, current_column_index):
        current_column_index = np.random.randint(world.WORLD_WIDTH)
        current_row_index = np.random.randint(world.WORLD_HEIGHT)
    return current_row_index, current_column_index

def next_action(current_row_index, current_column_index, epsilon):
    # finding next action using epsilon method
    # if a randomly chosen value between 0 and 1 is less than epsilon,
    # then choose the most promising value from the Q-table for this state.
    if np.random.random() < epsilon:
        return np.argmax(q_values[current_row_index, current_column_index])
    else:  # choose a random action
        return np.random.randint(4)


def next_location(current_row_index, current_column_index, action_to_take):
    # Update the new_row_index and new_column_index based on the chosen action
    if action_to_take == 0:  # move up
        new_row_index = max(0, current_row_index - 1)
        new_column_index = current_column_index
    elif action_to_take == 1:  # move down
        new_row_index = min(world.WORLD_HEIGHT - 1, current_row_index + 1)
        new_column_index = current_column_index
    elif action_to_take == 2:  # move left
        new_row_index = current_row_index
        new_column_index = max(0, current_column_index - 1)
    elif action_to_take == 3:  # move right
        new_row_index = current_row_index
        new_column_index = min(world.WORLD_WIDTH - 1, current_column_index + 1)

    next_state, reward = world.step([new_row_index, new_column_index], action_to_take)
    return next_state[0], next_state[1]


# define training parameters
epsilon = 0.9
discount_factor = 1 
learning_rate = 0.9

# running through 2000 training episodes
for episode in range(2000):
    # get a random starting location
    row_index, column_index = random_starting_location()
    
    # choose which is the best action to take
    action_to_take = np.argmax(q_values[row_index, column_index])
    
    # performing the chosen action and moving to a new position
    # store the old row and column indexes
    old_row_position, old_column_position = row_index, column_index
    row_index, column_index = next_location(row_index, column_index, action_to_take)
    
    while is_terminal_state(row_index, column_index):
        row_index, column_index = old_row_position, old_column_position
        action_to_take = np.random.randint(4)  # Choose a random action
        row_index, column_index = next_location(row_index, column_index, action_to_take)
    
    # storing the reward and calculating the temporal difference to move to the next state
    reward = rewards[row_index, column_index]
    old_q_value = q_values[old_row_position, old_column_position, action_to_take]
    temporal_difference = reward + \
        (discount_factor * np.max(q_values[row_index, column_index])) - old_q_value

    # update the Q-value
    new_q_value = old_q_value + (learning_rate * temporal_difference)
    q_values[old_row_position, old_column_position, action_to_take] = new_q_value

def get_shortest_path(start_row_index, start_column_index):
    if is_terminal_state(start_row_index, start_column_index):
        return []
    else:
        current_row_index, current_column_index = start_row_index, start_column_index
        shortest_path = []
        shortest_path.append([current_row_index, current_column_index])
        
        while [current_row_index, current_column_index] != world.GOAL:
            # finding the best action to take
            action_to_take = np.argmax(q_values[current_row_index, current_column_index])
            current_row_index, current_column_index = next_location(
                current_row_index, current_column_index, action_to_take)
            shortest_path.append([current_row_index, current_column_index])
        
        return shortest_path

print(get_shortest_path(world.START[0], world.START[1]))

