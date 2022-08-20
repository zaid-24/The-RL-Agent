# The Grid World environment

The Grid world as described in the Task is shown below:

![grid-world](https://github.com/hskalin/swarm-gridworld/blob/main/grid.png)

You are required to write a program to move the agent to the goal. To use the gridworld, import `gridworld.py` (your script should be in the same directory as `gridworld.py`)

## Using Gridworld

```
import gridworld
```

Then you can create a world object and access various constants and the `step` function which is used to move the agent

```
# create a world object
world = gridworld.GridWorld()

# access Height and Width
print(world.WORLD_WIDTH)
print(world.WORLD_HEIGHT)

# access the obstacles
print(wolrd.obstacles)

# access start and goal position
print(world.START)
print(world.GOAL)

# access the actions, for others see gridworld.py
print(world.ACTION_UP)

# run the step function
present_state = world.START

next_state, reward = world.step(present_state, world.ACTION_UP)

print(next_state)
print(reward)

output
15
10
[[2, 3], [3, 3], [7, 3], [8, 3], [9, 3], [7, 8], [8, 8], [9, 8]]
[6, 1]
[8, 11]
0
[5, 0]
-1
```

## Important Notes
 - In the grid world the origin is at the top left corner and the coordinates are specified as (y,x) and not (x,y).
 - You should not make any changes to `gridworld.py`
