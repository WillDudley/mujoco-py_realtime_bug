import numpy as np
import gym
import timeit

env = gym.make('Humanoid-v2')
env.reset()

print(f"Target iteration duration is 1s."
      f"\nEach step advances {env.dt}s."
      f"\nHence executing 1/{env.dt}={1/env.dt} steps should make every iteration approx 1s, as desired."
      f"\nmujoco-py claims to render in real-time, so the entire experiment should be 20s long.")

n_iterations = 20
experiment_start_time = timeit.default_timer()

for iteration in range(n_iterations):
    action = np.random.rand(17)

    for step in range(int(1/env.dt)):  # If we want a step duration other than 1, change 1 to the desired duration.
        step_start_time = timeit.default_timer()

        env.step(action)
        env.render()

        print(f"Iteration {iteration} of {n_iterations}, step {step} of {int(1/env.dt)}."
              f"\nDuration of step should be {env.dt}s."
              f"\nActual duration of step is {timeit.default_timer() - step_start_time}s."
              f"\n********************************************")

env.close()

print(f"Duration of simulation should be 20s."
      f"\nActual duration of simulation is {timeit.default_timer() - experiment_start_time}s.")