import numpy as np
import time
import gym
import timeit
import os

os.environ['LD_LIBRARY_PATH'] = '$LD_LIBRARY_PATH:/home/will/.mujoco/mujoco200/bin'  # workaround for trouble setting path

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

    time_threshold = timeit.default_timer() + 1  # If we want a step duration other than 1, change 1 to the desired duration.
    step = 0

    while timeit.default_timer() < time_threshold:
        step_start_time = timeit.default_timer()

        env.step(action)
        env.render()

        # time.sleep(0.01)

        print(f"Iteration {iteration} of {n_iterations}, step {step} of ???."
              f"\nDuration of step should be ???s."
              f"\nActual duration of step is {timeit.default_timer() - step_start_time}s."
              f"\n********************************************")

        step += 1

env.close()

print(f"Duration of simulation should be 20s."
      f"\nActual duration of simulation is {timeit.default_timer() - experiment_start_time}s.")