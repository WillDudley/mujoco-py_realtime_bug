import numpy as np
import time
import gym
import timeit
import os

os.environ['LD_LIBRARY_PATH'] = '$LD_LIBRARY_PATH:/home/will/.mujoco/mujoco200/bin' # workaround for trouble setting path

env = gym.make('Humanoid-v2')
env.reset()

print(f"Target iteration duration is 1s."
      f"\nEach step advances (1/25)s, so we need to ensure env.dt is 1/25."  # 25 is chosen as dt and 0.002 as timestep so frame_skip can be an integer
      f"\nHence by padding each step we ensure each step lasts (1/25)s, and is thus in sync with real-time"
      f"\nThis assumes env.render() renders one frame at the system clock time.")

env.model.opt.timestep = 0.008  # default timestep for Humanoid-v2 is 0.003
env.frame_skip = 5  # this value can't be written for some reason, but it's already 5
print(env.model.opt.timestep)
print(env.frame_skip)
print(env.dt)
assert (1/25)*1.01 > env.dt > (1/25)*0.99

n_iterations = 20
experiment_start_time = timeit.default_timer()

for iteration in range(n_iterations):
    action = np.random.rand(17)

    for step in range(25):
        step_start_time = timeit.default_timer()

        env.step(action)
        env.render()

        time.sleep(0.01)

        time_threshold = step_start_time + (1/25) # If we want a step duration other than 1, change 1 to the desired duration.
        while timeit.default_timer() < time_threshold:
            pass

        print(f"Iteration {iteration} of {n_iterations}, step {step} of {25}."
              f"\nDuration of step should be {1/25}s."
              f"\nActual duration of step is {timeit.default_timer() - step_start_time}s."
              f"\n********************************************")

env.close()

print(f"Duration of simulation should be 20s."
      f"\nActual duration of simulation is {timeit.default_timer() - experiment_start_time}s.") # meant to be 25FPS, in actuality it looks like ~2FPS