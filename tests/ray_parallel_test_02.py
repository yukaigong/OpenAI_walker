import gym
import numpy as np
import time
import ray
import random
from gym import error, spaces, utils
from gym.utils import seeding
# env = gym.make('Walker2d-v3')


@ray.remote
def parallel():
    time.sleep(1)
    return 1

def serial():
    time.sleep(1)
    return 1

ray.init(num_cpus=4, ignore_reinit_error=True, include_dashboard=True)


start_time = time.time()
sum(ray.get([parallel.remote() for _ in range(4)]))
end_time = time.time()

print(end_time - start_time)

# start_time = time.time()
# sum([serial() for _ in range(4)])
# end_time = time.time()
# print(end_time - start_time)

time.sleep(1)

ray.timeline(filename="timeline02.json")