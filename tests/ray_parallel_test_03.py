import gym
import numpy as np
import time
import ray
import random
from gym import error, spaces, utils
from gym.utils import seeding
# env = gym.make('Walker2d-v3')


@ray.remote
def sample():
    time.sleep(10)
    return 0


ray.init(num_cpus=4, ignore_reinit_error=True, include_dashboard=True)


start_time = time.time()
samples = [sample.remote() for _ in range(4)]
result = sum(ray.get(samples))
end_time = time.time()

# print(result)
print(-start_time + end_time)


time.sleep(1)
ray.timeline(filename="timeline03.json")