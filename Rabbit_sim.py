import gym
import numpy as np
import time
from gym import error, spaces, utils
from gym.utils import seeding
# env = gym.make('Walker2d-v3')
env = gym.make('gym_Rabbit:Rabbit-v1')
env.reset()
env.render()
t0 = 0
stanceLeg = 1
u = np.zeros(4)
# env.model.opt.timestep = 0.001
env.viewer._paused = True
env.viewer._render_every_frame = True
# env._time_per_render = 1
# env.viewer._record_video = True
env.render()



for _ in range(10000):

    env.render()

    observation, reward, done, info = env.step(u)

    # To regulate the render speed
    print('new iter')
    print(env.sim.data.time)
    # frame skip Defined by Rabbit_env, in another folder
    sleep_time = (env.frame_skip*env.model.opt.timestep - time.time()%(env.frame_skip*env.model.opt.timestep))
    time.sleep(sleep_time)

env.close()
