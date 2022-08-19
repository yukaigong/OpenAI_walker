import gym
import numpy as np
import time
from gym import error, spaces, utils
from gym.utils import seeding
import torch
# env = gym.make('Walker2d-v3')

policy_path = "/home/gyk/Robot/OpenAI_walker/policy_params/2021_11_28_19_24_57/policy_1800.pt"

policy = torch.load(policy_path)
env_test = gym.make('gym_Rabbit:Rabbit-v1')

env = env_test

q0 = np.array([0,0.8,0,np.pi/6,-np.pi/3,np.pi/12,-np.pi/6])
dq0 = np.zeros(7)
env.reset()
env.set_state(q0,dq0)
env.render()

env.model.opt.timestep = 0.001
env.viewer._paused = True
# env.viewer._render_every_frame = True
# env._time_per_render = 1
# env.viewer._record_video = True
env.render()




t0 = 0
stanceLeg = 1
u = torch.zeros(4)
for _ in range(500000):

    env.render()
    observation, reward, done, info = env.step(u.detach().numpy())

    state = torch.Tensor(observation)
    u = policy(state,deterministic = True)
    sleep_time = (env.frame_skip*env.model.opt.timestep - time.time()%(env.frame_skip*env.model.opt.timestep))
    time.sleep(sleep_time)

env.close()
