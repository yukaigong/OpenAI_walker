import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils
import gym


DEFAULT_CAMERA_CONFIG = {
    'trackbodyid': 2,
    'distance': 4.,
    'lookat': np.array((0.0, 0.0, 1.15)),
    'elevation': -20.0,
}

# class RabbitEnv(gym.Env):
#     metadata = {'render.modes': ['human']}
#
#     def __init__(self):
#         print('test')
#         pass
#     def step(self, action):
#         pass
#     def reset(self):
#         pass
#     def render(self, mode='human'):
#         pass
#     def close(self):
#         pass

class RabbitEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, "/home/gyk/Robot/OpenAI_walker/gym-Rabbit/Rabbit.xml", 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        # posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        # posafter, height, ang = self.sim.data.qpos[0:3]
        # alive_bonus = 1.0
        # reward = (posafter - posbefore) / self.dt
        # reward += alive_bonus
        # reward -= 1e-3 * np.square(a).sum()
        # done = not (height > 0.8 and height < 2.0 and ang > -1.0 and ang < 1.0)
        reward = 0
        done = False
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos, qvel]).ravel()

    def reset_model(self):
        # self.set_state(
        #     self.init_qpos
        #     + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nq),
        #     self.init_qvel
        #     + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv),
        # )
        # import pdb
        # pdb.set_trace()
        self.set_state(
            np.array([0,1,0,np.pi/6,-np.pi/3,np.pi/6,-np.pi/3]),
            np.zeros(7))
        # self.set_state(q,dq)
        return self._get_obs()
        # return
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
