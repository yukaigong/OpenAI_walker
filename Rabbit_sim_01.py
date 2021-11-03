import gym
import numpy as np
import time
from gym import error, spaces, utils
from gym.utils import seeding
# env = gym.make('Walker2d-v3')
env = gym.make('gym_Rabbit:Rabbit-v1')


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
u = np.zeros(4)
for _ in range(500000):

    env.render()
    observation, reward, done, info = env.step(u)

    q = env.sim.data.qpos
    dq = env.sim.data.qvel
    t_total = env.sim.data.time
    T = 0.4
    s = (t_total - t0) / T
    # print(s)
    if s > 1:
        stanceLeg = -stanceLeg
        t0 = t_total

    Kp = 200
    Kd = 10

    Kp_torso = 200
    Kd_torso = 10

    knee_straight_ref = -np.pi / 6
    knee_bending_ref = np.pi / 3
    q_st_knee_ref = knee_straight_ref
    # q_st_foot_ref = -knee_straight_ref / 2

    q_sw_thigh_ref = (-knee_straight_ref  + knee_bending_ref  * 4 * s * (1 - s))/2
    q_sw_knee_ref = knee_straight_ref - knee_bending_ref * 4 * s * (1 - s)
    # q_sw_foot_ref = - q_sw_knee_ref / 2

    dq_st_knee_ref = 0
    dq_sw_thigh_ref = 0
    dq_sw_knee_ref = 0

    q_sw_thigh_ref += 0.4 * (dq[0] - 0)
    u = np.zeros(4)
    if stanceLeg == -1:
        u[0] = -Kp_torso * q[2] - Kd_torso * dq[2]
        u[1] = -Kp * (q[4] - q_st_knee_ref) - Kd * (dq[4] - dq_st_knee_ref)
        # u[2] = -Kp * (q[5] - q_st_foot_ref) - Kd * (dq[5])

        u[2] = -Kp * (q[5] - q_sw_thigh_ref) - Kd * (dq[5] - dq_sw_thigh_ref)
        u[3] = -Kp * (q[6] - q_sw_knee_ref) - Kd * (dq[6] - dq_sw_knee_ref)
        # u[5] = -Kp * (q[8] - q_sw_foot_ref) - Kd * (dq[8])
    else:
        u[2] = -Kp_torso * q[2] - Kd_torso * dq[2]
        u[3] = -Kp * (q[6] - q_st_knee_ref) - Kd * (dq[6] - dq_st_knee_ref)
        # u[5] = -Kp * (q[8] - q_sw_foot_ref) - Kd * (dq[8])

        u[0] = -Kp * (q[3] - q_sw_thigh_ref) - Kd * (dq[3] - dq_sw_thigh_ref)
        u[1] = -Kp * (q[4] - q_sw_knee_ref) - Kd * (dq[4] - dq_sw_knee_ref)
        # u[2] = -Kp * (q[5] - q_sw_foot_ref) - Kd * (dq[5])








    # u = np.zeros(4)
    # u[4] = 32*9.81
    # u[0] = -Kp_torso * q[2] - Kd_torso * dq[2]
    # u[0] = 100
    # u[0] = -Kp * (q[3]) - Kd * dq[3]
    # u[1] = -Kp * (q[4]) - Kd * dq[4]
    # u[2] = -Kp * (q[5]) - Kd * dq[5]
    # u[3] = -Kp * (q[6]) - Kd * dq[6]
    # print(u)
    # print(q)
    # print(dq)
    # u[3] = 100
    # To regulate the render speed
    print('new iter')
    print(env.sim.data.time)
    print(u)
    # frame skip Defined by Rabbit_env, in another folder
    sleep_time = (env.frame_skip*env.model.opt.timestep - time.time()%(env.frame_skip*env.model.opt.timestep))
    time.sleep(sleep_time)

env.close()
