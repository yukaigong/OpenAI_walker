import gym
import numpy as np
from time import sleep
env = gym.make('Walker2d-v2')
# env = gym.make('CartPole-v0')
env.reset()
env._record_video = True
t0 = 0
stanceLeg = 1
u = np.zeros(6)

env.render()
env.viewer._paused = True
observation, reward, done, info = env.step(env.action_space.sample())
# env.viewer._record_video = True
for _ in range(1000):
    # take a random action
    #import pdb
    #pdb.set_trace()
    # sleep(0.1)
    q = env.sim.data.qpos
    dq = env.sim.data.qvel
    t_total = env.sim.data.time
    T = 0.3
    s = (t_total - t0) / T
    if s > 1:
        stanceLeg = -stanceLeg
        t0 = t_total

    Kp = 5
    Kd = 0.1

    Kp_torso = 10
    Kd_torso = 0.4

    knee_straight_ref = -np.pi / 6
    knee_bending_ref = np.pi / 3
    q_st_knee_ref = knee_straight_ref
    q_st_foot_ref = -knee_straight_ref / 2

    q_sw_thigh_ref = -knee_straight_ref / 2 + knee_bending_ref / 2 * 4 * s * (1 - s)
    q_sw_knee_ref = knee_straight_ref - knee_bending_ref * 4 * s * (1 - s)
    q_sw_foot_ref = - q_sw_knee_ref / 2

    dq_st_knee_ref = 0
    dq_sw_thigh_ref = 0
    dq_sw_knee_ref = 0

    q_sw_thigh_ref += 0.2*(dq[0]-0)
    # if stanceLeg == 1:
    #     u[0] = -Kp_torso * q[2] - Kd_torso * dq[2]
    #     u[1] = -Kp * (q[4] - q_st_knee_ref) - Kd * (dq[4] - dq_st_knee_ref)
    #     u[2] = -Kp * (q[5] - q_st_foot_ref) - Kd * (dq[5])
    #
    #     u[3] = -Kp * (q[6] - q_sw_thigh_ref) - Kd * (dq[6] - dq_sw_thigh_ref)
    #     u[4] = -Kp * (q[7] - q_sw_knee_ref) - Kd * (dq[7] - dq_sw_knee_ref)
    #     u[5] = -Kp * (q[8] - q_sw_foot_ref) - Kd * (dq[8 ])
    # else:
    #     u[3] = -Kp_torso * q[2] - Kd_torso * dq[2]
    #     u[4] = -Kp * (q[7] - q_st_knee_ref) - Kd * (dq[7] - dq_st_knee_ref)
    #     u[5] = -Kp * (q[8] - q_sw_foot_ref) - Kd * (dq[8])
    #
    #     u[0] = -Kp * (q[3] - q_sw_thigh_ref) -  Kd * (dq[3] - dq_sw_thigh_ref)
    #     u[1] = -Kp * (q[4] - q_sw_knee_ref) - Kd * (dq[4] - dq_sw_knee_ref)
    #     u[2] = -Kp * (q[5] - q_sw_foot_ref) - Kd * (dq[5])

    u = np.zeros(6)
    u[0] = -Kp_torso * q[2] - Kd_torso * dq[2]
    # u[0:2] = 10
    env.render()
    observation, reward, done, info = env.step(u)
    # print(done)
    # print(observation)
    print(env.sim.data.qpos)
    print(env.sim.data.qvel)
    print(env.sim.data.time)
    print(stanceLeg)
    print(u)
env.close()
