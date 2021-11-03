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
    # time.sleep(1)
    env = gym.make('gym_Rabbit:Rabbit-v1')
    env.reset()
    q0 = np.array([0, 0.8, 0, np.pi / 6, -np.pi / 3, np.pi / 12, -np.pi / 6])
    # print(q0)
    dq0 = np.zeros(7)
    env.reset()
    env.set_state(q0, dq0)


    env.model.opt.timestep = 0.001
    t0 = 0
    stanceLeg = 1
    u = np.zeros(4)
    for _ in range(1000):
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

        Kp = 200 +np.random.randint(0, 3) * 10
        Kd = 10+ np.random.randint(0, 3)

        Kp_torso = 200
        Kd_torso = 10

        knee_straight_ref = -np.pi / 6
        knee_bending_ref = np.pi / 3
        q_st_knee_ref = knee_straight_ref
        # q_st_foot_ref = -knee_straight_ref / 2

        q_sw_thigh_ref = (-knee_straight_ref + knee_bending_ref * 4 * s * (1 - s)) / 2
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
        # print(q)
        # print(observation)
        # print(worker_ind)
    return 0


ray.init(num_cpus=4, ignore_reinit_error=True, include_dashboard=True)

# env = gym.make('gym_Rabbit:Rabbit-v1')
start_time = time.time()
# args =(env)

# workers = [worker.remote(env,200 + np.random.randint(0, 9) * 10) for _ in range(4)]
# samples = [ray.get(sample.remote(_)) for _ in range(8)]
samples = [sample.remote() for _ in range(2)]
result = sum(ray.get(samples))
# result = [worker(_) for _ in range(4)]
end_time = time.time()

time.sleep(1)
# print(result)
print(-start_time + end_time)
ray.timeline(filename="timeline03.json")