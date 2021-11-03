from numpy import linalg
def reward_func_01(q,dq,u):
    height_cost = - (q[1]-0.7)**2
    vel_cost = -0.1*dq[0]**2
    pitch_cost = -q[2]**2
    torque_cost = 0.01*linalg.norm(u)
    reward = height_cost + vel_cost + pitch_cost + torque_cost
    return reward
