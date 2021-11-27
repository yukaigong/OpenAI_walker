import Rabbit_PP0_Train
from datetime import datetime

class arg_empty:
    pass

args = arg_empty()
args.recurrent = 1
args.run_name = "run_name"
args.max_traj_len = 4000
args.seed = 1
args.num_procs = 1
args.lr = 1e-4
args.eps = 1e-5
args.lam = 0.95
args.gamma = 0.99
args.learn_stddev = False
args.std_dev = 1.5
args.entropy_coeff = 0
args.clip = 0.2
args.minibatch_size = 64
args.epochs = 3
# args.num_steps = 5096
args.num_steps = 5096*2
args.max_grad_norm = 0.05
args.previous = None
args.sim_timestep = 0.0005
args.ctrl_timestep = 0.005 # It will decides ctrl_timestep in each episodes

now = datetime.now()
date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
args.logdir = "./trained_models/ppo/" + date_time
args.env_name = "gym_Rabbit:Rabbit-v1"
args.n_itr = 10000
# args.ann, Inf or huge value in QACC at DOF 0. neal = 1

Rabbit_PP0_Train.run_experiment(args)
