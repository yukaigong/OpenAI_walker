import Rabbit_PP0_Train

class arg_empty:
    pass

args = arg_empty()
args.recurrent = 1
args.run_name = "run_name"
args.max_traj_len = 400
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
args.num_steps = 5096
args.max_grad_norm = 0.05
args.logdir = "./trained_models/ppo/"
args.env_name = "'gym_Rabbit:Rabbit-v1'"
args.n_itr = 10000
args.anneal = 1

Rabbit_PP0_Train.run_experiment(args)
