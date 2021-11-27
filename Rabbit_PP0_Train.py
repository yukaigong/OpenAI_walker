import gym
import numpy as np
import time
from gym import error, spaces, utils
from gym.utils import seeding
from policies.actor import FF_Actor, Gaussian_FF_Actor
from policies.critic import FF_V
import torch.optim as optim
import torch
import os, sys
from util.log import create_logger
from copy import deepcopy
import ray
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import kl_divergence
from util.rewards_functions import reward_func_01
from datetime import datetime

class PPOBuffer:

    def __init__(self, gamma = 0.99, lam = 0.95, use_gae = False):
        self.states = []
        self.actions = []
        self.rewards = [] # store the rewards of each data point.
        self.values = []
        self.returns = [] # store the returns of each data point.

        self.ep_returns = [] # store the returns of each episodes ( no discount)
        self.ep_lens = [] # length of each episodes
        self.gamma, self.lam = gamma, lam # gamma is the discount rate

        self.ptr = 0 # record how many data points has been stored
        self.traj_idx = [0]

    def __len__(self):
        return len(self.states)

    def storage_size(self):
        return len(self.states)

    def store(self, state, action, reward, value):
        self.states +=[state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]

        self.ptr +=1

    def finish_path(self, last_val = None):
        self.traj_idx += [self.ptr]
        rewards = self.rewards[self.traj_idx[-2]:self.traj_idx[-1]]

        returns = []

        R = last_val.copy()
        for reward in reversed(rewards):
            R = self.gamma*R + reward
            returns.insert(0,R)

        self.returns += returns

        self.ep_returns +=[np.sum(rewards)]
        self.ep_lens += [len(rewards)]

    def get(self):
        return(
            self.states,
            self.actions,
            self.returns,
            self.values
        )

class PPO:
    def __init__(self,args, save_path):
        self.env_name       = args['env_name']
        self.gamma          = args['gamma']
        self.lam            = args['lam']
        self.lr             = args['lr']
        self.eps            = args['eps']
        self.entropy_coeff  = args['entropy_coeff']
        self.clip           = args['clip']
        self.minibatch_size = args['minibatch_size']
        self.epochs         = args['epochs']
        self.num_steps      = args['num_steps']
        self.max_traj_len   = args['max_traj_len']
        # self.use_gae        = args['use_gae']
        self.n_proc         = args['num_procs']
        self.grad_clip      = args['max_grad_norm']
        self.recurrent      = args['recurrent']
        self.sim_timestep = args['sim_timestep']
        self.ctrl_timestep = args['ctrl_timestep']

        self.total_steps = 0
        self.highest_reward = -1
        self.limit_cores = 0

        self.save_path = save_path

    def save(self, policy, critic):
        os.makedirs(self.save_path)
        filetype = ".pt" #pytorch model
        torch.save(policy, os.path.join(self.save_path,"actor"+filetype))
        torch.save(critic, os.path.joint(self.save_path, "critic" + filetype))

    # @ray.remote
    @torch.no_grad()
    def sample(self, policy, critic, min_steps, max_traj_len, deterministic = False, anneal =1.0, term_thresh = 0 ):
        # torch.set_num_threads(1)
        env = gym.make('gym_Rabbit:Rabbit-v1')
        env.model.jnt_range[[3, 5], :] = (np.pi / 13, np.pi * 11 / 13)
        env.model.jnt_range[[4, 6], :] = (-np.pi * 11 / 13, -np.pi / 13)
        env.reset()
        env.model.opt.timestep = self.sim_timestep
        # env.model.opt.timestep = 0.001
        env.frameskip = self.ctrl_timestep/self.sim_timestep
        memory = PPOBuffer(self.gamma, self.lam)

        num_steps  = 0
        while num_steps < min_steps:
            state = torch.Tensor(env.reset())

            done = False
            value = 0
            traj_len = 0
            # print(state)
            # while not done and traj_len<max_traj_len:
            while state[1]>0.3 and state[1]<1.2 and traj_len < max_traj_len:
                # print(state)
                action = policy(state,deterministic = False, anneal = anneal) # Should implement a probabilistic policy here for exploring
                # action = torch.clamp(action, min=-200, max=200)
                # print(action)
                # print(state)
                value = critic(state)
                # time.sleep(0.002)
                next_state, reward, done, _ = env.step(action.detach().numpy())
                reward = reward_func_01(next_state[0:7],next_state[7:],action.detach().numpy()) + 1
                memory.store(state.numpy(),action.detach().numpy(), reward, value.detach().numpy())

                state = torch.Tensor(next_state)

                traj_len +=1
                num_steps +=1

            value = critic(state)
            memory.finish_path(last_val = (not done) * value.detach().numpy())

        return memory

    def sample_parallel(self, policy, critic, min_steps, max_traj_len, deterministic = False, anneal = 1, term_thresh = 0):

        worker = self.sample
        args = (self, policy, critic, min_steps // self.n_proc, max_traj_len, deterministic, anneal, term_thresh )

        workers = [worker.remote(*args) for _ in range(self.n_proc)]

        result = []
        total_steps = 0

        while total_steps < min_steps:
            # get result from a worker
            ready_ids, _ = ray.wait(workers, num_returns = 1)

            # update result
            result.append(ray.get(ready_ids[0]))
            # remove ready_ids from workers (O(n)) but n isn't that big
            workers.remove(ready_ids[0])

            # update total steps
            total_steps += len(result[-1])

            # start a new worker
            workers.append(worker.remote(*args))

            def merge(buffers):
                merged = PPOBuffer(self.gamma, self.lam)
                for buf in buffers:
                    offset = len(merged)

                    merged.states += buf.states
                    merged.actions += buf.actions
                    merged.rewards += buf.rewards
                    merged.values += buf.values
                    merged.returns += buf.returns

                    merged.ep_returns += buf.ep_returns
                    merged.ep_lens += buf.ep_lens

                    merged.traj_idx += [ offset + i for i in buf.traj_idx[1:]]
                    merged.ptr += buf.ptr

                return merged

        total_buf = merge(result)

        return total_buf
    def update_policy(self,obs_batch, action_batch, return_batch, advantage_batch, mask):
        policy = self.policy
        critic = self.critic
        old_policy = self.old_policy
        policy_temp = self.policy
        values = critic(obs_batch)
        try:
            pdf = policy.distribution(obs_batch)
        except:
            import pdb
            pdb.set_trace()

        with torch.no_grad():
            old_pdf = old_policy.distribution(obs_batch)
            old_log_probs = old_pdf.log_prob(action_batch).sum(-1, keepdim = True)
        log_probs = pdf.log_prob(action_batch).sum(-1, keepdim = True)

        ratio = (log_probs - old_log_probs).exp()

        cpi_loss = ratio * advantage_batch * mask
        clip_loss = ratio.clamp(1.0 - self.clip, 1.0 + self.clip) * advantage_batch * mask
        actor_loss = -torch.min(cpi_loss, clip_loss).mean()

        critic_loss = 0.5 * ((return_batch - values) * mask).pow(2).mean()

        entropy_penalty =( -self.entropy_coeff * pdf.entropy() * mask).mean()

        self.actor_optimizer.zero_grad()
        (actor_loss + entropy_penalty).backward() # It is trying to minimize instant entropy not tototal entropy. It might be difficult to caculate total entropy for each state?
        torch.nn.utils.clip_grad_norm_(policy.parameters(), self.grad_clip)
        # Gradient clipped is removed for actor
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), self.grad_clip)
        # Gradient clipped is removed for critics
        self.critic_optimizer.step()

        with torch.no_grad():
            kl = kl_divergence(pdf,old_pdf)

        params = list(policy.parameters())
        for i in range(len(params)):
            if params[i].mean().isnan():
                import pdb
                pdb.set_trace()

        return actor_loss.item(), pdf.entropy().mean().item(), critic_loss.item(), ratio.mean().item(), kl.mean().item()


    def train(self, policy, critic, n_itr, logger =None, anneal_rate = 1.0):
        now = datetime.now()
        date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
        policy_save_path = "./policy_params/"+date_time
        os.mkdir(policy_save_path)

        self.old_policy = deepcopy(policy)
        self.policy = policy
        self.critic = critic

        self.actor_optimizer = optim.Adam(policy.parameters(),lr = self.lr, eps = self.eps)
        self.critic_optimizer = optim.Adam(critic.parameters(), lr=self.lr, eps = self.eps)

        start_time = time.time()

        # env = env_fn()

        curr_anneal = 1
        curr_thresh = 0
        start_itr = 0
        ep_counter = 0
        do_term = False
        for itr in range(n_itr):
            print("********** Iteration {} ************".format(itr))

            sample_start = time.time()
            # if self.highest_reward > (2/3)*self.max_traj_len and curr_anneal > 0.5:
            #     curr_anneal *=anneal_rate
            curr_anneal = 0.01 + (1 - itr/n_itr)
            # batch = self.sample_parallel( self.policy, self.critic, self.num_steps, self.max_traj_len, anneal = curr_anneal, term_thresh = curr_thresh)
            batch = self.sample(self.policy, self.critic, self.num_steps, self.max_traj_len, anneal=curr_anneal, term_thresh=curr_thresh)
            print("time elapsed: {:.2f} s".format(time.time()-start_time))
            sample_time = time.time() - sample_start
            print("sample time elapsed: {:.2f} s".format(sample_time))

            observations, actions, returns, values = map(torch.Tensor, batch.get())

            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)

            minibatch_size = self.minibatch_size

            print("timesteps in batch: %i" % advantages.numel())
            self.total_steps += advantages.numel()
            self.old_policy.load_state_dict(policy.state_dict())

            optimizer_start = time.time()

            for epoch in range(self.epochs):
                losses = []
                entropies = []
                kls = []

                random_indices = SubsetRandomSampler(range(advantages.numel()))
                sampler = BatchSampler(random_indices, minibatch_size, drop_last = True)

                for indices in sampler:
                    obs_batch = observations[indices]
                    action_batch = actions[indices]
                    return_batch = returns[indices]
                    advantage_batch = advantages[indices]
                    mask = 1

                    scalars = self.update_policy(obs_batch, action_batch, return_batch, advantage_batch, mask)
                    actor_loss, entropy, critic_loss, ratio, kl = scalars

                    entropies.append(entropy)
                    kls.append(kl)
                    losses.append([actor_loss, entropy, critic_loss, ratio, kl])
                    # Early stopping
                    if np.max(kls) > 0.5:
                        print("Max kl reached, stopping optimization early.")
                        break

                # Early stopping
                if np.max(kls) > 0.5:
                    print("Max kl reached, stopping optimization early.")
                    break

            opt_time = time.time() - optimizer_start
            print("optimizer time elapsed: {:.2f} s".format(opt_time))

            ep_counter+=1

            if do_term == False and ep_counter > 50:
                do_term = True
                start_iter = itr

            if logger is not None:
                evaluate_start = time.time()
                # test = self.sample_parallel( self.policy, self.critic, self.num_steps//2, self.max_traj_len, deterministic = True)
                test = self.sample(self.policy, self.critic, self.num_steps // 2, self.max_traj_len, deterministic=True)
                eval_time = time.time() - evaluate_start
                print("evaluate time elapsed: {:.2f} s".format(eval_time))

                avg_eval_reward = np.mean(test.ep_returns)
                avg_batch_reward = np.mean(batch.ep_returns)
                avg_ep_len = np.mean(batch.ep_lens)
                total_ep_len = np.sum(batch.ep_lens)
                mean_losses = np.mean(losses, axis =0)
                print("avg eval reward: {:.2f}".format(avg_eval_reward))

                sys.stdout.write("-" * 37 + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Return (test)', avg_eval_reward) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Return (batch)', avg_batch_reward) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Mean Eplen', avg_ep_len) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Mean KL Div', "%8.3g" % kl) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Mean Entropy', "%8.3g" % entropy) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Total Eplen', total_ep_len) + "\n")

                sys.stdout.write("-" * 37 + "\n")
                sys.stdout.flush()

                entropy = np.mean(entropies)
                kl = np.mean(kls)

                logger.add_scalar("Test/Return", avg_eval_reward, itr)
                logger.add_scalar("Train/Return", avg_batch_reward, itr)
                logger.add_scalar("Train/Mean Eplen", avg_ep_len, itr)
                logger.add_scalar("Train/Mean KL Div", kl, itr)
                logger.add_scalar("Train/Mean Entropy", entropy, itr)


                logger.add_scalar("Misc/Critic Loss", mean_losses[2], itr)
                logger.add_scalar("Misc/Actor Loss", mean_losses[0], itr)
                # logger.add_scalar("Misc/Mirror Loss", mean_losses[5], itr)
                logger.add_scalar("Misc/Timesteps", self.total_steps, itr)

                logger.add_scalar("Misc/Sample Times", sample_time, itr)
                logger.add_scalar("Misc/Optimize Times", opt_time, itr)
                logger.add_scalar("Misc/Evaluation Times", eval_time, itr)
                logger.add_scalar("Misc/Termination Threshold", curr_thresh, itr)
            if itr % 100 == 0:
                torch.save(policy, policy_save_path+ "/policy_" + str(itr)+".pt")
                torch.save(critic, policy_save_path + "/critic_" + str(itr) + ".pt")
def run_experiment(args):

    # env = gym.make('Walker2d-v3')
    env_name = 'gym_Rabbit:Rabbit-v1'
    env = gym.make('gym_Rabbit:Rabbit-v1')
    policy = Gaussian_FF_Actor(state_dim = 14, action_dim = 4, layers = (128,128), fixed_std = 10)
    critic = FF_V(state_dim = 14, layers = (128,128))

    policy.train()
    critic.train()
    logger = create_logger(args)
    algo = PPO(args=vars(args), save_path=logger.dir)
    print()
    print("Synchronous Distributed Proximal Policy Optimization:")
    print(" ├ recurrent:      {}".format(args.recurrent))
    print(" ├ run name:       {}".format(args.run_name))
    print(" ├ max traj len:   {}".format(args.max_traj_len))
    print(" ├ seed:           {}".format(args.seed))
    print(" ├ num procs:      {}".format(args.num_procs))
    print(" ├ lr:             {}".format(args.lr))
    print(" ├ eps:            {}".format(args.eps))
    print(" ├ lam:            {}".format(args.lam))
    print(" ├ gamma:          {}".format(args.gamma))
    print(" ├ learn stddev:  {}".format(args.learn_stddev))
    print(" ├ std_dev:        {}".format(args.std_dev))
    print(" ├ entropy coeff:  {}".format(args.entropy_coeff))
    print(" ├ clip:           {}".format(args.clip))
    print(" ├ minibatch size: {}".format(args.minibatch_size))
    print(" ├ epochs:         {}".format(args.epochs))
    print(" ├ num steps:      {}".format(args.num_steps))
    print(" ├ max grad norm:  {}".format(args.max_grad_norm))
    print(" └ max traj len:   {}".format(args.max_traj_len))
    print()

    # algo.train( policy, critic, args.n_itr, logger=logger, anneal_rate=args.anneal)
    algo.train(policy, critic, args.n_itr, logger=logger)



