import torch
import torch.nn as nn
import torch.nn.functional as F
from policies.base import normc_fn

from torch import sqrt

class FF_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, layers = (128,128), env_name = None, nonlinearity = F.relu, max_action = 1):
        super(FF_Actor,self).__init__()

        self.actor_layers = nn.ModuleList()

        self.actor_layers += [nn.Linear(state_dim,layers[0])]
        for i in range(len(layers)-1):
            self.actor_layers +=[nn.Linear(layers[i],layers[i+1])]
        self.network_out = nn.Linear(layers[-1], action_dim)

        self.action = None
        self.action_dim = action_dim
        self.env_name = env_name
        # self.nonlinearity = nonlinearity

        self.initialize_parameters()

    def forward(self, state):
        x = state
        for idx, layer in enumerate(self.actor_layers):
            x = F.relu(layer(x))

        self.action = self.network_out(x)

    def initialize_parameters(self):
        self.apply(normc_fn)

    def get_action(self):
        return self.action

class Gaussian_FF_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, layers = (256,256), fixed_std = 1):
        super(Gaussian_FF_Actor, self).__init__()

        self.actor_layers = nn.ModuleList()
        self.actor_layers += [nn.Linear(state_dim, layers[0])]
        for i in range(len(layers) - 1):
            self.actor_layers += [nn.Linear(layers[i], layers[i+1])]
        self.network_out = nn.Linear(layers[-1], action_dim)

        self.action = None
        self.action_dim = action_dim
        self.nonlinearity = nn.functional.relu

        self.fixed_std = fixed_std

        self.init_parameters()

    def init_parameters(self):
        self.apply(normc_fn)
        self.network_out.weight.data.mul_(0.01)

    def _get_dist_params(self, state):

        x = state
        for l in self.actor_layers:
            x = self.nonlinearity(l(x))
        mean_unsat = self.network_out(x)
        sat_torque = 200
        mean = sat_torque*torch.tanh(mean_unsat)

        sd = self.fixed_std

        return mean, sd

    def forward(self, state, deterministic = True, anneal = 1.0):
        mu, sd = self._get_dist_params(state)
        sd *= anneal

        if not deterministic:
            self.action = torch.distributions.Normal(mu, sd).sample()
        else:
            self.action = mu

        return self.action

    def get_action(self):
        return self.action

    def distribution(self, inputs):
        mu, sd = self._get_dist_params(inputs)
        # import pdb
        # pdb.set_trace()
        return torch.distributions.Normal(mu,sd)



