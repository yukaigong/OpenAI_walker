import torch
import torch.nn as nn
import torch.nn.functional as F
from policies.base import normc_fn


class FF_V(nn.Module):
    def __init__(self, state_dim, layers=(128,128)):
        super(FF_V, self).__init__()

        self.critic_layers = nn.ModuleList()
        self.critic_layers += [nn.Linear(state_dim, layers[0])]
        for i in range(len(layers)-1):
            self.critic_layers += [nn.Linear(layers[i], layers[i+1])]
        self.network_out = nn.Linear(layers[-1],1)

    def forward(self, inputs):
        x = inputs
        for layer in self.critic_layers:
            x = F.relu(layer(x))
        value = self.network_out(x)

        return value

    def initialize_parameters(self):
        self.apply(normc_fn)