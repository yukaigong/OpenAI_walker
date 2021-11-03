import torch
from policies.actor import FF_Actor

a = FF_Actor(state_dim=3, action_dim=4)

# a.find_submodule()
a.actor_layers