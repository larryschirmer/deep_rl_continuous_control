import torch
import numpy as np
from torch import nn

torch.manual_seed(0)


class ActorCritic(nn.Module):
    def __init__(self, params):
        super(ActorCritic, self).__init__()
        self.shared_linear0 = nn.Linear(params['input_dim'], params['shared_hidden0'])
        self.shared_linear1 = nn.Linear(params['shared_hidden0'], params['shared_hidden1'])
        self.shared_linear2 = nn.Linear(params['shared_hidden1'], params['shared_hidden2'])

        self.actor_linear0 = nn.Linear(params['shared_hidden2'], params['actor_hidden'])
        self.actor_linear1 = nn.Linear(params['actor_hidden'], params['actor_hidden'])
        self.actor_linear2 = nn.Linear(params['actor_hidden'], params['output_dim_actor'])

        self.critic_linear0 = nn.Linear(params['shared_hidden2'], params['critic_hidden'])
        self.critic_linear1 = nn.Linear(params['critic_hidden'], params['critic_hidden'])
        self.critic_linear2 = nn.Linear(params['critic_hidden'], params['output_dim_critic'])

    def forward(self, x):
        y = torch.tanh(self.shared_linear0(x))
        y = torch.tanh(self.shared_linear1(y))
        y = torch.tanh(self.shared_linear2(y))

        a = torch.tanh(self.actor_linear0(y))
        a = torch.tanh(self.actor_linear1(a))
        actor = self.actor_linear2(a)
        actor_mean = torch.tanh(actor)

        c = torch.relu(self.critic_linear0(y.detach()))
        c = torch.relu(self.critic_linear1(c))
        critic = torch.relu(self.critic_linear2(c))
        return actor_mean, critic
