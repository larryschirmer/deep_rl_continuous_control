import torch
from torch import nn
from torch.nn import functional as F


class ActorCritic(nn.Module):
    def __init__(self, input_dim, shared_hidden0, shared_hidden1, shared_hidden2, critic_hidden, output_dim_actor, output_dim_critic):
        super(ActorCritic, self).__init__()
        self.shared_linear0 = nn.Linear(input_dim, shared_hidden0)
        self.shared_linear1 = nn.Linear(shared_hidden0, shared_hidden1)
        self.shared_linear2 = nn.Linear(shared_hidden1, shared_hidden2)

        self.actor_linear = nn.Linear(shared_hidden2, output_dim_actor)

        self.critic_linear1 = nn.Linear(shared_hidden1, critic_hidden)
        self.critic_linear2 = nn.Linear(critic_hidden, output_dim_critic)

    def forward(self, x):
        y = torch.tanh(self.shared_linear0(x))
        y = torch.tanh(self.shared_linear1(y))
        y = torch.tanh(self.shared_linear2(y))

        actor = self.actor_linear(y)

        c = F.relu(self.critic_linear1(y.detach()))
        critic = torch.tanh(self.critic_linear2(c))
        return actor, critic
