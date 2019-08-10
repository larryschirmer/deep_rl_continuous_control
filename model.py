import torch
from torch import nn
from torch.nn import functional as F


class ActorCritic(nn.Module):
    def __init__(self, input_dim, shared_hidden0, shared_hidden1, shared_hidden2, critic_hidden, actor_hidden, output_dim_actor, output_dim_critic):
        super(ActorCritic, self).__init__()
        self.shared_linear0 = nn.Linear(input_dim, shared_hidden0)
        self.shared_linear1 = nn.Linear(shared_hidden0, shared_hidden1)
        self.shared_linear2 = nn.Linear(shared_hidden1, shared_hidden2)

        self.actor_linear0 = nn.Linear(shared_hidden2, actor_hidden)
        self.actor_linear1 = nn.Linear(actor_hidden, output_dim_actor)

        self.critic_linear1 = nn.Linear(shared_hidden2, critic_hidden)
        self.critic_linear2 = nn.Linear(critic_hidden, output_dim_critic)

    def forward(self, x):
        y = torch.relu(self.shared_linear0(x))
        y = torch.relu(self.shared_linear1(y))
        y = torch.relu(self.shared_linear2(y))

        a = torch.relu(self.actor_linear0(y))
        actor_mean = torch.tanh(self.actor_linear1(a))
        actor_std = torch.relu(self.actor_linear1(a)) + 0.001

        c = F.relu(self.critic_linear1(y.detach()))
        critic = torch.tanh(self.critic_linear2(c))
        return actor_mean, actor_std, critic
