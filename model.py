import torch
import numpy as np
from torch import nn

torch.manual_seed(0)


class ActorCritic(nn.Module):
    def __init__(self, params):
        super(ActorCritic, self).__init__()
        self.start_epsilon = params['start_epsilon']
        self.end_epsilon = params['end_epsilon']
        self.epochs = params['epochs']
        self.shared_linear0 = nn.Linear(params['input_dim'], params['shared_hidden0'])
        self.shared_linear1 = nn.Linear(params['shared_hidden0'], params['shared_hidden1'])
        self.shared_linear2 = nn.Linear(params['shared_hidden1'], params['shared_hidden2'])

        self.actor_linear0 = nn.Linear(params['shared_hidden2'], params['actor_hidden'])
        self.actor_linear1 = nn.Linear(params['actor_hidden'], params['actor_hidden'])
        self.actor_linear2 = nn.Linear(params['actor_hidden'], params['output_dim_actor'])

        self.critic_linear0 = nn.Linear(params['shared_hidden2'], params['critic_hidden'])
        self.critic_linear1 = nn.Linear(params['critic_hidden'], params['critic_hidden'])
        self.critic_linear2 = nn.Linear(params['critic_hidden'], params['output_dim_critic'])

    def forward(self, x, epoch):
        epsilon = np.clip(
            (self.end_epsilon - self.start_epsilon)
            / (self.epochs - 0) * epoch + self.start_epsilon, self.end_epsilon, self.start_epsilon)

        actor_std = torch.tensor(epsilon)
        
        y = torch.tanh(self.shared_linear0(x))
        y = torch.tanh(self.shared_linear1(y))
        y = torch.tanh(self.shared_linear2(y))

        a0 = torch.tanh(self.actor_linear0(y))
        a0 = torch.tanh(self.actor_linear1(a0))
        actor0 = self.actor_linear2(a0)
        actor_mean0 = torch.tanh(actor0)

        a1 = torch.tanh(self.actor_linear0(y))
        a1 = torch.tanh(self.actor_linear1(a1))
        actor1 = self.actor_linear2(a1)
        actor_mean1 = torch.tanh(actor1)

        a2 = torch.tanh(self.actor_linear0(y))
        a2 = torch.tanh(self.actor_linear1(a2))
        actor2 = self.actor_linear2(a2)
        actor_mean2 = torch.tanh(actor2)

        a3 = torch.tanh(self.actor_linear0(y))
        a3 = torch.tanh(self.actor_linear1(a3))
        actor3 = self.actor_linear2(a3)
        actor_mean3 = torch.tanh(actor3)

        action_dist0 = torch.distributions.Normal(actor_mean0, actor_std)
        action_dist1 = torch.distributions.Normal(actor_mean1, actor_std)
        action_dist2 = torch.distributions.Normal(actor_mean2, actor_std)
        action_dist3 = torch.distributions.Normal(actor_mean3, actor_std)

        c = torch.relu(self.critic_linear0(y.detach()))
        c = torch.relu(self.critic_linear1(c))
        critic = torch.relu(self.critic_linear2(c))
        return [action_dist0, action_dist1, action_dist2, action_dist3], critic
