import torch
from torch import nn
from torch.nn import functional as F

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
        self.actor_linear1 = nn.Linear(params['actor_hidden'], params['output_dim_actor'])

        self.critic_linear1 = nn.Linear(params['shared_hidden2'], params['critic_hidden'])
        self.critic_linear2 = nn.Linear(params['critic_hidden'], params['output_dim_critic'])

    def forward(self, x, epoch):
        epsilon = (self.end_epsilon - self.start_epsilon) / (self.epochs - 0) * epoch + self.start_epsilon

        y = torch.tanh(self.shared_linear0(x))
        y = torch.tanh(self.shared_linear1(y))
        y = torch.tanh(self.shared_linear2(y))

        a = torch.tanh(self.actor_linear0(y))
        actor = self.actor_linear1(a)

        actor_mean = torch.tanh(actor[0])
        actor_std = torch.clamp(actor[1], min=epsilon, max=self.start_epsilon)
        action_dist = torch.distributions.Normal(actor_mean, actor_std)

        c = F.relu(self.critic_linear1(y.detach()))
        critic = torch.tanh(self.critic_linear2(c))
        return action_dist, critic
