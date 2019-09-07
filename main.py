# dropped epsilon limit, turned up learning rate
from unityagents import UnityEnvironment
from time import perf_counter
import pandas as pd
import copy
import torch

from model import ActorCritic
from helpers import save_model, worker, plot_losses, plot_scores

# hyperparameters
epochs = 40000
lr = 0.00008
gamma = 0.99
clc = 0.1
start_epsilon = 0.3
end_epsilon = 0.1
start_reward_leadup = 50
end_reward_leadup = 5
batch_size = 40

input_dim = 33
shared_hidden0 = 64
shared_hidden1 = 128
shared_hidden2 = 64
actor_hidden = 32
critic_hidden = 32
output_dim_actor = 4
output_dim_critic = 1

model_params = {
    'input_dim': input_dim,
    'shared_hidden0': shared_hidden0,
    'shared_hidden1': shared_hidden1,
    'shared_hidden2': shared_hidden2,
    'critic_hidden': critic_hidden,
    'actor_hidden': actor_hidden,
    'output_dim_actor': output_dim_actor,
    'output_dim_critic': output_dim_critic
}

env = UnityEnvironment(file_name='Reacher_20.app')
# get the default brain
brain_name = env.brain_names[0]

# train model

losses = []
actor_losses = []
critic_losses = []
scores = []
ave_scores = []

params = {
    'env': env,
    'brain_name': brain_name,
    'start_epsilon': start_epsilon,
    'end_epsilon': end_epsilon,
    'epochs': epochs,
    'lr': lr,
    'gamma': gamma,
    'clc': clc,
    'start_reward_leadup': start_reward_leadup,
    'end_reward_leadup': end_reward_leadup,
    'batch_size': batch_size,
    'losses': losses,
    'scores': scores,
    'ave_scores': ave_scores,
    'actor_losses': actor_losses,
    'critic_losses': critic_losses
}

model = ActorCritic(model_params)
optimizer = torch.optim.Adam(lr=params['lr'], params=model.parameters())

start = perf_counter()

if __name__ == '__main__':
    try:
        worker(model, optimizer, params)
        save_model(model, optimizer, 'actor_critic.pt')
        plot_losses(params['losses'], 'loss.png')
        plot_losses(params['actor_losses'], filename='actor_loss.png', plotName="Actor Losses")
        plot_losses(params['critic_losses'], filename='critic_loss.png', plotName="Critic Losses")
        plot_scores(params['scores'], params['ave_scores'], filename='scores.png')
        end = perf_counter()
        print((end - start))
    except KeyboardInterrupt:
        pass
    finally:
        save_model(model, optimizer, 'actor_critic.pt')
        plot_losses(params['losses'], 'loss.png')
        plot_losses(params['actor_losses'], filename='actor_loss.png', plotName="Actor Losses")
        plot_losses(params['critic_losses'], filename='critic_loss.png', plotName="Critic Losses")
        plot_scores(params['scores'], params['ave_scores'], filename='scores.png')
        end = perf_counter()
        print((end - start))


