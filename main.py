# dropped epsilon limit, turned up learning rate
from unityagents import UnityEnvironment
from time import perf_counter
import pandas as pd
import copy

from model import ActorCritic, loss_fn
from helpers import  plot_losses, plot_scores, save_model, worker

# hyperparameters
lr = 0.01
gamma = 0.99
gae = 0.9
clc = 0.1
step_update = 100
ppo_epsilon = 0.2

input_dim = 33
shared_hidden0 = 50
shared_hidden1 = 150
shared_hidden2 = 150
critic_hidden = 75
output_dim_actor = 2
output_dim_critic = 1

model = ActorCritic(
    input_dim, shared_hidden0, shared_hidden1, shared_hidden2, critic_hidden, output_dim_actor, output_dim_critic)

env = UnityEnvironment(file_name='Reacher.app')
# get the default brain
brain_name = env.brain_names[0]

# train model

epochs = 2000
epsilon = 1.0  # decays over the course of training
losses = []
actor_losses = []
critic_losses = []
scores = []

params = {
    'env': env,
    'brain_name': brain_name,
    'epochs': epochs,
    'lr': lr,
    'step_update': step_update,
    'gamma': gamma,
    'gae': gae,
    'ppo_epsilon': ppo_epsilon,
    'clc': clc,
    'losses': losses,
    'scores': scores,
    'actor_losses': actor_losses,
    'critic_losses': critic_losses
}

start = perf_counter()
worker(model, params)
save_model(model, 'actor_critic.pt')
end = perf_counter()
print((end - start))

rolling_window = 50

ave_loss = pd.Series(losses).rolling(rolling_window).mean()
plot_losses(losses, 'ave_loss-{}.png'.format(epochs))

plot_scores(scores, 'scores-{}.png'.format(epochs))

ave_scores = pd.Series(scores).rolling(50*2).mean()
plot_scores(ave_scores, 'ave_scores-{}.png'.format(epochs),
            plotName='Ave Score')
