# dropped epsilon limit, turned up learning rate
from unityagents import UnityEnvironment
from time import perf_counter
import pandas as pd
import copy

from model import ActorCritic
from helpers import plot_losses, plot_scores, save_model, worker

# hyperparameters
epochs = 2000
lr = 0.0001
gamma = 0.999
clc = 0.1
start_epsilon = 0.8
end_epsilon = 0.2

input_dim = 33
shared_hidden0 = 128
shared_hidden1 = 256
shared_hidden2 = 128
actor_hidden = 62
critic_hidden = 62
output_dim_actor = 2
output_dim_critic = 1

model_params = {
    'start_epsilon': start_epsilon,
    'end_epsilon': end_epsilon,
    'epochs': epochs,
    'input_dim': input_dim,
    'shared_hidden0': shared_hidden0,
    'shared_hidden1': shared_hidden1,
    'shared_hidden2': shared_hidden2,
    'critic_hidden': critic_hidden,
    'actor_hidden': actor_hidden,
    'output_dim_actor': output_dim_actor,
    'output_dim_critic': output_dim_critic
}

model0 = ActorCritic(model_params)
model1 = ActorCritic(model_params)
model2 = ActorCritic(model_params)
model3 = ActorCritic(model_params)

model = [model0, model1, model2, model3]

env = UnityEnvironment(file_name='Reacher.app')
# get the default brain
brain_name = env.brain_names[0]

# train model

losses = []
actor_losses = []
critic_losses = []
scores = []

params = {
    'env': env,
    'brain_name': brain_name,
    'epochs': epochs,
    'lr': lr,
    'gamma': gamma,
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
