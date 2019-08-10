import torch
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt


def plot_losses(losses, filename='', plotName='Loss', show=False):
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(len(losses)), losses)
    plt.ylabel(plotName)
    plt.xlabel("Training Steps")
    if show:
        plt.show()

    if (filename):
        plt.savefig(filename)


def plot_durations(durations, filename='', plotName='Duration', show=False):
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(len(durations)), durations)
    plt.ylabel(plotName)
    plt.xlabel('Episode #')
    if show:
        plt.show()

    if (filename):
        plt.savefig(filename)


def plot_scores(scores, filename='', plotName='Score', show=False):
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel(plotName)
    plt.xlabel('Episode #')
    if show:
        plt.show()

    if (filename):
        plt.savefig(filename)


def save_model(models, filename):

    for index, model in enumerate(models):
        state = {
            'state_dict': model.state_dict(),
        }
        torch.save(state, '{}-{}'.format(index, filename))


def load_model(model, filename, evalMode=True):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])

    if evalMode:
        model.eval()
    else:
        model.train()

    return model


def worker(model, params, train=True):     # reset the environment
    optimizer0 = torch.optim.Adam(
        lr=params['lr'], params=model[0].parameters())
    optimizer1 = torch.optim.Adam(
        lr=params['lr'], params=model[1].parameters())
    optimizer2 = torch.optim.Adam(
        lr=params['lr'], params=model[2].parameters())
    optimizer3 = torch.optim.Adam(
        lr=params['lr'], params=model[3].parameters())
    optimizers = [optimizer0, optimizer1, optimizer2, optimizer3]

    highest_score = 0
    for epoch in range(params['epochs']):
        values, actions, rewards, final_score = run_episode(
            model, optimizers, params, train)

        if train and final_score >= highest_score:
            highest_score = final_score
            save_model(model, 'actor_critic_checkpoint@highest.pt')

        if train:
            loss, actor_loss, critic_loss = update_params(
                optimizers, values, actions, rewards, params)

            params['losses'].append(loss.item())
            params['scores'].append(final_score)
            params['actor_losses'].append(actor_loss.item())
            params['critic_losses'].append(critic_loss.item())

            average_score = 0. if len(params['scores']) < 100 else np.average(
                params['scores'][-100:])
            if epoch % 10 == 0:
                print("Epoch: {}, Loss: {:.7f}, Ave Score: {:.4f}, highest: {:.4f}".format(
                    epoch, loss, average_score, highest_score))


def run_episode(model, optimizers, params, train):

    env_info = params['env'].reset(train_mode=train)[params['brain_name']]
    state_ = env_info.vector_observations[0]        # get the current state
    state = torch.from_numpy(state_).float()
    score = np.zeros(1)                            # initialize the score

    values, actions, rewards = [], [], []
    done = False

    loss, actor_loss, critic_loss = (
        torch.tensor(0), torch.tensor(0), torch.tensor(0))

    step_count = 0
    while (done == False):
        step_count += 1
        policies0_mean, policies0_std, value0 = model[0](state)
        policies1_mean, policies1_std, value1 = model[1](state)
        policies2_mean, policies2_std, value2 = model[2](state)
        policies3_mean, policies3_std, value3 = model[3](state)

        action_dist0 = torch.distributions.Normal(
            policies0_mean, policies0_std)
        action_dist1 = torch.distributions.Normal(
            policies1_mean, policies1_std)
        action_dist2 = torch.distributions.Normal(
            policies2_mean, policies2_std)
        action_dist3 = torch.distributions.Normal(
            policies3_mean, policies3_std)
        action0 = action_dist0.sample()
        action1 = action_dist1.sample()
        action2 = action_dist2.sample()
        action3 = action_dist3.sample()
        entropy0 = action_dist0.entropy()
        entropy1 = action_dist1.entropy()
        entropy2 = action_dist2.entropy()
        entropy3 = action_dist3.entropy()

        values.append([value0, value1, value2, value3])
        actions.append([entropy0, entropy1, entropy2, entropy3])

        action_list = np.array([action0.detach().numpy(), action1.detach(
        ).numpy(), action2.detach().numpy(), action3.detach().numpy()])
        # send all actions to tne environment
        env_info = params['env'].step(action_list)[params['brain_name']]
        # get next state (for each agent)
        state_ = env_info.vector_observations[0]
        # get reward (for each agent)
        reward = env_info.rewards[0]
        # see if episode finished
        done = env_info.local_done[0]

        state = torch.from_numpy(state_).float()
        rewards.append(reward)

    return values, actions, rewards, sum(rewards)


def update_params(optimizers, values, actions, rewards, params):
    actions0 = [a[0] for a in actions]
    actions1 = [a[1] for a in actions]
    actions2 = [a[2] for a in actions]
    actions3 = [a[3] for a in actions]

    values0 = [v[0] for v in values]
    values1 = [v[1] for v in values]
    values2 = [v[2] for v in values]
    values3 = [v[3] for v in values]

    rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1)
    actions0 = torch.stack(actions0).flip(dims=(0,)).view(-1)
    actions1 = torch.stack(actions1).flip(dims=(0,)).view(-1)
    actions2 = torch.stack(actions2).flip(dims=(0,)).view(-1)
    actions3 = torch.stack(actions3).flip(dims=(0,)).view(-1)
    values0 = torch.stack(values0).flip(dims=(0,)).view(-1)
    values1 = torch.stack(values1).flip(dims=(0,)).view(-1)
    values2 = torch.stack(values2).flip(dims=(0,)).view(-1)
    values3 = torch.stack(values3).flip(dims=(0,)).view(-1)

    Returns = []
    total_return = torch.Tensor([0])

    for reward_index in range(len(rewards)):
        total_return = rewards[reward_index] + total_return * params['gamma']
        Returns.append(total_return)

    Returns = torch.stack(Returns).view(-1)
    Returns = F.normalize(Returns, dim=0)

    actor_loss0 = -1*actions0 * (Returns - values0.detach())
    actor_loss1 = -1*actions1 * (Returns - values1.detach())
    actor_loss2 = -1*actions2 * (Returns - values2.detach())
    actor_loss3 = -1*actions3 * (Returns - values3.detach())

    critic_loss0 = torch.pow(values0 - Returns, 2)
    critic_loss1 = torch.pow(values1 - Returns, 2)
    critic_loss2 = torch.pow(values2 - Returns, 2)
    critic_loss3 = torch.pow(values3 - Returns, 2)

    loss0 = actor_loss0.mean() + params['clc']*critic_loss0.mean()
    loss1 = actor_loss1.mean() + params['clc']*critic_loss1.mean()
    loss2 = actor_loss2.mean() + params['clc']*critic_loss2.mean()
    loss3 = actor_loss3.mean() + params['clc']*critic_loss3.mean()

    optimizers[0].zero_grad()
    optimizers[1].zero_grad()
    optimizers[2].zero_grad()
    optimizers[3].zero_grad()

    loss0.backward()
    loss1.backward()
    loss2.backward()
    loss3.backward()

    optimizers[0].step()
    optimizers[1].step()
    optimizers[2].step()
    optimizers[3].step()

    loss_sum = loss0 + loss1 + loss2 + loss3
    actor_loss_sum = actor_loss0.mean() + actor_loss1.mean() + \
        actor_loss2.mean() + actor_loss3.mean()
    critic_loss_sum = critic_loss0.mean() + critic_loss1.mean() + \
        critic_loss2.mean() + critic_loss3.mean()

    return loss_sum, actor_loss_sum, critic_loss_sum
