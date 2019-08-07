import torch
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
from model import loss_fn


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


def save_model(model, filename):
    state = {
        'state_dict': model.state_dict(),
    }
    torch.save(state, filename)


def load_model(model, filename, evalMode=True):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])

    if evalMode:
        model.eval()
    else:
        model.train()

    return model


def worker(model, params, train=True):     # reset the environment
    optimizer = torch.optim.Adam(lr=params['lr'], params=model.parameters())

    highest_score = 0
    for epoch in range(params['epochs']):
        values, actions, rewards, final_score = run_episode(
            model, optimizer, params, train)

        if train and final_score >= highest_score:
            highest_score = final_score
            save_model(model, 'actor_critic_checkpoint@highest.pt')

        if train:
            loss, actor_loss, critic_loss = update_params(
                optimizer, values, actions, rewards, params)

            params['losses'].append(loss.item())
            params['scores'].append(final_score)
            params['actor_losses'].append(actor_loss.item())
            params['critic_losses'].append(critic_loss.item())

            average_score = 0. if len(params['scores']) < 100 else np.average(
                params['scores'][-100:])
            if epoch % 10 == 0:
                print("Epoch: {}, Loss: {:.7f}, Ave Score: {:.4f}, highest: {:.4f}".format(
                    epoch, loss, average_score, highest_score))


def run_episode(model, optimizer, params, train):

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
        policies, value = model(state)

        action_dist0 = torch.distributions.Normal(
            policies[0][0], policies[0][1])
        action_dist1 = torch.distributions.Normal(
            policies[1][0], policies[1][1])
        action_dist2 = torch.distributions.Normal(
            policies[2][0], policies[2][1])
        action_dist3 = torch.distributions.Normal(
            policies[3][0], policies[3][1])
        action0 = action_dist0.sample()
        action1 = action_dist1.sample()
        action2 = action_dist2.sample()
        action3 = action_dist3.sample()

        values.append(value)
        actions.append([action0, action1, action2, action3])

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

        if done:
            params['env'].reset(train_mode=train)[params['brain_name']]

        rewards.append(reward)

        # if train and step_count % params['step_update'] == 0:
        #     update_params(optimizer, values, actions,
        #                   rewards, params, mid_update=True)

    return values, actions, rewards, sum(rewards)


prev_actions0 = torch.Tensor([0])
prev_actions1 = torch.Tensor([0])
prev_actions2 = torch.Tensor([0])
prev_actions3 = torch.Tensor([0])


def update_params(optimizer, values, actions, rewards, params, mid_update=False):
    global prev_actions0
    global prev_actions1
    global prev_actions2
    global prev_actions3

    actions0 = [a[0] for a in actions]
    actions1 = [a[1] for a in actions]
    actions2 = [a[2] for a in actions]
    actions3 = [a[3] for a in actions]

    rewards = torch.Tensor(rewards).view(-1)
    actions0 = torch.stack(actions0).view(-1)
    actions1 = torch.stack(actions1).view(-1)
    actions2 = torch.stack(actions2).view(-1)
    actions3 = torch.stack(actions3).view(-1)
    values = torch.stack(values).view(-1)
    Returns = []
    total_return = torch.Tensor([0])

    if mid_update:
        rewards = rewards[-params['step_update']:]
        actions0 = actions0[-params['step_update']:]
        actions1 = actions1[-params['step_update']:]
        actions2 = actions2[-params['step_update']:]
        actions3 = actions3[-params['step_update']:]
        values = values[-params['step_update']:]

    for reward_index in range(len(rewards)):
        total_return = rewards[reward_index] + total_return # * params['gamma']
        Returns.append(total_return)

    gae_reduction = torch.Tensor(
        [(1 - params['gae']) * params['gae'] ** i for i in range(len(Returns))]).flip(dims=(0,))
    gae_reduction = gae_reduction if not mid_update else 1
    Returns = torch.stack(Returns).view(-1)
    Returns = F.normalize(Returns, dim=0)

    ppo_ratio0 = (actions0 - prev_actions0[-1:]).exp()
    torch.cat((prev_actions0[1:], actions0))
    advantage = Returns - values.detach()
    surrogate00 = ppo_ratio0 * advantage
    surrogate01 = torch.clamp(
        ppo_ratio0, 1.0 - params['ppo_epsilon'], 1.0 + params['ppo_epsilon']) * advantage

    ppo_ratio1 = (actions1 - prev_actions1[-1:]).exp()
    torch.cat((prev_actions1[1:], actions1))
    advantage = Returns - values.detach()
    surrogate10 = ppo_ratio1 * advantage
    surrogate11 = torch.clamp(
        ppo_ratio1, 1.0 - params['ppo_epsilon'], 1.0 + params['ppo_epsilon']) * advantage

    ppo_ratio2 = (actions2 - prev_actions2[-1:]).exp()
    torch.cat((prev_actions2[1:], actions2))
    advantage = Returns - values.detach()
    surrogate20 = ppo_ratio2 * advantage
    surrogate21 = torch.clamp(
        ppo_ratio2, 1.0 - params['ppo_epsilon'], 1.0 + params['ppo_epsilon']) * advantage

    ppo_ratio3 = (actions3 - prev_actions3[-1:]).exp()
    torch.cat((prev_actions3[1:], actions3))
    advantage = Returns - values.detach()
    surrogate30 = ppo_ratio3 * advantage
    surrogate31 = torch.clamp(
        ppo_ratio3, 1.0 - params['ppo_epsilon'], 1.0 + params['ppo_epsilon']) * advantage

    actor_loss0 = - torch.min(surrogate00, surrogate01) * gae_reduction
    actor_loss1 = - torch.min(surrogate10, surrogate11) * gae_reduction
    actor_loss2 = - torch.min(surrogate20, surrogate21) * gae_reduction
    actor_loss3 = - torch.min(surrogate30, surrogate31) * gae_reduction
    critic_loss = torch.pow(values - (Returns * gae_reduction), 2)

    actor_loss_mean = (actor_loss0.sum() + actor_loss1.sum() + actor_loss2.sum() + actor_loss3.sum()) / 4
    loss = loss_fn(actor_loss_mean + params['clc']*critic_loss.mean(), sum(rewards))
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

    return loss, actor_loss_mean, critic_loss.mean()
