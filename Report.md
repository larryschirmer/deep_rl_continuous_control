# Solved Model Training and Evaluation

The base code for this project comes from the cart pole implementation in Chapter 5 of [Deep Reinforcement Learning in Action](https://www.manning.com/books/deep-reinforcement-learning-in-action). The original implementation uses REINFORCE and a discrete distribution to predict which action (left or right) the cart should make to keep the pole upright. 

These two articles also helped shape my understanding of the Actor Critic algorithm

- [Intuitive RL: Intro to Advantage-Actor-Critic (A2C)](https://hackernoon.com/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752)
- [Maximum Entropy Policies in Reinforcement Learning & Everyday Life](https://medium.com/@awjuliani/maximum-entropy-policies-in-reinforcement-learning-everyday-life-f5a1cc18d32d)

In deep q-learning, a deep neural network takes the state of the environment and returns the expected value for all the actions in the environment's action space. The goal being for the agent to chose the right amount of arm motion to put its hand where the rewards are in the most efficient manor.

## Algorithms and Methods

The model makes these predictions using a feedforward neural network with two output tails:

```python
# helpers.py Line 23

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
```

The first tail outputs the mean of the distribution between -1 and 1. The agent uses this output to predict where best to position the arm. The mean range is handled by the `tanh` activation function. The second tail outputs the predicted value of the state. The agent uses this value to predict how valuable the action is at that time step. This is useful for understanding the difference between traveling between rewards and actions that result in reward.

The environment is reset at the start of each episode and new training loop begins until the environment outputs that one of the arm agents is done.

```python
# helpers.py Line 124

def run_episode(model, replay, params, epoch, train):

    env_info = params['env'].reset(train_mode=train)[params['brain_name']]
    state_ = env_info.vector_observations
    num_agents = len(env_info.agents)
    states = torch.from_numpy(state_).float()
    scores = np.zeros(num_agents)               # initialize the score

    values, logprobs, rewards, mean_entropy = [], [], [], torch.tensor(0.)
    done = False

    epsilon = np.clip((params['end_epsilon'] - params['start_epsilon']) / (params['epochs'] - 0) * epoch + params['start_epsilon'], params['end_epsilon'], params['start_epsilon'])
    step_count = 0
    while (done == False):
        step_count += 1
        actor_mean, value = model(states)
        actor_std = torch.tensor(epsilon)

        actor_mean = actor_mean.t()

        action_dist0 = torch.distributions.Normal(actor_mean[0], actor_std)
        action_dist1 = torch.distributions.Normal(actor_mean[1], actor_std)
        action_dist2 = torch.distributions.Normal(actor_mean[2], actor_std)
        action_dist3 = torch.distributions.Normal(actor_mean[3], actor_std)

        mean_entropy = action_dist0.entropy().mean()

        action0 = torch.clamp(action_dist0.sample(), min=-1, max=1)
        action1 = torch.clamp(action_dist1.sample(), min=-1, max=1)
        action2 = torch.clamp(action_dist2.sample(), min=-1, max=1)
        action3 = torch.clamp(action_dist3.sample(), min=-1, max=1)
        logprob0 = action_dist0.log_prob(action0)
        logprob1 = action_dist1.log_prob(action1)
        logprob2 = action_dist2.log_prob(action2)
        logprob3 = action_dist3.log_prob(action3)

        values.append(value.view(-1))
        logprobs.append([logprob0.view(-1), logprob1.view(-1), logprob2.view(-1), logprob3.view(-1)])

        action_list = [action0.detach().numpy().squeeze(), action1.detach().numpy().squeeze(), action2.detach().numpy().squeeze(), action3.detach().numpy().squeeze()]
        action_list = np.stack(action_list, axis=1)
        # send all actions to the environment
        env_info = params['env'].step(action_list)[params['brain_name']]
        # get next state (for each agent)
        state_ = env_info.vector_observations
        # get reward (for each agent)
        reward = env_info.rewards
        # see if episode finished
        done = env_info.local_done[0]

        states = torch.from_numpy(state_).float()
        rewards.append(reward)
        scores += np.array(reward)


    # Update replay buffer for each agent
    stacked_logprob0 = torch.stack([a[0] for a in logprobs], dim=1)
    stacked_logprob1 = torch.stack([a[1] for a in logprobs], dim=1)
    stacked_logprob2 = torch.stack([a[2] for a in logprobs], dim=1)
    stacked_logprob3 = torch.stack([a[3] for a in logprobs], dim=1)

    stacked_values = torch.stack(values, dim=1)
    stacked_rewards = np.stack(rewards, axis=1)

    for agent_index in range(len(env_info.agents)):
  
        agent_values = stacked_values[agent_index]
        agent_logprobs = [stacked_logprob0[agent_index], stacked_logprob1[agent_index], stacked_logprob2[agent_index], stacked_logprob3[agent_index]]
        agent_rewards = stacked_rewards[agent_index]

        actor_losses, critic_losses, losses, reward_leadup = get_trjectory_loss(agent_values, agent_logprobs, agent_rewards, mean_entropy, epoch, params)
        replay.append((scores[agent_index], actor_losses, critic_losses, losses))

    return scores, epsilon, reward_leadup
```

At each training step, the agent converts the output mean from the model into a Normal Distribution with a standard distribution that is annealed over the course of training (epsilon). By annealing the value of the standard distribution, the entropy of the distribution starts off high while the accuracy of the model is low and slowly narrows the distribution as accuracy improves.

Information is collected at each trajectory and used to calculate the loss from that episode.

```python
# helpers.py Line 247

def get_trjectory_loss(values, logprobs, rewards, mean_entropy, epoch, params):

    reward_leadup = np.clip((params['end_reward_leadup'] - params['start_reward_leadup']) / (params['epochs'] - 0) * epoch + params['start_reward_leadup'], params['end_reward_leadup'], params['start_reward_leadup'])

    [logprob0, logprob1, logprob2, logprob3] = logprobs

    values = values.flip(dims=(0,))
    rewards = torch.Tensor(rewards).flip(dims=(0,))
    logprob0 = logprob0.flip(dims=(0,))
    logprob1 = logprob1.flip(dims=(0,))
    logprob2 = logprob2.flip(dims=(0,))
    logprob3 = logprob3.flip(dims=(0,))

    Returns = []
    total_return = torch.Tensor([0])
    leadup = 0

    for reward_index in range(len(rewards)):
        if rewards[reward_index].item() > 0:
            leadup = reward_leadup
        if leadup == 0:
            total_return = torch.Tensor([0])
        
        total_return = rewards[reward_index] + total_return * params['gamma']
        Returns.append(total_return)
        leadup = leadup - 1 if leadup > 0 else 0

    Returns = torch.stack(Returns).view(-1)
    Returns = F.normalize(Returns, dim=0)

    actor_loss0 = -logprob0 * (Returns - values.detach())
    actor_loss1 = -logprob1 * (Returns - values.detach())
    actor_loss2 = -logprob2 * (Returns - values.detach())
    actor_loss3 = -logprob3 * (Returns - values.detach())

    critic_loss = torch.pow(values - Returns, 2)

    actor_loss0 = actor_loss0.sum()
    actor_loss1 = actor_loss1.sum()
    actor_loss2 = actor_loss2.sum()
    actor_loss3 = actor_loss3.sum()

    critic_loss = critic_loss.sum()

    loss0 = actor_loss0 + params['clc']*critic_loss + 0.01 * mean_entropy
    loss1 = actor_loss1 + params['clc']*critic_loss + 0.01 * mean_entropy
    loss2 = actor_loss2 + params['clc']*critic_loss + 0.01 * mean_entropy
    loss3 = actor_loss3 + params['clc']*critic_loss + 0.01 * mean_entropy

    actor_losses = (actor_loss0, actor_loss1, actor_loss2, actor_loss3)
    losses = (loss0, loss1, loss2, loss3)

    return actor_losses, critic_loss, losses, reward_leadup
```

The loss of each trajectory is calculated using the general [REINFORCE formula](https://pytorch.org/docs/stable/distributions.html#score-function). The most notable change to this implementation though is how rewards are discounted. 

```python
Returns = []
total_return = torch.Tensor([0])
leadup = 0

for reward_index in range(len(rewards)):
    if rewards[reward_index].item() > 0:
        leadup = reward_leadup
    if leadup == 0:
        total_return = torch.Tensor([0])
    
    total_return = rewards[reward_index] + total_return * params['gamma']
    Returns.append(total_return)
    leadup = leadup - 1 if leadup > 0 else 0
```

I understand concept of GAE (Generalized Advantage Estimation). This technique changes how far ahead the agent can look ahead in terms of future reward. I, however, had issues implementing the technique so I ended up adding a simpler variation of the technique I called `Reward Leadup`. What Reward Leadup does is only reward actions closer to where they happen. This solves the issue where random motion hundreds of timesteps away from when a reward occurs confuses the model into thinking that random usless motion somehow contributed to that reward occuring.

The trick is that after the rewards are reversed, if there is not a reward during the leadup window then the reward is not cumulative. The result of this change makes future reward (from the perspective of the agent) much higher right when rewards occurs, then reinforcing the actions that lead to the reward.

This technique also has the added benefit of having the ability of annealing the leadup window. A larger annealing leadup window at the beginning of training catches the random motion early on that results in reward, and at the end of training a smaller reward leadup rewards the small corrective motion that keeps the arm in the target bubble.

Losses from each trajectory are batched together and the model is updated using the average of the loss across the batch. This is done to reduce to the variability in training data and smooth loss.

```python
# helpers.py Line 202

def update_params(replay, optimizer, params):
    loss0 = torch.tensor(0.)
    loss1 = torch.tensor(0.)
    loss2 = torch.tensor(0.)
    loss3 = torch.tensor(0.)
    actor_loss0 = torch.tensor(0.)
    actor_loss1 = torch.tensor(0.)
    actor_loss2 = torch.tensor(0.)
    actor_loss3 = torch.tensor(0.)
    critic_loss = torch.tensor(0.)

    for trajectory in replay:
        rewards_sum, actor_losses, critic_loss, losses = trajectory
        loss0 += losses[0]
        loss1 += losses[1]
        loss2 += losses[2]
        loss3 += losses[3]
        actor_loss0 += actor_losses[0]
        actor_loss1 += actor_losses[1]
        actor_loss2 += actor_losses[2]
        actor_loss3 += actor_losses[3]
        critic_loss += critic_loss
    

    loss0 = loss0 / len(replay)
    loss1 = loss1 / len(replay)
    loss2 = loss2 / len(replay)
    loss3 = loss3 / len(replay)
    actor_loss0 = actor_loss0 / len(replay)
    actor_loss1 = actor_loss1 / len(replay)
    actor_loss2 = actor_loss2 / len(replay)
    actor_loss3 = actor_loss3 / len(replay)
    critic_loss = critic_loss / len(replay)

    loss_mean = (loss0 + loss1 + loss2 + loss3) / 4

    optimizer.zero_grad()
    loss_mean.backward()
    optimizer.step()

    actor_loss_sum = actor_loss0 + actor_loss1 + actor_loss2 + actor_loss3

    return loss_mean, actor_loss_sum, critic_loss
```

## Training Evaluation

The model was trained for **25490 episodes** and achieved a moving average reward over 100 episodes of +30 for all 20 agents 5 times. The model was set to go for 40000 epochs but stopped early after achieving target accuracy.

```python
# early stopping loop break
# helpers.py Line 95

if train and len(early_stop_captures) >= early_stop_threshold:
    print("stopped early because net has reached target score")
    print(early_stop_captures)
    break
```

**hyperparameters**
```python
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
```

**Average Score and Scores each Episode**
![](https://github.com/larryschirmer/deep_rl_continuous_control/raw/macbook/scores.png)

- Black: average of the moving average of each agent at each episode
    - Each episode an average for each agent over 100 episodes is calculated. To represent these average scores, they are averaged together into a single value and plotted at the black line. This single average value should NOT be confused for the target accuracy of the model which is an average value of +30 for every agent, and not the average of the average of the agents.

**Model Training Loss**
![](https://github.com/larryschirmer/deep_rl_continuous_control/raw/macbook/loss.png)

**Actor Loss**
![](https://github.com/larryschirmer/deep_rl_continuous_control/raw/macbook/actor_loss.png)

**Critic Loss**
![](https://github.com/larryschirmer/deep_rl_continuous_control/raw/macbook/actor_loss.png)