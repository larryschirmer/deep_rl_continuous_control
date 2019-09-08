# Continuous Control with Deep Reinforcement Learning

<img src="https://github.com/larryschirmer/deep_rl_continuous_control/blob/master/solved_continuous_control.gif" alt="solved continuous control" width="600"/>

## What is this Project

A deep reinforcement learning implementation of [Unity ML Agents Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment. The Reacher environment generates an arm with a 3 degree-of-freedom shoulder and elbow. The goal of the arm is to bend the shoulder and elbow so that the fist of the arm is inside a bubble that is floating around the arm anchor. This project uses a feedforward network with a natural distribution output for each control. The agent learns to associate the mean of the distribution with the correct output for each state.

## How to Install

Because one of the project dependencies requires a specific version of tensorflow only available in python 3.5 and earlier, its easiest to use conda to build the environment for this project.

Run the following command to build this environment using the same dependancies I used:

```bash
conda env create -f environment.yml python=3.5
```

[See the conda docs for installation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

As for the game engine, select the environment that matches your operating system:

**Reacher with one arm**

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

**Reacher with twenty arms**

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

Then, place the file in the project folder of this repository, and unzip (or decompress) the file.

## Getting Started

After installing this project's dependencies, launching `main.py` will begin training the model.

```bash
python main.py
```

This file launches the unity environment with twenty arms and begin posting to the console training updates every 100 epochs. In its current configuration, this model takes about **36 hours** to train on a 6-core macmini.

If you would like to run the solved model checkpoint that I have provided, launch a jupyter notebook environment:

```bash
jupyter notebook
```

and open `Reacher.ipynb`

## Problem/Solution

Control environment output needs come in two types: discrete and continuous. Discrete environments such as grid world require an agent to make decision from an know set of actions (up, down, left, right). For control environments such as driving or manufacturing equipment, it is necessary to have continuous outputs. These controls have a range where they can act and an agent needs to be able to serve any value in the range. 

For example, in an autonomous driving car, both discrete and continuous agents could be applied. A discrete agent would be able to turn the wheels all the way to the left or right (or some other amount the model was trained to output). However, a continuous agent can turn the wheels at any amount from left to right. With continuous control deep learning, an agent is trained to output the correct amount of turning motion with some amount of standard deviation.

## Important Files

- This `README.md`: describes the project and its files in detail
- `Reacher.ipynb`: working demonstration of the model loaded after reaching training target
- `Report.md`: document containing algorithms and methods, plots of training, and a discussion of future work
- `actor_critic.pt`: trained model checkpoint
- `main.py`: python file used to develop and train network
- `helpers.py`: collection of functions used to train, test, and monitor model
- `model.py`: functions to return new model, optimizer, and loss function
- `scores.png`: a plot of scores and average scores from the 20 agents averaged together from each episode
    - light gray: score from each episode. Example for two agents, `[ 12.1, 13.6 ] -> 12.85`
    - black: Average over 100 episodes. Example for two agent, `[ 12.7, 12.9] -> 12.8`
- `losses.png`: model training loss over all of training


## The Environment

A double-jointed arm is tasked to move its hand into target location. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 

## Help

Contributions, issues and feature requests are welcome.
Feel free to check [issues page](https://github.com/larryschirmer/deep_rl_continuous_control/issues) if you want to contribute.

## Author

- Larry Schirmer https://github.com/larryschirmer