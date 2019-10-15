from collections import deque

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from dqn_agent import Agent

is_ipython = 'inline' in plt.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

if __name__ == "__main__":

    real_env = gym.make('LunarLander-v2')
    sim_env = gym.make('noisy-lander-v0', max_skew=0.25, seed=11)
    real_env.seed(10)
    sim_env.seed(111)
    print('State shape: ', real_env.observation_space.shape)
    print('Number of actions: ', sim_env.action_space.n)

    agent = Agent(state_size=8, action_size=4, seed=0)


    def train(gym_env, n_episodes=600, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        """Deep Q-Learning.

        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        """
        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = eps_start  # initialize epsilon
        for i_episode in range(1, n_episodes + 1):
            state = gym_env.reset()
            score = 0
            for t in range(max_t):
                action = agent.act(state, eps)
                next_state, reward, done, _ = gym_env.step(action)
                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            scores_window.append(score)  # save most recent score
            scores.append(score)  # save most recent score
            eps = max(eps_end, eps_decay * eps)  # decrease epsilon
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if np.mean(scores_window) >= 150.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                             np.mean(scores_window)))
                torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_Dueling_DDQN.pth')
                break
        return scores


    run_training = True
    if run_training:
        scores = train(gym_env=sim_env)
        #
        # plot the scores
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()

    # Run the agent in the real environment
    agent = Agent(state_size=8, action_size=4, seed=0)
    # load the weights from file
    agent.qnetwork_local.load_state_dict(
        torch.load('checkpoint_Dueling_DDQN.pth', map_location=lambda storage, loc: storage))
    total_rewards = []

    display_games = False
    print("Running agent on REAL")
    for i in range(20):
        state = real_env.reset()
        if display_games:
            img = plt.imshow(real_env.render(mode='rgb_array'))
        tot = 0
        for j in range(250):
            action = agent.act(state)
            if display_games:
                img.set_data(real_env.render(mode='rgb_array'))
            plt.axis('off')
            if is_ipython:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            state, reward, done, _ = real_env.step(action)
            tot += reward
            if done:
                break
        total_rewards.append(tot)
    real_env.close()
    print("total rewards: ", total_rewards)
    print("Average reward and std: ", np.mean(total_rewards), np.std(total_rewards))

    # Run the agent in the perturbed environment
    print("Running agent on Sim")
    agent = Agent(state_size=8, action_size=4, seed=0)
    # load the weights from file
    agent.qnetwork_local.load_state_dict(
        torch.load('checkpoint_Dueling_DDQN.pth', map_location=lambda storage, loc: storage))
    total_rewards = []

    for i in range(20):
        state = sim_env.reset()

        if display_games:
            img = plt.imshow(sim_env.render(mode='rgb_array'))
        tot = 0
        for j in range(250):
            action = agent.act(state)
            if display_games:
                img.set_data(sim_env.render(mode='rgb_array'))
            plt.axis('off')
            if is_ipython:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            state, reward, done, _ = sim_env.step(action)
            tot += reward
            if done:
                break
        total_rewards.append(tot)
    sim_env.close()
    print("total rewards: ", total_rewards)
    print("Average reward and std: ", np.mean(total_rewards), np.std(total_rewards))
