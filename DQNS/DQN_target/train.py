from environment import Environment
import torch
from agent import Agent
from environment import Environment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import time

env = Environment()

agent = Agent(state_size=2, action_size=4, seed=0)


def dqn(n_episodes=100_000, max_t=1500, eps_start=1.0, eps_end=0.01, eps_decay=0.999936):
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    max_score = 300.0
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            # env.render()
            # time.sleep(15)

            action = agent.act(state, eps)
            next_state, reward, done = env.step(action)
            agent.step(state, action, reward, next_state, done)
            env.render()
            time.sleep(0.5)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(
            i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_window)))

        if i_episode % 10_000 == 0:
            torch.save(agent.qnetwork_local.state_dict(),
                       'eps'+str(i_episode) + 'checkpoint.pth')

        if np.mean(scores_window) >= max_score:
            max_score = np.mean(scores_window)
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(),
                       'checkpoint.pth')
    return scores


scores = dqn()

scores_dqn_np = np.array(scores)
np.savetxt("scores.txt", scores_dqn_np)


def moving_average(a, n=100):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


scores_ma_dqn = moving_average(scores, n=100)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores_ma_dqn)), scores_ma_dqn)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
