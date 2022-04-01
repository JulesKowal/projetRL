
import gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from time import time
import matplotlib.animation as animation

from agent import DDQNAgent

env_name = None
initial_timestamp = 0.0
np.random.seed(1024)

def train(numEpisodes=2000, startEpsilon=1.0, endEpsilon=0.001, epsDecayRate=0.9, targetReward=1000):
    scores = []
    scoresWindow = deque(maxlen=100)
    eps = startEpsilon
    print("Starting model training for {} episodes.".format(numEpisodes))
    avgScoreGreaterThanTargetCounter = 0
    for episode in range(1, numEpisodes + 1):
        initialTime = time()
        state = agent.resetEpisode()
        score = 0
        done = False
        while not done:
            action = agent.act(state, eps)
            nextState, reward, done, _ = env.step(action)
            agent.step(action, reward, nextState, done)
            state = nextState
            score += reward
            if done:
                agent.updateTargetModel()
                break
        timeTaken = time() - initialTime
        scoresWindow.append(score)
        scores.append(score)
        eps = max(endEpsilon, epsDecayRate * eps)
        print('Episode {}\tTime Taken: {:.2f} sec\tScore: {:.2f}\tState: {}\tAverage Q-Target: {:.4f}'
                     '\tEpsilon: {:.3f}\tAverage Score: {:.2f}\t'.format(
            episode, timeTaken, score, state[0], np.mean(agent.qTargets), eps, np.mean(scoresWindow)))
        if episode % 100 == 0:
            print(
                'Episode {}\tTime Taken: {:.2f} sec\tScore: {:.2f}\tState: {}\tAverage Q-Target: {:.4f}\tAverage Score: {:.2f}'.format(
                    episode, timeTaken, score, state[0], np.mean(agent.qTargets), np.mean(scoresWindow)))
            agent.localNetwork.model.save('save/{}_local_model_{}.h5'.format(envName, initialTime))
            agent.targetNetwork.model.save('save/{}_target_model_{}.h5'.format(envName, initialTime))
        if np.mean(scoresWindow) >= targetReward:
            avgScoreGreaterThanTargetCounter += 1
            if avgScoreGreaterThanTargetCounter >= 5:
                print("Model training finished! \nAverage Score over last 100 episodes: {}\tNumber of Episodes: {}".format(
                    np.mean(scoresWindow), episode))
                return scores
        else:
            avgScoreGreaterThanTargetCounter = 0
    print("Model training finished! \nAverage Score over last 100 episodes: {}\tNumber of Episodes: {}".format(
        np.mean(scoresWindow), numEpisodes))
    return scores


def play_model(actor, renderEnv=False, shouldReturnImages=False):
    state = env.reset()
    score = 0
    done = False
    images = []

    while not done:
        if renderEnv:
            if shouldReturnImages:
                images.append(env.render("rgb_array"))
            else:
                env.render()
        state = np.reshape(state, [-1, env.observation_space.shape[0]])
        action = actor.predict(state)
        nextState, reward, done, _ = env.step(np.argmax(action))
        state = nextState
        score += reward
        if done:
            return score, images
    return 0, images

def displayFramesAsGif(frames):
    """
    Displays a list of frames as a gif, with controls
    """
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    display(display_animation(anim, default_mode='loop'))

#train
envName = "MountainCar-v0"
env = gym.make(envName)
agent = DDQNAgent(env, bufferSize=100000, gamma=0.99, batchSize=64, lr=0.0001, callbacks=[])
scores = train(numEpisodes=2000, targetReward=-110, epsDecayRate=0.9)
