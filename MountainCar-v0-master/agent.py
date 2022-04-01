
import numpy as np
import random

from network import DQNetwork
from replay import ReplayBuffer

class DDQNAgent:

    def __init__(self, env, bufferSize=int(1e5), batchSize=64, gamma=0.99, tau=1e-3, lr=5e-4, callbacks=()):
        self.env = env
        self.env.seed(1024)
        self.batchSize = batchSize
        self.gamma = gamma
        self.tau = tau
        self.qTargets = 0.0
        self.numStates = env.observation_space.shape[0]
        self.numActions = env.action_space.n
        self.callbacks = callbacks

        layerSizes = [256, 256]
        batchNormPerLayer = [False, False]
        dropoutPerLayer = [0, 0]

        print("Initialising DDQN Agent with params : {}".format(self.__dict__))

        # Make local & target model
        self.localNetwork = DQNetwork(self.numStates, self.numActions,
                                       layerSizes=layerSizes,
                                       batchNormPerLayer=batchNormPerLayer,
                                       dropoutPerLayer=dropoutPerLayer,
                                       learningRate=lr)
        print("Finished initializing local network.")
        self.targetNetwork = DQNetwork(self.numStates, self.numActions,
                                        layerSizes=layerSizes,
                                        batchNormPerLayer=batchNormPerLayer,
                                        dropoutPerLayer=dropoutPerLayer,
                                        learningRate=lr)
        print("Finished initializing target network")
        self.memory = ReplayBuffer(bufferSize=bufferSize, batchSize=batchSize)

    def resetEpisode(self):
        state = self.env.reset()
        self.prevState = state
        return state

    def step(self, action, reward, nextState, done):
        self.memory.add(self.prevState, action, reward, nextState, done)

        if len(self.memory) > self.batchSize:
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma)

        self.prevState = nextState

    def act(self, state, eps=0.):
        state = np.reshape(state, [-1, self.numStates])
        action = self.localNetwork.model.predict(state)

        if random.random() > eps:
            return np.argmax(action)
        else:
            return random.choice(np.arange(self.numActions))

    def learn(self, experiences, gamma):
        states, actions, rewards, nextStates, dones = experiences

        for itr in range(len(states)):
            state, action, reward, nextState, done = states[itr], actions[itr], rewards[itr], nextStates[itr], dones[
                itr]
            state = np.reshape(state, [-1, self.numStates])
            nextState = np.reshape(nextState, [-1, self.numStates])

            self.qTargets = self.localNetwork.model.predict(state)
            if done:
                self.qTargets[0][action] = reward
            else:
                nextQ = self.targetNetwork.model.predict(nextState)[0]
                self.qTargets[0][action] = (reward + gamma * np.max(nextQ))

            self.localNetwork.model.fit(state, self.qTargets, epochs=1, verbose=0, callbacks=self.callbacks)

    def updateTargetModel(self):
        self.targetNetwork.model.set_weights(self.localNetwork.model.get_weights())
