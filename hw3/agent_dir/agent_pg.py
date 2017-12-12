from agent_dir.agent import Agent
from collections import deque
from keras.models import Sequential
from keras.layers import Reshape, Lambda, Dense, Activation, GRU, TimeDistributed, Conv1D, MaxPooling1D, MaxPooling2D, Flatten, Dropout, SimpleRNN, Bidirectional, Convolution2D, Permute, BatchNormalization, RepeatVector, Input, MaxPool2D 
from keras.optimizers import Adam
from keras.models import load_model, Model
from keras.layers import merge, Concatenate, Merge, add, Masking, Activation, dot, multiply, concatenate
from keras import backend as K
import tensorflow as tf 
import numpy as np
import random
import time
class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        self.action_size = 3
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.model = self._build_model()
        self.model.summary()
        self.record = deque(maxlen=30)
        self.prev_state = None
        super(Agent_PG,self).__init__(env)

        if args.test_pg:
            self.model = load_model('pong.h5')

    def init_game_setting(self):
        pass

    def _build_model(self):
        model = Sequential()
        model.add(Convolution2D(32, 6, 6, subsample=(3, 3), input_shape=(80,80,1), border_mode='same',activation='relu', init='he_uniform'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu', init='he_uniform'))
        model.add(Dense(32, activation='relu', init='he_uniform'))
        model.add(Dense(self.action_size, activation='softmax'))
        opt = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        return model

    def preprocess(self, I):
        I = I[35:195]
        I = I[::2, ::2, 0]
        I[I == 144] = 0
        I[I == 109] = 0
        I[I != 0] = 1
        I = I.astype(np.float).ravel()
        return I.reshape((80,80,1))

    def remember(self, state, action, prob, reward):
        y = np.zeros([self.action_size])
        y[action] = 1
        gradient = np.array(y).astype('float32') - prob
        self.gradients.append(gradient)
        self.states.append(state)
        self.rewards.append(reward)
        # self.memory.append((state, gradient, aprob, reward))

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def train(self):

        env = self.env
        state = env.reset()
        score = 0
        episode = 0

        while True:
            state = self.preprocess(state)
            x = state - self.prev_state if self.prev_state is not None else np.zeros(state.shape)
            self.prev_state = state

            action, prob = self.make_action(x, test=False)
            state, reward, done, info = env.step(action + 1)
            score += reward
            self.remember(x, action, prob, reward)
            if done:
                episode += 1
                gradients = np.vstack(self.gradients)
                rewards = np.vstack(self.rewards)
                rewards = self.discount_rewards(rewards)
                rewards = (rewards - np.mean(rewards)) / np.std(rewards)
                gradients *= rewards
                X = np.asarray(self.states)
                Y = self.probs + self.learning_rate * np.vstack([gradients])
                self.model.train_on_batch(X, Y)
                self.record.append(score) 
                running_reward = np.array(self.record).mean() if len(self.record) <= 30 else np.array(self.record).mean()
                self.states, self.probs, self.gradients, self.rewards = [], [], [], []
                print('Episode: %d - Score: %f. - moving average: %f' % (episode, score, running_reward))
                score = 0
                state = env.reset()
                self.prev_state = None
                if episode > 1 and episode % 20 == 0:
                    self.model.save('pong.h5')

        pass


    def make_action(self, state, test=True):
        if test :
            state = self.preprocess(state)
            x = state - self.prev_state if self.prev_state is not None else np.zeros(state.shape)
            self.prev_state = state
            state = np.expand_dims(x,axis=0)
            aprob = self.model.predict(state, batch_size=1).flatten()
            action = np.random.choice(self.action_size, 1, p=aprob)[0]
            return action
        else:
            state = np.expand_dims(state,axis=0)
            aprob = self.model.predict(state, batch_size=1).flatten()
            self.probs.append(aprob)
            action = np.random.choice(self.action_size, 1, p=aprob)[0] 
            return action, aprob


