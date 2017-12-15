from agent_dir.agent import Agent
from collections import deque
from keras.models import Sequential
from keras.layers import Lambda, Dense, Activation, GRU, TimeDistributed, Conv1D, MaxPooling1D, MaxPooling2D, Flatten, Dropout, SimpleRNN, Bidirectional, Convolution2D, Permute, BatchNormalization, RepeatVector, Input, MaxPool2D 
from keras.models import load_model, Model
from keras.layers import merge, Concatenate, Merge, add, Masking, Activation, dot, multiply, concatenate
from keras import backend as K
import tensorflow as tf 
import numpy as np
import random
import time
class Agent_DQN(Agent):
    def __init__(self, env, args):

        super(Agent_DQN,self).__init__(env)
        self.env = env
        self.state = env.reset()
        self.action_size = env.action_space
        self.memory = deque(maxlen=50000)
        self.gamma = 0.99    # discount rate
        self.epsilon = 0.001 # exploration rate
        self.epsilon_min = 0.00001
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.session = ""
        self.graph_ops = ""
        if args.test_dqn:
            g = tf.Graph()
            self.session = tf.Session(graph=g)
            with g.as_default(), self.session.as_default():
                K.set_session(self.session)
                self.graph_ops = self.build_graph(3)
                saver = tf.train.Saver()
                saver.restore(self.session, "./breakout.ckpt")
                self.session.run(self.graph_ops["reset_target_network_params"])

    def build_network(self, num_actions, w, h, c):
        state = tf.placeholder("float", [None, w, h, c])
        inputs = Input(shape=(w, h, c))
        model = Convolution2D(32, (3,3), activation='relu', padding='same')(inputs)
        model = MaxPool2D((2,2))(model)
        model = Convolution2D(64, (3,3), activation='relu', padding='same')(model)
        model = MaxPool2D((2,2))(model)
        model = Flatten()(model)
        model = Dense(512, activation='relu')(model)
        q_values = Dense(num_actions, activation='linear')(model)
        m = Model(inputs, q_values)
        return state, m

    def build_graph(self, num_actions):
        # Create shared deep q network
        s, q_network = self.build_network(num_actions, 84, 84, 4)
        network_params = q_network.trainable_weights
        q_values = q_network(s)

        # Create shared target network
        st, target_q_network = self.build_network(num_actions, 84, 84, 4)
        target_network_params = target_q_network.trainable_weights
        target_q_values = target_q_network(st)

        # Op for periodically updating target network with online network weights
        reset_target_network_params = [target_network_params[i].assign(network_params[i]) for i in range(len(target_network_params))]
        
        # Define cost and gradient update op
        a = tf.placeholder("float", [None, num_actions])
        y = tf.placeholder("float", [None])
        action_q_values = tf.reduce_sum(q_values * a, reduction_indices=1)
        cost = tf.reduce_mean(tf.square(y - action_q_values))
        optimizer = tf.train.AdamOptimizer(0.001)
        grad_update = optimizer.minimize(cost, var_list=network_params)

        graph_ops = {"s" : s, 
                     "q_values" : q_values,
                     "st" : st, 
                     "target_q_values" : target_q_values,
                     "reset_target_network_params" : reset_target_network_params,
                     "a" : a,
                     "y" : y,
                     "grad_update" : grad_update}
        return graph_ops

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def init_game_setting(self): 
        pass


    def replay(self, batch_size, session, graph_ops):
        minibatch = random.sample(self.memory, batch_size)
        X_state, X_action, Y = [], [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                       np.amax(graph_ops['target_q_values'].eval(session = session, feed_dict = {graph_ops['st'] : [next_state]}))

            action_input = np.zeros(3)
            action_input[action-1] = 1

            Y.append(target)
            X_state.append(state)
            X_action.append(action_input)

        session.run(graph_ops['grad_update'], feed_dict = { graph_ops['y'] : np.asarray(Y),
                                                            graph_ops['a'] : np.asarray(X_action),
                                                            graph_ops['s'] : np.asarray(X_state)} )
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self):
        g = tf.Graph()
        session = tf.Session(graph=g)
        with g.as_default(), session.as_default():
            K.set_session(session)
            graph_ops = self.build_graph(3)
            saver = tf.train.Saver()
            #saver.restore(session, "./breakout.ckpt")
            session.run(tf.global_variables_initializer())
            session.run(graph_ops["reset_target_network_params"])
            #writer = tf.train.SummaryWriter('./', session.graph)
            episodes = 100000
            
            for e in range(episodes):
                
                env = self.env
                state = env.reset()
                total_reward = 0
                
                for time_t in range(10000):
                    action = self.make_action(state, session, graph_ops)
                    next_state, reward, done, info = self.env.step(action)
                    total_reward += reward
                    self.remember(state, action, reward, next_state, done)
                    state = next_state
                    if done:
                        print("episode: {}/{}, score: {}"
                              .format(e, episodes, total_reward))
                        break
                
                replay = len(self.memory) if len(self.memory) < 64 else 64
                self.replay(replay, session, graph_ops)

                if e % 100 == 0:
                    session.run(graph_ops['reset_target_network_params'])

                # if e % 200 == 0:
                #     saver.save(session, "./breakout.ckpt")
        pass

    def make_action(self, state, test=True):
        if test:
            self.epsilon = 0.01

        if np.random.rand() <= self.epsilon:
            return random.randrange(3)+1

        action = self.graph_ops['q_values'].eval(session = self.session, feed_dict = {self.graph_ops['s'] : [state]})
        return np.argmax(action)+1
