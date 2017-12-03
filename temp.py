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
class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)

        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
        self.env = env
        self.state = env.reset()
        self.action_size = env.action_space
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99    # discount rate
        self.epsilon = 1  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        # self.graph_ops = self.build_graph(4)
        # self.model = self._build_model()
        # self.Qval_model, self.ActionQval_model = self.build_ActionQval_model()
        # self.action_qval_model = self.build_ActionQval_model()
        ##################
        # YOUR CODE HERE #
        ##################
        self.state_inputs = Input(shape=(self.state.shape[0], self.state.shape[1], self.state.shape[2]))
        self.action_inputs = Input(shape=(4,))
        
        shared_model = Convolution2D(16, (3,3), padding ='same', activation='relu')(self.state_inputs)
        shared_model = MaxPool2D((2,2))(shared_model)
        shared_model = Convolution2D(32, (3,3), padding ='same', activation='relu')(shared_model)
        shared_model = MaxPool2D((2,2))(shared_model)
        shared_model = Flatten()(shared_model)
        self.shared_model = Dense(256, activation='relu')(shared_model)
        
        self.Q_val = Dense(4)(shared_model)
        
        self.merge = multiply([self.action_inputs,self.Q_val])
        self.Sum = Lambda(lambda x: K.sum(x, axis=0), output_shape=(1,))

        self.ActionQval_model = Model([self.state_inputs, self.action_inputs], self.Sum(self.merge))
        self.ActionQval_model.compile(loss='mse', optimizer='adam')
        #m = load_model("breakout_dqn.h5")
        #print("model loaded !")
        print(self.ActionQval_model.summary())

    def build_Qval_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Convolution2D(16, (4,4), padding ='same', activation='relu', input_shape=(self.state.shape[0], self.state.shape[1], self.state.shape[2])))
        model.add(MaxPool2D((2,2)))
        model.add((Convolution2D(32, (4,4), activation='relu', padding='same')))
        model.add((MaxPool2D((2,2))))
        model.add((Flatten()))
        model.add((Dense(256, activation='relu')))
        # model.add(GRU(32, activation='relu'))
        model.add(Dense(4))
        model.compile(loss='mse', optimizer='adam')
        #model = load_model("breakout_dqn.h5")
        #print("model loaded !")
        print(model.summary())
        return model

    def build_ActionQval_model(self):
        # Neural Net for Deep-Q learning Model
        state_inputs = Input(shape=(self.state.shape[0], self.state.shape[1], self.state.shape[2]))
        action_inputs = Input(shape=(4,))
        
        shared_model = Convolution2D(16, (3,3), padding ='same', activation='relu')(state_inputs)
        shared_model = MaxPool2D((2,2))(shared_model)
        shared_model = Convolution2D(32, (3,3), padding ='same', activation='relu')(shared_model)
        shared_model = MaxPool2D((2,2))(shared_model)
        shared_model = Flatten()(shared_model)
        shared_model = Dense(256, activation='relu')(shared_model)
        
        Q_val = Dense(4)(shared_model)
        Q_val_est = Model(state_inputs, Q_val)
        
        merge = multiply([action_inputs,Q_val])
        Sum = Lambda(lambda x: K.sum(x, axis=0), output_shape=(1,))

        ActionQval_model = Model([state_inputs, action_inputs], Sum(merge))
        ActionQval_model.compile(loss='mse', optimizer='adam')
        #m = load_model("breakout_dqn.h5")
        #print("model loaded !")
        print(ActionQval_model.summary())
        return Q_val_est, ActionQval_model

    # def build_network(self, num_actions, agent_history_length, resized_width, resized_height):
    #     with tf.device("/cpu:0"):
    #         state = tf.placeholder("float", [None, agent_history_length, resized_width, resized_height])
    #         inputs = Input(shape=(agent_history_length, resized_width, resized_height,))
    #         model = Convolution2D(16, (8,8), strides = (4,4), activation='relu', padding='same')(inputs)
    #         model = Convolution2D(32, (4,4), strides = (2,2), activation='relu', padding='same')(model)
    #         model = Flatten()(model)
    #         model = Dense(256, activation='relu')(model)
    #         q_values = Dense(num_actions, activation='linear')(model)
    #         m = Model(inputs, q_values)
    #     return state, m

    # def build_graph(self, num_actions):
    #     # Create shared deep q network
    #     s, q_network = self.build_network(num_actions, 4, 84, 84)
    #     network_params = q_network.trainable_weights
    #     q_values = q_network(s)

    #     # Create shared target network
    #     st, target_q_network = self.build_network(num_actions, 4, 84, 84)
    #     target_network_params = target_q_network.trainable_weights
    #     target_q_values = target_q_network(st)

    #     # Op for periodically updating target network with online network weights
    #     reset_target_network_params = [target_network_params[i].assign(network_params[i]) for i in range(len(target_network_params))]
        
    #     # Define cost and gradient update op
    #     a = tf.placeholder("float", [None, num_actions])
    #     y = tf.placeholder("float", [None])
    #     action_q_values = tf.reduce_sum(q_values * a, reduction_indices=1)
    #     cost = tf.reduce_mean(tf.square(y - action_q_values))
    #     optimizer = tf.train.AdamOptimizer(0.001)
    #     grad_update = optimizer.minimize(cost, var_list=network_params)

    #     graph_ops = {"s" : s, 
    #                  "q_values" : q_values,
    #                  "st" : st, 
    #                  "target_q_values" : target_q_values,
    #                  "reset_target_network_params" : reset_target_network_params,
    #                  "a" : a,
    #                  "y" : y,
    #                  "grad_update" : grad_update}
    #     return graph_ops

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def init_game_setting(self):
        """
        
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        #self.model = load_model("breakout_dqn.h5")
        pass


    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(4)
        state = state.reshape((1, state.shape[0],state.shape[1],state.shape[2]))
        Q_est = Model(self.state_inputs, self.Q_val)
        act_values = Q_est.predict(state)
        return np.argmax(act_values[0])  
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        X_state, X_action, Y = [], [], []
        Q_est = Model(self.state_inputs, self.Q_val)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            state = state.reshape((1, state.shape[0],state.shape[1],state.shape[2]))
            next_state = next_state.reshape((1, next_state.shape[0],next_state.shape[1],next_state.shape[2]))
            if not done:
                target = reward + self.gamma * \
                       np.amax(Q_est.predict(next_state)[0])

            target_f = Q_est.predict(state)
            target_f[0][action] = target

            action_input = np.zeros(4)
            action_input[action] = 1

            Y.append(target)
            X_state.append(state.reshape((state.shape[1],state.shape[2],state.shape[3])))
            X_action.append(action_input)
        
        self.ActionQval_model.fit([np.asarray(X_state),np.asarray(X_action)] , np.asarray(Y), epochs=1, verbose=0, batch_size=64)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        # g = tf.Graph()
        # with g.as_default(), tf.Session() as session:
        #     K.set_session(session)
        #     graph_ops = self.build_graph(4)
        #     session.run(graph_ops["reset_target_network_params"])
        #     init = tf.global_variables_initializer()
        #     session.run(init)
        episodes = 20000
        for e in range(episodes):
            # reset state in the beginning of each game
            # time_t represents each frame of the game
            # Our goal is to keep the pole upright as long as possible until score of 500
            # the more time_t the more score
            env = self.env
            state = env.reset()
            total_reward = 0
            for time_t in range(3000):
                # turn this on if you want to render
                # self.env.render()
                # Decide action
                action = self.act(state)
                # Advance the game to the next frame based on the action.
                # Reward is 1 for every frame the pole survived
                next_state, reward, done, info = self.env.step(action)
                total_reward += reward
                # Remember the previous state, action, reward, and done
                self.remember(state, action, reward, next_state, done)
                # make next_state the new current state for the next frame.
                state = next_state
                # done becomes True when the game ends
                # ex) The agent drops the pole
                if done:
                    # print the score and break out of the loop
                    print("episode: {}/{}, score: {}, epsilon: {}"
                          .format(e, episodes, total_reward, self.epsilon))
                    break
            # train the agent with the experience of the episode
            replay = len(self.memory) if len(self.memory) < 128 else 128
            self.replay(replay)
            if e%500 == 0:
                #self.Qval_model.save("breakout_dqn_Qval.h5")
                self.ActionQval_model.save_weights("breakout_dqn_ActionQval.h5")
        pass



    def make_action(self, observation, test=True):

        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        if np.random.rand() <= self.epsilon:
            return random.randrange(4)
        observation = observation.reshape((1, observation.shape[0],observation.shape[1],observation.shape[2]))
        act_values = self.model.predict(observation)
        return np.argmax(act_values[0])  
        # return self.env.get_random_action()

