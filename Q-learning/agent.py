import tensorflow as tf
import numpy as np
import pandas as pd

class Agent(object):

    def __init__(self, state_dim, action_dim, batch_size):

        self.gamma = 0.99
        self.epsilon = 1.0
        self.learning_rate = 0.002
        
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.model = pd.DataFrame(columns=range(0,self.action_dim), dtype=np.float32)

    def get_action(self, state, test=False):
        state = np.array2string(state)
        self.check_state_exist(state)

        if np.random.rand() <= self.epsilon and not test:
            action = np.random.randint(self.action_dim)
        else:
            q_values = self.model.loc[state, :]
            action = np.argmax(q_values)
            #if test:
            #    print(state, q_values, action)

        return action

    def learn(self, state, action, reward, next_state, done):
        state = np.array2string(state)
        next_state = np.array2string(next_state)

        self.check_state_exist(next_state)
        q_val = self.model.loc[state, action]
        q_targ = (1-done)*self.gamma*np.max(self.model.loc[next_state,:])+reward

        self.model.loc[state, action] += self.learning_rate * (q_targ - q_val)

    def check_state_exist(self, state):
        if state not in self.model.index:
            self.model = self.model.append(pd.Series([0]*self.action_dim, index=self.model.columns, name=state))

    def epsilon_decay(self):
        self.epsilon = self.epsilon*0.997
        if self.epsilon<0.001:
            self.epsilon = 0.001