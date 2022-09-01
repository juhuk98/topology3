from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Lambda

class DQN(Model):
    def __init__(self, action_dim):
        super(DQN, self).__init__()

        self.h1 = Dense(64, activation='relu')
        self.h2 = Dense(64, activation='relu')
        self.v = Dense(action_dim, activation='linear')

    def call(self, state):
        x = self.h1(state)
        x = self.h2(x)
        v = self.v(x)
        return v