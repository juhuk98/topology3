import tensorflow as tf
import numpy as np

from memory import MemoryBuffer
from networks import DQN
from tensorflow.keras.optimizers import Adam

class DDQNAgent(object):

    def __init__(self, state_dim, action_dim, batch_size):

        self.gamma = 0.99
        self.epsilon = 1.0
        self.learning_rate = 0.002
        self.with_per = False
        self.buffer = MemoryBuffer(buffer_size=50000, with_per=self.with_per)
        self.batch_size = batch_size
        self.fixed_target = True
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.model = DQN(action_dim)
        self.target_model = DQN(action_dim)
        self.model.build(input_shape=(None, self.state_dim))
        self.target_model.build(input_shape=(None, self.state_dim))

        #self.model.summary()

        lr_decay_fn = tf.keras.optimizers.schedules.CosineDecay(self.learning_rate, 2500)
        self.model_opt = Adam(lr_decay_fn)

    def compute_loss(self, y_true, y_pred):
        h = tf.keras.losses.Huber()
        return h(y_true, y_pred)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def store_memory(self, state, action, reward, next_state, done):
        if self.fixed_target:
            next_q = self.target_model(tf.convert_to_tensor(next_state, dtype=tf.float32))
        else:
            next_q = self.model(tf.convert_to_tensor(next_state, dtype=tf.float32))
        
        max_next_q = tf.reduce_max(next_q, axis=1)
        max_next_q = tf.reshape(max_next_q, (-1, 1))
        new_val = (1-done) * self.gamma * max_next_q + reward

        q_val = self.model(tf.convert_to_tensor(state, dtype=tf.float32))

        td_error = abs(new_val - q_val)[0]
        
        self.buffer.memorize(state, action, reward, next_state, done, td_error)

    def get_action(self, state, test=False):
        state = tf.convert_to_tensor(state, dtype=tf.float32)

        if np.random.rand() <= self.epsilon and not test:
            action = np.random.randint(self.action_dim)
        else:
            q_values = self.model(state)
            action = np.argmax(q_values[0])
            #if test:
            #    print(state, q_values, action)

        return action

    def learn(self):
        states, actions, rewards, next_states, dones, idx = self.buffer.sample_batch(self.batch_size)
        states = self.unpack_batch(states)
        actions = self.unpack_batch(actions)
        rewards = self.unpack_batch(rewards)
        next_states = self.unpack_batch(next_states)
        dones = self.unpack_batch(dones)


        with tf.GradientTape() as tape:
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.int32)
            actions = tf.squeeze(actions)
            dones = tf.convert_to_tensor(dones, dtype=tf.float32)

            if self.fixed_target:
                next_q = self.target_model(tf.convert_to_tensor(next_states, dtype=tf.float32))
            else:
                next_q = self.model(tf.convert_to_tensor(next_states, dtype=tf.float32))
            
            max_next_q = tf.reduce_max(next_q, axis=1)
            max_next_q = tf.reshape(max_next_q, (-1, 1))
            new_val = (1-dones) * self.gamma * max_next_q + rewards

            q_val = self.model(tf.convert_to_tensor(states, dtype=tf.float32))
            q_val = tf.reduce_sum(tf.one_hot(actions, self.action_dim)*q_val, axis=1)
            q_val = tf.reshape(q_val, (-1, 1))

            td_error = tf.stop_gradient(q_val) - tf.stop_gradient(new_val)

            #loss = self.compute_loss(new_val, q_val)
            loss = tf.math.reduce_mean(tf.sqrt(1+tf.square(new_val - q_val))-1)

        td_error = np.abs(td_error)
        if (self.with_per):
            for i in range(self.state_dim):
                self.buffer.update(idx[i], td_error[i][0])

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model_opt.apply_gradients(zip(grads, self.model.trainable_variables))
            

    def load_weights(self, path):
        self.model.load_weights(path + 'dqn_model.h5')

    def save_weights(self, path):
        self.model.save_weights(path + 'dqn_model.h5')

    def unpack_batch(self, batch):
        unpack = batch[0]
        for idx in range(len(batch)-1):
            unpack = np.append(unpack, batch[idx+1], axis=0)

        return unpack

    def print_lr(self):
        print(self.model_opt._decayed_lr(tf.float32))

    def epsilon_decay(self):
        self.epsilon = self.epsilon*0.997
        if self.epsilon<0.001:
            self.epsilon = 0.001