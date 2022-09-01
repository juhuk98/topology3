import numpy as np
import random
from collections import deque

class MemoryBuffer(object):
    """ Memory Buffer Helper class for Experience Replay
    using a double-ended queue or a Sum Tree (for PER)
    """
    def __init__(self, buffer_size, with_per = True):
        """ Initialization
        """
        if(with_per):
            # Prioritized Experience Replay
            self.alpha = 0.5
            self.epsilon = 0.01
            self.buffer = SumTree(buffer_size)
        else:
            # Standard Buffer
            self.buffer = deque()
        self.count = 0
        self.with_per = with_per
        self.buffer_size = buffer_size

    def memorize(self, state, action, reward, new_state, done, error=None):
        """ Save an experience to memory, optionally with its TD-Error
        """
        state = np.reshape(state, (1,-1))
        action = np.reshape(action, (1,-1))
        reward = np.reshape(reward, (1,-1))
        new_state = np.reshape(new_state, (1,-1))
        done = np.reshape(done, (1,-1))

        experience = (state, action, reward, done, new_state)
        if(self.with_per):
            priority = self.priority(error[action[0,0]])
            self.buffer.add(priority, experience)
            self.count += 1
        else:
            # Check if buffer is already full
            if self.count < self.buffer_size:
                self.buffer.append(experience)
                self.count += 1
            else:
                self.buffer.popleft()
                self.buffer.append(experience)

    def priority(self, error):
        """ Compute an experience priority, as per Schaul et al.
        """
        priority = (error + self.epsilon) ** self.alpha
        return np.float32(priority)

    def size(self):
        """ Current Buffer Occupation
        """
        return self.count

    def sample_batch(self, batch_size):
        """ Sample a batch, optionally with (PER)
        """
        batch = []

        # Sample using prorities
        if(self.with_per):
            T = self.buffer.total() // (batch_size)
            for i in range(batch_size):
                a, b = T * i, T * (i + 1)
                s = np.random.uniform(a, b, 1)
                idx, error, data = self.buffer.get(s)
                if data==0:
                    print(idx, error, data, a, b, s, self.buffer.total())
                    idx, error, data = self.buffer.get(s, 1)
                batch.append((*data, idx))
            idx = np.array([i[5] for i in batch])
        # Sample randomly from Buffer
        elif self.count < batch_size:
            idx = None
            batch = random.sample(self.buffer, self.count)
        else:
            idx = None
            batch = random.sample(self.buffer, batch_size)

        # Return a batch of experience
        s_batch = np.array([i[0] for i in batch])
        a_batch = np.array([i[1] for i in batch])
        r_batch = np.array([i[2] for i in batch])
        d_batch = np.array([i[3] for i in batch])
        new_s_batch = np.array([i[4] for i in batch])
        return s_batch, a_batch, r_batch, new_s_batch, d_batch, idx

    def update(self, idx, new_error):
        """ Update priority for idx (PER)
        """
        self.buffer.update(idx, self.priority(new_error))

    def clear(self):
        """ Clear buffer / Sum Tree
        """
        if(self.with_per): self.buffer = SumTree(self.buffer_size)
        else: self.buffer = deque()
        self.count = 0


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1 , dtype=np.float32)
        self.data = np.zeros( capacity, dtype=object )

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s, log=0):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx
        elif log:
            print(idx, left, right, s, self.tree[left], self.tree[right])
        
        if s <= self.tree[left]:
            return self._retrieve(left, s, log)
        else:
            return self._retrieve(right, s-self.tree[left], log)

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)
        
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s, log=0):
        idx = self._retrieve(0, s, log)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

    def priority_tree(self):
        return self.tree[self.capacity - 1:]