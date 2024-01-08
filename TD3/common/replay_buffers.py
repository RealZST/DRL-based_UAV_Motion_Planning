import random
import numpy as np
from collections import deque
from common.data_structures import SumTree

#random.seed(7)

class BasicBuffer:

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.buffer_human = deque(maxlen=2000)

    def push(self, state, action, reward, done):
        # experience = (state, action, np.array([reward]), next_state, np.array([done]))
        experience = (state, action, np.array([reward]), np.array([done]))
        self.buffer.append(experience)
        
    def push_human(self, state, action, reward, done):
        # experience = (state, action, np.array([reward]), next_state, np.array([done]))
        experience = (state, action, np.array([reward]), np.array([done]))
        self.buffer_human.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        if len(self.buffer_human) == 0:
            index = np.random.randint(len(self.buffer) - 1, size=batch_size)
            for i in index:
                state, action, reward, done = self.buffer[i]
                next_state, _, _, _ = self.buffer[i+1]
                state_batch.append(state)
                action_batch.append(action)
                reward_batch.append(reward)
                next_state_batch.append(next_state)
                done_batch.append(done)

            # batch = random.sample(self.buffer, batch_size)
        else:
            index = np.random.randint(len(self.buffer) - 1, size=int(batch_size/2))
            index_human = np.random.randint(len(self.buffer_human) - 1, size=int(batch_size/2))
            for i in index:
                state, action, reward, done = self.buffer[i]
                next_state, _, _, _ = self.buffer[i + 1]
                state_batch.append(state)
                action_batch.append(action)
                reward_batch.append(reward)
                next_state_batch.append(next_state)
                done_batch.append(done)
            for j in index_human:
                state, action, reward, done = self.buffer[j]
                next_state, _, _, _ = self.buffer[j + 1]
                state_batch.append(state)
                action_batch.append(action)
                reward_batch.append(reward)
                next_state_batch.append(next_state)
                done_batch.append(done)

        #     batch = random.sample(self.buffer, int(batch_size/2)) + random.sample(self.buffer_human, int(batch_size/2))
        #
        # for experience in batch:
        #     state, action, reward, next_state, done = experience
        #     state_batch.append(state)
        #     action_batch.append(action)
        #     reward_batch.append(reward)
        #     next_state_batch.append(next_state)
        #     done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def sample_sequence(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        min_start = len(self.buffer) - batch_size
        start = np.random.randint(0, min_start)

        for sample in range(start, start + batch_size):
            state, action, reward, next_state, done = self.buffer[start]
#            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)


class PrioritizedBuffer:

    def __init__(self, max_size, alpha=0.6, beta=0.4):
        self.sum_tree = SumTree(max_size)
        self.alpha = alpha
        self.beta = beta
        self.current_length = 0

    def push(self, state, action, reward, next_state, done):
#        priority = 1.0 if self.current_length is 0 else self.sum_tree.tree.max()
        priority = 1.0 if self.current_length == 0 else self.sum_tree.tree.max()
        self.current_length = self.current_length + 1
        #priority = td_error ** self.alpha
        experience = (state, action, np.array([reward]), next_state, done)
        self.sum_tree.add(priority, experience)

    def sample(self, batch_size):
        batch_idx, batch, IS_weights = [], [], []
        segment = self.sum_tree.total() / batch_size
        p_sum = self.sum_tree.tree[0]

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            idx, p, data = self.sum_tree.get(s)

            batch_idx.append(idx)
            batch.append(data)
            prob = p / p_sum
            IS_weight = (self.sum_tree.total() * prob) ** (-self.beta)
            IS_weights.append(IS_weight)

        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        for transition in batch:
            state, action, reward, next_state, done = transition
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch), batch_idx, IS_weights

    def update_priority(self, idx, td_error):
        priority = td_error ** self.alpha
        self.sum_tree.update(idx, priority)

    def __len__(self):
        return self.current_length
