#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from models import Critic, Actor
from common.replay_buffers import BasicBuffer

np.random.seed(2) #numpy
torch.manual_seed(2) # cpu
torch.cuda.manual_seed(2) #gpu

class TD3Agent:

    def __init__(self, env, gamma, tau, buffer_maxlen, delay_step, noise_std, noise_bound, critic_lr, actor_lr):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#        self.device = torch.device("cpu")
        
        self.env = env
        self.obs_dim = env.stateDim
        self.action_dim = env.actionDim
        self.action_low = env.action_low
        self.action_high = env.action_high

        # hyperparameters    
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.noise_bound = noise_bound
        self.update_step = 0 
        self.delay_step = delay_step
        
        # initialize actor and critic networks
        self.critic1 = Critic(self.obs_dim, self.action_dim).to(self.device)
        self.critic2 = Critic(self.obs_dim, self.action_dim).to(self.device)
        self.critic1_target = Critic(self.obs_dim, self.action_dim).to(self.device)
        self.critic2_target = Critic(self.obs_dim, self.action_dim).to(self.device)
        
        self.actor = Actor(self.obs_dim, self.action_dim).to(self.device)
        self.actor_target = Actor(self.obs_dim, self.action_dim).to(self.device)
    
        # Copy critic target parameters
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(param.data)

        # initialize optimizers        
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr) 
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        self.replay_buffer = BasicBuffer(buffer_maxlen)
#        self.replay_buffer = PrioritizedBuffer(buffer_maxlen)
        
        self.temp_states = []
        self.temp_actions = []
        self.temp_rewards = []
        self.temp_next_states = []
        self.temp_dones = []

    def get_action(self, obs):
        state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action = self.actor.forward(state)
        action = action.squeeze(0).cpu().detach().numpy()

        return action
    
    def update(self, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, masks = self.replay_buffer.sample(batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        masks = torch.FloatTensor(masks).to(self.device)
        
        action_space_noise = self.generate_action_space_noise(action_batch)
        # next_actions = self.actor.forward(state_batch) + action_space_noise
        next_actions = (self.actor_target.forward(next_state_batch) + action_space_noise).clamp(self.action_low, self.action_high)
        next_Q1 = self.critic1_target.forward(next_state_batch, next_actions)
        next_Q2 = self.critic2_target.forward(next_state_batch, next_actions)
#        expected_Q = reward_batch + self.gamma * torch.min(next_Q1, next_Q2)
        expected_Q = reward_batch + (1 - masks) * self.gamma * torch.min(next_Q1, next_Q2)

        # critic loss
        curr_Q1 = self.critic1.forward(state_batch, action_batch)
        curr_Q2 = self.critic2.forward(state_batch, action_batch)
        critic1_loss = F.mse_loss(curr_Q1, expected_Q.detach())
        critic2_loss = F.mse_loss(curr_Q2, expected_Q.detach())
        
        # update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # delyaed update for actor & target networks  
        if self.update_step % self.delay_step == 0:
            # actor
            self.actor_optimizer.zero_grad()
            policy_gradient = -self.critic1(state_batch, self.actor(state_batch)).mean()
            policy_gradient.backward()
            self.actor_optimizer.step()

            # target networks
            self.update_targets()

        self.update_step += 1

    def generate_action_space_noise(self, action_batch):
        noise = torch.normal(torch.zeros(action_batch.size()), self.noise_std).clamp(-self.noise_bound, self.noise_bound).to(self.device)
        return noise

    def update_targets(self):
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
        
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def q_value_target(self, state_q, action_q):
        state_q = torch.FloatTensor([state_q]).to(self.device)
        action_q = torch.FloatTensor([action_q]).to(self.device)
        Q1 = self.critic1_target.forward(state_q, action_q)
        Q2 = self.critic2_target.forward(state_q, action_q)
        q_value = torch.min(Q1, Q2)[0][0].cpu().detach().numpy()
        return q_value

    def save(self, directory):
        torch.save(self.critic1.state_dict(), directory+'critic1.pth')
        torch.save(self.critic2.state_dict(), directory+'critic2.pth')
        torch.save(self.critic1_target.state_dict(), directory+'critic1_target.pth')
        torch.save(self.critic2_target.state_dict(), directory+'critic2_target.pth')
        torch.save(self.actor.state_dict(), directory+'actor.pth')
        torch.save(self.actor_target.state_dict(), directory+'actor_target.pth') 

    def load(self, directory):
        self.critic1.load_state_dict(torch.load(directory + 'critic1.pth'))
        self.critic2.load_state_dict(torch.load(directory + 'critic2.pth'))
        self.critic1_target.load_state_dict(torch.load(directory + 'critic1_target.pth'))
        self.critic2_target.load_state_dict(torch.load(directory + 'critic2_target.pth'))
        self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
        self.actor_target.load_state_dict(torch.load(directory + 'actor_target.pth'))  
        
    def read_human_data(self):
        data = np.load('human_data.npz')
        states = data['s']
        actions = data['a']
        rewards = data['r']
        # next_states = data['n_s']
        dones = data['d']
        for i in range(len(states)):
            self.replay_buffer.push_human(states[i], actions[i], rewards[i], dones[i])
        



