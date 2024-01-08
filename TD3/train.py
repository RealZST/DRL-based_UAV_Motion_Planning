#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 14:44:16 2021

@author: zst
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import time
from td3 import TD3Agent
from gazebo_env import envmodel

'''''''''''setting before running'''''''''''
ON_TRAIN = 0  # train or not. if train, world in sitl.launch is 'train_env'
aDRL_TEST = 1  # test DGlobal or not
oDRL_TEST = 0  # test TD3 or not
test_env = 'Env2'  # consistent with the world setting in sitl.launch(test_env1,test_env2...)
sim_speed = 1  # simulation acceleration factor, corresponding to PX4_SIM_SPEED_FACTOR in sitl.launch
''''''''''''''''''''''''''''''
human_action = 0  # using human data buffer or not during training
random.seed(2)
np.random.seed(2)
env = envmodel(sim_speed, aDRL_TEST, oDRL_TEST, test_env)

gamma = 0.99
tau = 1e-2
policy_noise = 0.2
noise_bound = 0.5
delay_step = 2
critic_lr = 1e-5 
actor_lr = 1e-5 
buffer_maxlen = 10000  
batch_size = 256
agent = TD3Agent(env, gamma, tau, buffer_maxlen, delay_step, policy_noise, noise_bound, critic_lr, actor_lr)

if ON_TRAIN:
    # train
    t1 = time.time()
    MAX_EPISODES = 1000
    MAX_STEPS = 1000
    expl_noise = 1
    scores = []
    success = []
    collision = []
    lost = []
    text = None
    
    if human_action:
        agent.read_human_data()
        print('Read Human Data. Len of Human Data :', len(agent.replay_buffer.buffer_human))
    
    for i_episode in range(1, MAX_EPISODES+1):
        env.reset()
        env.check_fail()
        state, _, _ = env.get_env()
        episode_rewards = 0
        done = False
        
        for step in range(MAX_STEPS):
            action = np.clip((agent.get_action(state) 
                            + np.random.normal(0, expl_noise, size=env.actionDim)),
                            env.action_low, env.action_high)
            env.step(action)
            next_state, reward, done = env.get_env()
            agent.replay_buffer.push(state, action, reward, done)
            episode_rewards += reward
            
            if len(agent.replay_buffer) == buffer_maxlen:
                agent.update(batch_size)
                expl_noise *= 0.99999   
          
            if done or step == MAX_STEPS-1:
                env.land()
                if done:
                    text = 'SUCCESS' if reward > 0 else 'COLLIDED'
                    success += [reward>0]
                    collision += [reward<0]
                    lost.append(0)
                else:
                    text = 'LOST'
                    success.append(0)
                    collision.append(0)
                    lost.append(1)
                break
            
            state = next_state
        scores += [episode_rewards]
        print('\nEpisode', i_episode, text,
              ', Total Reward=%.2f'%episode_rewards,
              ', Success rate=%.5f' %np.mean(success[-20:]),
              ', Collision rate=%.5f' %np.mean(collision[-20:]),
              ', Lost rate=%.5f' %np.mean(lost[-20:]),
              ', buffer size=%d' %len(agent.replay_buffer),
              ', exploration noise std=%.6f' %expl_noise, end="")

        agent.save(directory = './exp_original/')
    
    print('\nRunning time: ', time.time() - t1)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(success[:])
    plt.ylabel('Success')
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(scores[:])
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.savefig('./result/train')

    # save scores
    name = ['Value']
    reward_curve = pd.DataFrame(columns=name,data=scores)
    reward_curve.to_csv('./result/scores_original.csv')

else:
    # test
    agent.load(directory = './exp_original/')
    MAX_EPISODES = 50
    MAX_STEPS = 2000
    scores = []
    success = []
    collision = []
    lost = []
    text = None

    for i_episode in range(1, MAX_EPISODES+1):
        env.reset()
        env.check_fail()
        state, _, _ = env.get_env()
        episode_rewards = 0
        done = False
        t2 = time.time()

        for step in range(MAX_STEPS):
            action = agent.get_action(state)
            env.step(action)
            state, reward, done = env.get_env()
            episode_rewards += reward
            
            if done or step == MAX_STEPS-1:
                env.land()
                if done:
                    text = 'SUCCESS' if reward > 0 else 'COLLIDED'
                    success += [reward>0]
                    collision += [reward<0]
                    lost.append(0)
                else:
                    text = 'LOST'
                    success.append(0)
                    collision.append(0)
                    lost.append(1)
                break
        if aDRL_TEST or oDRL_TEST:
            env.save_traj()
            print('\nRunning time: ', time.time() - t2)
        scores += [episode_rewards]
        print('\nEpisode', i_episode, text,
              ', Total Reward=%.2f' % episode_rewards,
              ', Success rate=%.5f' % np.mean(success[:]),
              ', Collision rate=%.5f' % np.mean(collision[:]),
              ', Lost rate=%.5f' % np.mean(lost[:]), end="")

