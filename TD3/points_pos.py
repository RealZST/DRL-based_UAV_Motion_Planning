#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 17:11:20 2021

@author: ZST
"""

"1. Randomly generate starting points, goal points, and obstacle coordinates for TD3 training"
"2. Process the trajectory obtained from Astar using RDP to generate waypoints"

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdp import rdp

min_x = -10  
max_x = 10
min_y = -10
max_y = 10

h = 1  # fixed flight altitude
obs_h = 2  # obstacle height
rdp_e = 3  # the higher the value, the fewer the points

class points_pos:

    # Randomly generate starting and ending coordinates for TD3 training
    @staticmethod
    def waypoints_ori():
        start = [min_x, np.random.uniform(min_y+5,max_y-5), h] 
        goal = [max_x, np.random.uniform(min_y+5,max_y-5), h]
        
        return start, goal

    # Randomly generate obstacle coordinates for TD3 training
    @staticmethod
    def obspoints_ori(obs_n, obs_r):
        obspoints_list = []
        i = 0
        while i < obs_n:
            flag = True
            temp = [np.random.uniform(min_x+2, max_x-2), np.random.uniform(min_y+2*obs_r, max_y-2*obs_r), obs_h/2]
            for j in range(i):
                if math.sqrt((obspoints_list[j][0]-temp[0])**2+(obspoints_list[j][1]-temp[1])**2) < 2*obs_r:
                    flag = False
                    break
            if flag:
                i += 1
                obspoints_list.append(temp)
                
        return obspoints_list

    # Process the trajectory obtained from Astar using RDP to generate waypoints, starting, and ending points
    @staticmethod
    def rdp_path(test_env):
        path = pd.read_csv('./result/path_test' + test_env + '.csv')
        path_x = path['traj_x'].values # path_x为array
        path_y = path['traj_y'].values
        path_xy = np.vstack((path_x, path_y)).T
        path_xy_rdp = rdp(path_xy, epsilon=rdp_e)
        path_rdp = np.ones((len(path_xy_rdp),3))

        plt.figure()
        plt.plot(path_x, path_y)
        plt.scatter(path_xy_rdp[:,0],path_xy_rdp[:,1])
        # plt.show()
        plt.savefig('./result/path_test' + test_env)

        for i in range(len(path_xy_rdp)):
            path_rdp[i] = np.hstack((path_xy_rdp[i], h))  # add altitude information

        return path_rdp

if __name__=="__main__":
    points = points_pos()
    traj = points.rdp_path(test_env='Env2')
    print(traj)
