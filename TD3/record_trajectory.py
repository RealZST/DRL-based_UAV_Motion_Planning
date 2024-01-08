#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 10:21:30 2022

@author: zst
"""

"The actual files are located in '/Prometheus/Simulator/gazebo_simulator/py_nodes', this is a copy."
"Save the trajectory obtained using Astar, and call it in sitl_astar_2dlidar.launch"
"The trajectory will be stored in '/xxx/TD3/result/path.csv'"

import rospy
import time
import pandas as pd
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Quaternion
from prometheus_msgs.msg import DroneState
from tf.transformations import quaternion_from_euler

min_x = -45  # -10  
max_x = 45  # 10
min_y = -10  # -10
max_y = 10  # 10
h = 1  

class record_path:
    def __init__(self):
        rospy.init_node('path_subscriber', anonymous=True)
        self.goal_pub = rospy.Publisher('/prometheus/planning/goal', PoseStamped, queue_size=10)
        self.drone_state = DroneState()
        self.drone_state_sub = rospy.Subscriber('/prometheus/drone_state', DroneState, self.drone_state_callback)
        self.traj_x = []
        self.traj_y = []
#        self.traj_z = []

    def drone_state_callback(self, msg):
        self.drone_state = msg
     
    def subscribe_path(self):
        goal_coord = [max_x, np.random.uniform(min_y+5,max_y-5), h]  
        goal = PoseStamped()
        goal.pose.position.x = goal_coord[0]
        goal.pose.position.y = goal_coord[1]
        goal.pose.position.z = goal_coord[2]
        orien = [0.0, 0.0, 0.0]
        q = quaternion_from_euler(orien[0], orien[1], orien[2])
        goal.pose.orientation = Quaternion(*q)
        time.sleep(2)
        self.goal_pub.publish(goal)
        traj_last = []
        t1 = time.time()
        while 1:
            traj_temp = []
            path = rospy.wait_for_message("/prometheus/global_planning/path_cmd", Path, timeout=None)
            for i in range(len(path.poses)):
                traj_temp.append([path.poses[i].pose.position.x, path.poses[i].pose.position.y])
            
            if not traj_last:
                traj_last = traj_temp
            else:
                for i in range(len(traj_last)):
                    # When setting the destination, the x-coordinate of the destination must be greater than the x-coordinate of the starting point
                    if traj_temp[0][0] < traj_last[i][0]: 
                        del traj_last[i:]
                        traj_last.extend(traj_temp)
                        print('Update the trajectory')
                        break

                '''After each replanning, record the trajectory that has been flown so far'''
                traj_x = []
                traj_y = []
                for j in range(len(traj_last)):
                    if traj_last[j][0] > self.drone_state.position[0]:
                        break
                    traj_x.append(traj_last[j][0])
                    traj_y.append(traj_last[j][1])
                path_curve = pd.DataFrame({'traj_x': traj_x, 'traj_y': traj_y})
                path_curve.to_csv('/xxx/TD3/result/path.csv')
                print('Record the current trajectory')

            if len(traj_temp) < 10: # Assume that the last 10 points of the trajectory will not change anymore
                print('Get the final trajectory')
                print('Running time: ', time.time() - t1)
                break
        
        return traj_last

    def save_path(self):
        traj = self.subscribe_path()
        for i in range(len(traj)):
            self.traj_x.append(traj[i][0])
            self.traj_y.append(traj[i][1])
#            self.traj_z.append(traj[i][2])

        # Save the original path
        path_curve = pd.DataFrame({'traj_x':self.traj_x, 'traj_y':self.traj_y})
        path_curve.to_csv('/xxx/TD3/result/path.csv')
        print('Record the final trajectory')
    
if __name__=="__main__":
    record = record_path()
    # record.save_path()
    record.subscribe_path()