#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 18:48:47 2021

@author: zst
"""

"Env file"

import rospy
import numpy as np
import pandas as pd
import math
import time
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import SetModelState
from tf.transformations import quaternion_from_euler
from prometheus_msgs.msg import ControlCommand, DroneState
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import LaserScan
from points_pos import points_pos

points = points_pos()
MAX_X = 95  # 25  # Maximum distance in the x-direction between the drone and the goal 
MAX_Y = 20  # 20
MAXDIS = math.sqrt(MAX_X**2 + MAX_Y**2)  # maximum distance to the goal, used for normalization
MAXLASERDIS = 5.0
MAXVEL = 2.0  # Maximum horizontal flight speed of the drone, corresponding to the MPC_XY_VEL_MAX setting in QGroundControl
obs_safe_dis = 1  # Safe distance to obstacles
drone_r = 0.277*math.sqrt(2)  # Drone radius  # 0.277*math.sqrt(2)=0.39174
obs_r = 0.5  # Obstacle radius, corresponding to the parameters in obs_cylinder file
obs_n = 10  # num of obstacles
goal_r = 0.3  # radius of the goal circle
waypoint_r = 1.0  # radius of waypoints
epsilon = 0.8 # expansion range of waypoints

class envmodel:
    stateDim = 724  # 720(num of laser samples) + 4
    actionDim = 2  # 2(velocity in xy)
    action_low = -1
    action_high = 1
    
    def __init__(self, sim_speed, aDRL_TEST, oDRL_TEST, test_env):
        rospy.init_node('control_node' ,anonymous=True)
        rospy.wait_for_service('gazebo/set_model_state')
        self.drone_state = DroneState()
        self.laser = LaserScan()
        self.model_states = ModelStates()
        self.state = ModelState()
        self.drone_state_sub = rospy.Subscriber('/prometheus/drone_state', DroneState, self.drone_state_callback)
        self.laser_sub = rospy.Subscriber('/prometheus/sensors/2Dlidar_scan', LaserScan, self.laser_callback)
        self.model_states_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_callback)
        self.move_pub = rospy.Publisher('/prometheus/control_command', ControlCommand, queue_size=1000)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.set_state_service = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)
        self.sim_speed = sim_speed
        self.aDRL_TEST = aDRL_TEST
        self.oDRL_TEST = oDRL_TEST
        self.test_env = test_env
        self.path_rdp = []
        self.goal = []
        self.traj_x = []
        self.traj_y = []
        self.comid = 0
        self.i_rdp = 0
        self.dis_to_goal = 0
        self.angle = 0
        time.sleep(0.5)
        
    def drone_state_callback(self, msg):
        self.drone_state = msg
        
    def laser_callback(self, msg):
        self.laser = msg
        
    def model_states_callback(self, msg):
        self.model_states = msg
    
    def reset(self):
        if self.aDRL_TEST or self.oDRL_TEST:  # test of TD3 or DGlobal
            self.path_rdp = points.rdp_path(self.test_env)
            start_point = self.path_rdp[0]
            goal_point = self.path_rdp[-1]
            if self.aDRL_TEST:
                print(self.path_rdp)
                self.goal = self.path_rdp[1]  # self.path_rdp[1] # consistent with self.i_rdp = 1
            if self.oDRL_TEST:
                self.goal = self.path_rdp[-1]

            "Reset the starting point, target point, and drone position"
            for i in range(len(self.model_states.name)):
                if self.model_states.name[i] == 'start_point':
                    self.state.reference_frame = 'world'
                    self.state.model_name = self.model_states.name[i]
                    self.state.pose.position.x = start_point[0]
                    self.state.pose.position.y = start_point[1]
                    self.state.pose.position.z = start_point[2]
                    self.set_state_service(self.state)
                if self.model_states.name[i] == 'goal_point':
                    self.state.reference_frame = 'world'
                    self.state.model_name = self.model_states.name[i]
                    self.state.pose.position.x = goal_point[0]
                    self.state.pose.position.y = goal_point[1]
                    self.state.pose.position.z = goal_point[2]
                    self.set_state_service(self.state)
                if self.model_states.name[i] == 'p450_hokuyo_2Dlidar':
                    self.state.reference_frame = 'world'
                    self.state.model_name = self.model_states.name[i]
                    self.state.pose.position.x = start_point[0]
                    self.state.pose.position.y = start_point[1]
                    self.state.pose.position.z = 0.2
                    orien = [0.0, 0.0, 0.0]
                    q = quaternion_from_euler(orien[0], orien[1], orien[2])
                    self.state.pose.orientation = Quaternion(*q)
                    self.state.twist.linear.x = 0.0
                    self.state.twist.linear.y = 0.0
                    self.state.twist.linear.z = 0.0
                    self.state.twist.angular.x = 0.0
                    self.state.twist.angular.y = 0.0
                    self.state.twist.angular.z = 0.0
                    self.set_state_service(self.state)
                    time.sleep(10.0/self.sim_speed)

            if self.aDRL_TEST:  # DGlobal test requires setting waypoints
                "reset pos if waypoints"
                for k in range(20):
                    NAME_WAY = 'waypoint' + str(k)
                    self.state.model_name = NAME_WAY
                    self.state.reference_frame = 'world'
                    self.state.pose.position.x = -60
                    self.state.pose.position.y = -10
                    self.state.pose.position.z = 1
                    self.set_state_service(self.state)
                for k in range(1, len(self.path_rdp)-1):
                    NAME_WAY = 'waypoint' + str(k)
                    self.state.model_name = NAME_WAY
                    self.state.reference_frame = 'world'
                    self.state.pose.position.x = self.path_rdp[k][0]
                    self.state.pose.position.y = self.path_rdp[k][1]
                    self.state.pose.position.z = self.path_rdp[k][2]
                    self.set_state_service(self.state)

        else:  # training DGlobal
            start_point, goal_point = points.waypoints_ori()
            self.goal = goal_point
            obs_pos = points.obspoints_ori(obs_n, obs_r)

            "Reset the starting point, target point, and drone position"
            for i in range(len(self.model_states.name)):
                if self.model_states.name[i] == 'start_point':
                    self.state.reference_frame = 'world'
                    self.state.model_name = self.model_states.name[i]
                    self.state.pose.position.x = start_point[0]
                    self.state.pose.position.y = start_point[1]
                    self.state.pose.position.z = start_point[2]
                    self.set_state_service(self.state)
                if self.model_states.name[i] == 'goal_point':
                    self.state.reference_frame = 'world'
                    self.state.model_name = self.model_states.name[i]
                    self.state.pose.position.x = goal_point[0]
                    self.state.pose.position.y = goal_point[1]
                    self.state.pose.position.z = goal_point[2]
                    self.set_state_service(self.state)
                if self.model_states.name[i] == 'p450_hokuyo_2Dlidar':
                    self.state.reference_frame = 'world'
                    self.state.model_name = self.model_states.name[i]
                    self.state.pose.position.x = start_point[0]
                    self.state.pose.position.y = start_point[1]
                    self.state.pose.position.z = 0.2
                    orien = [0.0, 0.0, 0.0]
                    q = quaternion_from_euler(orien[0], orien[1], orien[2])
                    self.state.pose.orientation = Quaternion(*q)
                    self.state.twist.linear.x = 0.0
                    self.state.twist.linear.y = 0.0
                    self.state.twist.linear.z = 0.0
                    self.state.twist.angular.x = 0.0
                    self.state.twist.angular.y = 0.0
                    self.state.twist.angular.z = 0.0
                    self.set_state_service(self.state)
                    time.sleep(10.0 / self.sim_speed)

            "Reset the obstacle positions"
            for k in range(11):
                NAME_OBS = 'obs' + str(k)
                self.state.model_name = NAME_OBS
                self.state.reference_frame = 'world'
                self.state.pose.position.x = -20
                self.state.pose.position.y = -10 + 2 * k
                self.state.pose.position.z = 1
                self.set_state_service(self.state)
            for k in range(len(obs_pos)):
                NAME_OBS = 'obs' + str(k)
                self.state.model_name = NAME_OBS
                self.state.reference_frame = 'world'
                self.state.pose.position.x = obs_pos[k][0]
                self.state.pose.position.y = obs_pos[k][1]
                self.state.pose.position.z = obs_pos[k][2]
                self.set_state_service(self.state)

        self.comid = 0
        self.i_rdp = 1  # 1
        self.arm_offboard()
        self.takeoff()
        
    def check_fail(self):    
        while self.drone_state.armed == False or self.drone_state.position[2] < 0.3:  # locked or unsuccessful takeoff
            self.reset()

    @staticmethod
    def random_action():
        action = [np.random.uniform(-1,1), np.random.uniform(-1,1)]
        return action

    "Move mode"
    def step(self, action):
        self.unpause()

        if self.aDRL_TEST and self.i_rdp == len(self.path_rdp) - 1 and self.test_env != 'Env1':
            dx = self.goal[0] - self.drone_state.position[0]
            dy = self.goal[1] - self.drone_state.position[1]
            angle1 = math.atan2(dy, dx)
            action[0] = math.cos(angle1)
            action[1] = math.sin(angle1)

        cur_target = ControlCommand()
        cur_target.Mode = 4
        cur_target.Command_ID = self.comid
        cur_target.Reference_State.Move_mode = 0b10
        cur_target.Reference_State.Move_frame = 0
        cur_target.Reference_State.velocity_ref[0] = action[0] * MAXVEL #(action[0] + 1)/2 * MAXVEL  # [0,MAXVEL]
        cur_target.Reference_State.velocity_ref[1] = action[1] * MAXVEL  # [-MAXVEL,MAXVEL]
        cur_target.Reference_State.position_ref[2] = self.goal[2]
        self.comid += 1
        self.move_pub.publish(cur_target)    
        time.sleep(0.1/self.sim_speed)
        
        self.pause()

    "Idle mode, motor idles, waiting for control commands from a higher level"
    def idle(self):
        cur_target = ControlCommand()
        cur_target.Mode = 0
        cur_target.Command_ID = self.comid
        self.move_pub.publish(cur_target)
        self.comid += 1
        time.sleep(1.0/self.sim_speed)

    "Unlock and switch to offboard mode"
    def arm_offboard(self):
        cur_target = ControlCommand()
        cur_target.Mode = 0
        cur_target.Command_ID = self.comid
        cur_target.Reference_State.yaw_ref = 999
        self.move_pub.publish(cur_target)
        cur_target.Reference_State.yaw_ref = 0
        self.comid += 1
        time.sleep(1.0/self.sim_speed)

    "Takeoff mode, take off to a height of 1 meter"
    def takeoff(self):
        cur_target = ControlCommand()
        cur_target.Mode = 1
        cur_target.Command_ID = self.comid
        self.move_pub.publish(cur_target)
        self.comid += 1
        time.sleep(5.0/self.sim_speed)

    "Hold mode"
    def hold(self):
        cur_target = ControlCommand()
        cur_target.Mode = 2
        cur_target.Command_ID = self.comid
        self.move_pub.publish(cur_target)
        self.comid += 1
        time.sleep(1.0/self.sim_speed)

    "Land mode"
    def land(self):
        self.unpause()
        cur_target = ControlCommand()
        cur_target.Mode = 3
        cur_target.Command_ID = self.comid
        self.move_pub.publish(cur_target)
        self.comid += 1
        time.sleep(10.0/self.sim_speed)

    "Disarm mode"
    def disarm(self):
        cur_target = ControlCommand()
        cur_target.Mode = 5
        cur_target.Command_ID = self.comid
        self.move_pub.publish(cur_target)
        self.comid += 1
    
    def step_reward(self):
        reward = 0
        if min(self.laser.ranges) < obs_safe_dis:
            reward -= 0.1 * (obs_safe_dis-min(self.laser.ranges))
        reward -= 0.02
        reward -= 0.1 * (self.dis_to_goal/MAXDIS)
        reward -= 0.1 * (self.dis_to_goal/MAXDIS) * (self.angle/math.pi)
        
        return reward

    "reward and done"
    def get_reward(self):
        if self.aDRL_TEST:  # DGlobal test
            if min(self.laser.ranges) < drone_r:  # collide with obstacles
                reward = -10.0
                done = True
            elif self.dis_to_goal < goal_r + drone_r and self.i_rdp == len(self.path_rdp) - 1:  # reach the goal point
                reward = 20.0
                done = True
            else:
                reward = self.step_reward()
                done = False
        else:  # TD3 training and test
            if min(self.laser.ranges) < drone_r:  # collide with obstacles
                reward = -10.0
                done = True
            elif self.dis_to_goal < goal_r + drone_r:  # reach the goal point
                reward = 20.0
                done = True
            else:
                reward = self.step_reward()
                done = False
    
        return reward/5, done

    "observation"
    def get_env(self):
        # record the trajectory
        if self.aDRL_TEST or self.oDRL_TEST:
            self.traj_x.append(self.drone_state.position[0])
            self.traj_y.append(self.drone_state.position[1])
        # determine whether to update the goal
        if self.aDRL_TEST:
            # proposed goal-updating
            i_temp = self.i_rdp
            while i_temp < len(self.path_rdp)-1:
                dis_to_waypoint = math.sqrt((self.path_rdp[i_temp][0]-self.drone_state.position[0])**2 +
                                            (self.path_rdp[i_temp][1]-self.drone_state.position[1])**2)
                if dis_to_waypoint < waypoint_r + drone_r:
                    self.i_rdp = i_temp + 1
                i_temp += 1

            dis_to_goal = math.sqrt((self.path_rdp[self.i_rdp][0] - self.drone_state.position[0]) ** 2 +
                                    (self.path_rdp[self.i_rdp][1] - self.drone_state.position[1]) ** 2)
            if min(self.laser.ranges) < drone_r + epsilon and dis_to_goal < waypoint_r + drone_r + epsilon:
                self.i_rdp += 1

            # # traditional goal-updating
            # dis_to_waypoint = math.sqrt((self.path_rdp[self.i_rdp][0] - self.drone_state.position[0]) ** 2 +
            #                             (self.path_rdp[self.i_rdp][1] - self.drone_state.position[1]) ** 2)
            # if dis_to_waypoint < waypoint_r + drone_r and self.i_rdp != len(self.path_rdp) - 1:
            #     self.i_rdp += 1

            self.goal = self.path_rdp[self.i_rdp]

        observation = []
        # the relative position between drone and goal/waypoint
        dx = self.goal[0] - self.drone_state.position[0]
        dy = self.goal[1] - self.drone_state.position[1]
        # dz = self.goal[2] - self.drone_state.position[2]
        observation.append(dx/MAX_X)
        observation.append(dy/MAX_Y)
#        observation.append(dz)
        print(self.goal)
        
        # dis between drone and goal/waypoint
        self.dis_to_goal = math.sqrt(dx**2 + dy**2)
        observation.append(self.dis_to_goal/MAXDIS)
            
#        # drone velocity
#        observation.append(self.drone_state.velocity[0] / MAXVEL)
#        observation.append(self.drone_state.velocity[1] / MAXVEL)
        
        # angle between the direction of drone velocity and the line from drone to goal/waypoint
        angle1 = math.atan2(dy, dx) 
        angle2 = math.atan2(self.drone_state.velocity[1], self.drone_state.velocity[0]) 
        if angle1*angle2 >= 0:
            self.angle = abs(angle1-angle2)
        else:
            self.angle = abs(angle1) + abs(angle2)
            if self.angle > math.pi:
                self.angle = 2*math.pi - self.angle
        observation.append(self.angle/math.pi)
        
        # Lidar (specific settings in hokuyo_ust_20lx.sdf)
        for i in range(len(self.laser.ranges)):
            temp = self.laser.ranges[i]
            if temp > MAXLASERDIS:
                temp = MAXLASERDIS
            observation.append(temp/MAXLASERDIS)

        reward, done = self.get_reward()

        return observation, reward, done

    def save_traj(self):
        traj_curve = pd.DataFrame({'traj_x': self.traj_x, 'traj_y': self.traj_y})
        traj_curve.to_csv('/xxx/TD3/result/traj.csv')
        print('Record the final trajectory')

if __name__=="__main__":
    env = envmodel(sim_speed=1, aDRL_TEST=1, oDRL_TEST=0, test_env='Env1')
    np.random.seed(7)
    
    MAX_EPISODES = 50
    MAX_STEPS = 500
    
    for i_episode in range(1, MAX_EPISODES+1):
        env.reset()
        env.check_fail()
        episode_rewards = 0
        d = False
        for step in range(MAX_STEPS):
            a = env.random_action()
            env.step(a)
            next_state, r, d = env.get_env()
            episode_rewards += r
            if d or step == MAX_STEPS-1:
                env.land()
                break
        print("\nTotal reward this episode: {}".format(episode_rewards))
