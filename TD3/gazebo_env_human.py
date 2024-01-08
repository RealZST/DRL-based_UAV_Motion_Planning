#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 20:41:26 2021

@author: zst
"""

"Env file, actions controlled by keyboard input"

import rospy
import numpy as np
import math
import time
import sys, select, termios, tty
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import SetModelState
from tf.transformations import quaternion_from_euler
from prometheus_msgs.msg import ControlCommand, DroneState
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import LaserScan
from points_pos import points_pos

sim_speed = 1  # simulation acceleration, corresponding to PX4_SIM_SPEED_FACTOR in sitl.launch

points = points_pos()
MAX_X = 20
MAX_Y = 12
MAXDIS = math.sqrt(MAX_X**2 + MAX_Y**2)  # maximum distance to the goal, used for normalization
MAXLASERDIS = 5.0 
MAXVEL = 2.0  # Maximum horizontal flight speed of the drone, corresponding to the MPC_XY_VEL_MAX setting in QGroundControl
obs_safe_dis = 1  # Safe distance to obstacles
drone_r = 0.277*math.sqrt(2)  # Drone radius  # 0.277*math.sqrt(2)=0.39174
obs_r = 0.5  # Obstacle radius, corresponding to the parameters in obs_cylinder file
obs_n = 10  # num of obstacles
goal_r = 0.3  # radius of the goal circle
#VEL_XY_STEP_SIZE = 0.1/MAXVEL  # the speed changes 0.1m/s per command

class envmodel:
    
    def __init__(self):
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
        self.set_state_service = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)

        self.start = []
        self.goal = []
        self.obs_pos = []
        self.comid = 0
        self.vel_ref_x = 0
        self.vel_ref_y = 0
        self.dis_to_goal = 0
        self.angle = 0

        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        
        time.sleep(0.5)
        
    def drone_state_callback(self, msg):
        self.drone_state = msg
        
    def laser_callback(self, msg):
        self.laser = msg
        
    def model_states_callback(self, msg):
        self.model_states = msg
    
    def reset(self):
        self.start, self.goal = points.waypoints_ori()
        self.obs_pos = points.obspoints_ori(obs_n, obs_r)
        self.comid = 0
        self.vel_ref_x = 0
        self.vel_ref_y = 0
        
        "Reset the starting point, target point, and drone position"
        for i in range(len(self.model_states.name)):
            if self.model_states.name[i] == 'start_point':
                self.state.reference_frame = 'world'
                self.state.model_name = self.model_states.name[i]
                self.state.pose.position.x = self.start[0]
                self.state.pose.position.y = self.start[1]
                self.state.pose.position.z = self.start[2]
                self.set_state_service(self.state)
            if self.model_states.name[i] == 'goal_point':
                self.state.reference_frame = 'world'
                self.state.model_name = self.model_states.name[i]
                self.state.pose.position.x = self.goal[0]
                self.state.pose.position.y = self.goal[1]
                self.state.pose.position.z = self.goal[2]
                self.set_state_service(self.state)
            if self.model_states.name[i] == 'p450_hokuyo_2Dlidar':
                self.state.reference_frame = 'world'
                self.state.model_name = self.model_states.name[i]
                self.state.pose.position.x = self.start[0]
                self.state.pose.position.y = self.start[1]
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
                time.sleep(5.0/sim_speed)
        
        "Reset the obstacle positions"
        for k in range(11):
            NAME_OBS = 'obs' + str(k)
            self.state.model_name = NAME_OBS
            self.state.reference_frame = 'world'
            self.state.pose.position.x = -20
            self.state.pose.position.y = -10 + 2*k
            self.state.pose.position.z = 1
            self.set_state_service(self.state)
        for k in range(len(self.obs_pos)):
            NAME_OBS = 'obs' + str(k)
            self.state.model_name = NAME_OBS
            self.state.reference_frame = 'world'
            self.state.pose.position.x = self.obs_pos[k][0]
            self.state.pose.position.y = self.obs_pos[k][1]
            self.state.pose.position.z = self.obs_pos[k][2]
            self.set_state_service(self.state) 
        
        self.arm_offboard()
        self.takeoff()
        
    def check_fail(self):    
        while self.drone_state.armed == False or self.drone_state.position[2] < 0.3:  # locked or unsuccessful takeoff
            self.reset()

    "Move mode"
    def step(self, action):
        cur_target = ControlCommand()
        cur_target.Mode = 4
        cur_target.Command_ID = self.comid
        cur_target.Reference_State.Move_mode = 0b10
        cur_target.Reference_State.Move_frame = 0
        cur_target.Reference_State.velocity_ref[0] = action[0] * MAXVEL 
        cur_target.Reference_State.velocity_ref[1] = action[1] * MAXVEL  
        cur_target.Reference_State.position_ref[2] = self.goal[2]
        self.comid += 1
        self.move_pub.publish(cur_target)    
        time.sleep(0.1/sim_speed) 

    "Idle mode, motor idles, waiting for control commands from a higher level"
    def idle(self):
        cur_target = ControlCommand()
        cur_target.Mode = 0
        cur_target.Command_ID = self.comid
        self.move_pub.publish(cur_target)
        self.comid += 1
        time.sleep(1.0/sim_speed)

    "Unlock and switch to offboard mode"
    def arm_offboard(self):
        cur_target = ControlCommand()
        cur_target.Mode = 0
        cur_target.Command_ID = self.comid
        cur_target.Reference_State.yaw_ref = 999
        self.move_pub.publish(cur_target)
        cur_target.Reference_State.yaw_ref = 0
        self.comid += 1
        time.sleep(1.0/sim_speed)

    "Takeoff mode, take off to a height of 1 meter"
    def takeoff(self):
        cur_target = ControlCommand()
        cur_target.Mode = 1
        cur_target.Command_ID = self.comid
        self.move_pub.publish(cur_target)
        self.comid += 1
        time.sleep(3.0/sim_speed)

    "Hold mode"
    def hold(self):
        cur_target = ControlCommand()
        cur_target.Mode = 2
        cur_target.Command_ID = self.comid
        self.move_pub.publish(cur_target)
        self.comid += 1
        time.sleep(1.0/sim_speed)

    "Land mode"
    def land(self):
        cur_target = ControlCommand()
        cur_target.Mode = 3
        cur_target.Command_ID = self.comid
        self.move_pub.publish(cur_target)
        self.comid += 1
        time.sleep(5.0/sim_speed)

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
        "normalize the distance"
        reward -= 0.1 * (self.dis_to_goal/MAXDIS)
        reward -= 0.1 * (self.dis_to_goal/MAXDIS) * (self.angle/math.pi)
        
        return reward
    
    def get_reward(self):
        if min(self.laser.ranges) < drone_r:  # collide with obstacles
            print(min(self.laser.ranges))
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
    def read_get_env(self):
        observation = []
        # the relative position between drone and goal
        dx = self.goal[0] - self.drone_state.position[0]
        dy = self.goal[1] - self.drone_state.position[1]
#        dz = self.goal[2] - self.drone_state.position[2]
        observation.append(dx/MAX_X)
        observation.append(dy/MAX_Y)
#        observation.append(dz)
        
        # dis between drone and goal 
        self.dis_to_goal = math.sqrt(dx**2 + dy**2)
        observation.append(self.dis_to_goal/MAXDIS)
        
#        # drone velocity
#        observation.append(self.drone_state.velocity[0] / MAXVEL)
#        observation.append(self.drone_state.velocity[1] / MAXVEL)
        
        # angle between the direction of drone velocity and the line from drone to goal
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
        
        "reward and done"
        reward, done = self.get_reward()  
        
        return observation, reward, done

    @staticmethod
    def get_key():
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        
        return key
    
    def key_to_action(self, key):
        if key == 'w'or key == 'W':
            self.vel_ref_x = 1  #min(1, self.vel_ref_x + VEL_XY_STEP_SIZE) #1
        elif key == 's'or key == 'S':
            self.vel_ref_x = -1  #max(-1, self.vel_ref_x - VEL_XY_STEP_SIZE) #-1
        elif key == 'j'or key == 'J':
            self.vel_ref_y = 1  #min(1, self.vel_ref_y + VEL_XY_STEP_SIZE) #1
        elif key == 'l'or key == 'L':
            self.vel_ref_y = -1  #max(-1, self.vel_ref_y - VEL_XY_STEP_SIZE) #-1
        elif key == 'k'or key == 'K':
            self.vel_ref_y = 0
        vel_ref = [self.vel_ref_x, self.vel_ref_y]
        print('Current control speed :',[round(i*MAXVEL,2) for i in vel_ref],'m/s')
        
        return vel_ref 
    
    def record_human_data(self, state, action, reward, next_state, done, step):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        if done:
            if reward > 0:
                np.savez('human_data.npz', s=self.states, a=self.actions, r=self.rewards, n_s=self.next_states, d=self.dones)
                print('SAVE. Len of Human Data :', len(self.states))
            else:
                del self.states[-(step+1):]
                del self.actions[-(step+1):]
                del self.rewards[-(step+1):]
                del self.next_states[-(step+1):]
                del self.dones[-(step+1):]
                print('DELETE. Len of Human Data :', len(self.states))
            if len(self.states) > 2000: # consistent with the setting of self.buffer_human = deque(maxlen=2000)
                print('Human data is enough')
        
        
if __name__ == "__main__":
    settings = termios.tcgetattr(sys.stdin)
    env = envmodel()
    MAX_EPISODES = 1000
    MAX_STEPS = 1000
    scores = []
    success = []
    collision = []
    lost = []
    text = None
    
    for i_episode in range(1, MAX_EPISODES+1):
        env.reset()
        env.check_fail()
        state, _, _ = env.read_get_env()
        episode_rewards = 0
        d = False
        
        for step in range(MAX_STEPS):
            key = env.get_key()
            a = env.key_to_action(key)
            env.step(a)
            next_state, r, d = env.read_get_env()
            env.record_human_data(state, a, r, next_state, d, step)
            episode_rewards += r
            
            if d or step == MAX_STEPS-1:
                env.land()
                text = 'Success' if r>0 else 'Failed'
                if d:
                    success += [r>0]
                    collision += [r<0]
                    lost.append(0)
                else:
                    success.append(0)
                    collision.append(0)
                    lost.append(1)
                break
            
            state = next_state  
            
        scores += [episode_rewards]
        print('\nEpisode', i_episode, text,
              ', Total Reward=%.2f' % episode_rewards,
              ', Success rate=%.5f' % np.mean(success[:]),
              ', Collision rate=%.5f' % np.mean(collision[:]),
              ', Lost rate=%.5f' % np.mean(lost[:]), end="")

        
        
        