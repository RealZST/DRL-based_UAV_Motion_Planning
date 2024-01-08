# DRL-based_UAV_Motion_Planning
code for `[A Hybrid Human-in-the-Loop Deep Reinforcement Learning Method for UAV Motion Planning](https://scholar.google.com.hk/scholar?hl=zh-CN&as_sdt=0,5&q=A+Hybrid+Human-in-the-Loop+Deep+Reinforcement+Learning+Method+for+UAV+Motion+Planning+for+Long+Trajectories+with+Unpredictable+Obstacles&btnG=)'

# Dependencies
Ubuntu 20.04 LTS  
ROS Noetic  
gazebo 11  
Python 3.8

# Installation

This project is based on the project [Prometheus](https://github.com/amov-lab/Prometheus). To install it, please follow the instructions in [Prometheus wiki](https://github.com/amov-lab/Prometheus/wiki).  
This project is also based on the TD3 in the project [Policy-Gradient-Methods](https://github.com/cyoon1729/Policy-Gradient-Methods).

Notice:  
the guide [Installation and Compilation](https://github.com/amov-lab/Prometheus/wiki/%E5%AE%89%E8%A3%85%E5%8F%8A%E7%BC%96%E8%AF%91) here recommends installing Ubuntu 18.04 and ROS Melodic. However, Ubuntu 20.04 and ROS Noetic are also compatible. To use them, you may need to make modifications in the following places.  
<img src=https://github.com/RealZST/DRL-based_UAV_Motion_Planning/assets/53246001/ed7125ed-2a6b-4be0-a4e9-9e9fdd234823 width=48% />
<img src=https://github.com/RealZST/DRL-based_UAV_Motion_Planning/assets/53246001/0c74a8f5-64a7-42a9-a809-fc5b4ba6840e width=48% />
<img src=https://github.com/RealZST/DRL-based_UAV_Motion_Planning/assets/53246001/d49fd607-5fb3-4443-a0f2-8541e74f0cfd width=48% />
<img src=https://github.com/RealZST/DRL-based_UAV_Motion_Planning/assets/53246001/1a2397a7-9298-4b0f-a0a2-b0dc93a43872 width=48% />

During the DRL training process, resetting the drone's position at the beginning of each episode can cause issues with the onboard sensors. Modifying the following two places (in `Prometheus/Modules/control/src/px4_sender.cpp`) might be helpful:
<img src=https://github.com/RealZST/DRL-based_UAV_Motion_Planning/assets/53246001/fa587870-8ae3-4135-9847-e95d1679f289 width=48% />
<img src=https://github.com/RealZST/DRL-based_UAV_Motion_Planning/assets/53246001/47c68544-3183-48ca-b9ba-45dd1a1e764a width=38% />

I have made revisions to several files (such as sitl.launch, models, worlds, etc.) in [gazebo_simulator](https://github.com/amov-lab/Prometheus/tree/main/Simulator/gazebo_simulator) to accommodate DRL training. `gazebo_simulator` that I used has been uploaded here for your reference.

# How to run
reference: [Gazebo simulation run test](https://github.com/amov-lab/Prometheus/wiki/%E4%BB%BF%E7%9C%9F%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE#gazebo%E4%BB%BF%E7%9C%9F%E8%BF%90%E8%A1%8C%E6%B5%8B%E8%AF%95)
* Run the command in the terminal `roslaunch prometheus_gazebo sitl.launch`
* Run train.py
