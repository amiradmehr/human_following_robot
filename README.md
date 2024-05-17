Human Tracking Robot Perception Stack
=====================================

This repository contains the code for human following robot perception stack for triton robot. The perception stack is designed to enable a robot to track a specific individual in a crowded environment. The perception stack consists of two main components: human detection and instance segmentation using YOLOv7, and feature extraction and matching using SuperPoint.

Dependencies
------------

The following dependencies are required to run the code:

-   PyTorch
-   OpenCV
-   NumPy 


Create a virtual environment in the root directory:

`pip install virtualenv`

`virtualenv <env name>`

To install the dependencies, run the following command:

`pip install -r requirements.txt`

To activate the environment:

`source <env name>/bin/activate`


Usage
-----

To use the perception stack, run the following commands:

`python3 scripts/main.py`

Clone the repository in your catkin workspace:
`cd catkin_ws/src`

`git clone https://github.com/amiradmehr/human_following_robot`

`cd ..`

`catkin_make`

`roscd human_following_robot/scripts`

`chmod +x main.py`

`source devel/setup.bash`

First you need to run the launch file on the jetson nano:

`export ROS_MASTER_URI=http://<jetson_ip>:11311`

`export ROS_IP=<jetson_ip>`

`roslaunch stingray_camera triton.launch`

Now on your local machine:

`export ROS_MASTER_URI=http://<jetson_ip>:11311`

`export ROS_IP=<local_ip>`

`rosrun human_following_robot main.py`

Algorithms used
---------------

The perception stack utilizes the following algorithms:

-   YOLOv7 for human detection and instance segmentation
-   SuperPoint for feature extraction and matching