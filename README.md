Human Tracking Robot Perception Stack
=====================================

This repository contains the code for a perception stack designed and developed for a human tracking robot. The perception stack utilizes deep learning-based systems for person detection and re-identification.

Dependencies
------------

The following dependencies are required to run the code:

-   PyTorch
-   OpenCV
-   NumPy

Installation
------------
Create a virtual environment in the root directory:

`pip install virtualenv`

`virtualenv <env name>`

To activate the environment:

`source <env name>/bin/activate`

Clone the repository: 

`git clone https://github.com/Ashwij3/Human_following_robot.git`

To install the dependencies, run the following command:

`pip install -r requirements.txt`

Usage
-----

To use the perception stack, run the following command:

Copy code

`python3 scripts/main.py`

Algorithms used
---------------

The perception stack utilizes the following algorithms:

-   YOLOv7 for human detection and instance segmentation
-   SuperPoint for feature extraction and matching

Results
-------

The implemented perception stack achieves accurate identification of the target individual, allowing for effective human tracking by the robot.

Acknowledgments
---------------

The code in this repository is based on the following research papers:

-   [SuperPoint: Self-Supervised Interest Point Detection and Description](https://arxiv.org/abs/1712.07629)
-   [YOLO-v7](https://github.com/WongKinYiu/yolov7/tree/mask)
-   [Magic Leap](https://github.com/magicleap/SuperPointPretrainedNetwork)

