# Week 1-2 Instructions

## Introduction
We will use the [PenguinPi robot](https://cirrusrobotics.com.au/products/penguinpi/) in the [Gazebo simulator](http://gazebosim.org/) for our lab project. 

[ECE4078LabProject.pdf](ECE4078LabProject.pdf) provides an overview of the lab project and the six marking milestones.

## Objectives
1. Get an overview of the lab project.
2. Set up your environment locally or on cloud-based platforms.
3. Implement keyboard teleoperation of the robot.

## Marking schemes
- Teleoperation (80pts): 60pts for implementing robot teleoperation (drive foward, drive backward, turn left, turn right, stop); 20pts for adding any additional controls.
- Interface (20pts): 15pts for showing a live camera feed; 5pts for displaying any additional information.

## Getting-started
1. [GazeboPenguinPiStetup.md](GazeboPenguinPiStetup.md) provides instructions on setting up the simulator environment on cloud-based platforms or locally, as well as connecting to the physical robot.
2. Clone the repo by typing ```git clone https://github.com/tianleimin/ECE4078_Lab.git``` inside your terminal.
3. Testing the set-up:

    After setting up your environment and cloning the repo, open a terminal to launch the simulation environment by typing the commands:
    ```
    source ~/catkin_ws/devel/setup.bash
    roslaunch penguinpi_gazebo penguinpi.launch
    ```

    Then open another terminal window and run [test_camera_motors.py](test_camera_motors.py) by typing the commands:
    ```
    cd ECE4078_Lab/Week01-02/
    python3 test_camera_motors.py
    ```

    This python script rotates the robot's wheels and shows an image captured by its camera. You should see the robot inside the simulator moving and then see a pop-up window of its camera input.
4. Implementing keyboard teleoperation: 
    [penguinPiC.py](penguinPiC.py) provides functions needed for accessing the robot's wheels and camera through a webserver. 
    [keyboardControlStarter.py](keyboardControlStarter.py) provides skeleton codes that you may start your implementation of keyboard teleoperation with. 
    **You don't have to use penguinPiC.py or keyboardControlStarter.py.** Feel free to be creative and write your own scripts for teleoperating the robot with keyboard.
