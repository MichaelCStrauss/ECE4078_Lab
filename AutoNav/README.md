# Group 2_12 Milestone Two Submission

This is our submission for the fourth milestone. We have used the SLAM code from M2 and improved it for autonomous pathfinding.

# Running Instructions

Running our code for this milestone should be simple:

Simply run the file `autonav.py`: as `python3 autonav.py`.

We have not modified the SDF files.

The robot will then begin to semi-autonomously navigate the arena with prompts, and produce an output file `map.txt` in the same output format as the reference, with marker locations. Select 'drive' after the first full 360 degree spin to manually navigate using the keyboard.

NOTE: Running the YOLO network currently uses CUDA. Modify autonav.py line 89 to say "cpu" instead of CUDA to use CPU only.

# Results

Please see the two screen recordings of three runs at [this link](https://drive.google.com/drive/folders/1kcwVvzPDUJOjCZqao_3vvjZdrhVOiIoy?usp=sharing).

These three runs all score 94-96% (10-12/16 on marker locations, 100% on marker detection, paths and time.)

Please note that our implementation *does not* depend on the system FPS, and only uses the *real time elapsed*. Therefore, the multiplication based on system FPS should not need to apply to our runs. However, these runs all run at 60 FPS on Gazebo.

# Issues

All testing has been conducted with a system running native Ubuntu at 60 FPS in Gazebo. We have noticed that there can be some issues when the system runs at 30 FPS, as the motion model does not account for the slower run time in Gazebo as accurately. 

If this issue occurs during the run, the marker estimations and path distances are likely to be very bad. I would ask if you might schedule a meeting with me (Michael) to run the demo quickly on my own computer to show that it works! We have worked really hard on this, and would love the chance to show properly how it performs, like how it'll work in the final demo.