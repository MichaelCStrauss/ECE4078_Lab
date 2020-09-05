# Group 2_12 Milestone Two Submission

This is our submission for the second milestone, SLAM. 

## Notes

Please see the results of four successful (90-100%) runs, where all markers were detected within 1.5m except for one run.

We changed the handling of `dt` in the prediction step. After noticing that the predictions depended on how quickly the code was running on the computer (e.g. if `predict` is called 100 times per second vs 10 times per second, the motion model moves the robot 10 times further), we included the real `dt` value from the system clock. After tweaking calibration and covariances, we were able to get this working reasonably consistently, I hope you are able too as well :) 

These results were achieved by performing the following driving techniques: 
1. Initially, ensure that there is one marker in view ASAP. 
2. If possible, find a second marker that is able to be captured in the same frame as the first. This allows localisation with two markers.
3. When turning, try and stop every 45 degrees, otherwise the robot will slip and not stop moving quickly. Apologies for slow turning speed.
4. When driving, attempt to only view a marker when it reasonably close (not always possible and not hugely detrimental)

We noticed that infrequently, when viewing an Aruco marker, the SLAM will completely glitch and 'teleport' the robot. Our hypothesis is that the incorrect marker ID is detected, or there is some kind of singularity in the covariance which rapidly moves the bot. If this is happens, please restart the SLAM.

## Getting the code running

### **IMPORTANT**

**Please copy the `models/penguinpi.sdf` file into your workspace `workspace/src/penguinpi_description/urdf/penguinpi.sdf`. This file has been calibrated with `calibration/baseline.txt` and `calibration/scale.txt`.**

**Additionally, please observe your `real time factor` in Gazebo, and adjust the variable on line 93 of `manualSLAM.py` to match. For us, it was 0.5.**

The only other change that is required is to modify the port in PenguiPiC.py. On my computer it is `8080`, however I think it is `40000` on the Ubuntu image
