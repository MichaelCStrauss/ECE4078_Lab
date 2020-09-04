# Group 2_12 Milestone Two Submission

This is our submission for the second mileston, SLAM. 

## Notes

Please see the results of two successful (100%) runs, where all markers were detected within 1.5m. After tweaking calibration and covariances, we were able to get this working reasonably consistently, I hope you are able too as well :) 

These results were achieved by performing the following driving techniques: 
1. Initially, ensure that there is one marker in view ASAP. 
2. If possible, find a second marker that is able to be captured in the same frame as the first. This allows localisation with two markers.
3. When turning, try and stop every 45 degrees, otherwise the robot will slip and not stop moving quickly. Apologies for slow turning speed.
4. When driving, attempt to only view a marker when it reasonably close (not always possible and not hugely detrimental)

## Getting the code running

### **IMPORTANT**

**Please copy the `models/penguinpi.sdf` file into your workspace `workspace/src/penguinpi_description/urdf/penguinpi.sdf`. This file has been calibrated with `calibration/baseline.txt` and `calibration/scale.txt`.**

The only other change that is required is to modify the port in PenguiPiC.py. On my computer it is `8080`, however I think it is `40000` on the Ubuntu image
