# Manually drive the robot inside the arena and perform SLAM using ARUCO markers

# Import packages
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import json
import math

# Import keyboard teleoperation components
import penguinPiC
import keyboardControlStarter as Keyboard

# Import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
import slam.Slam as Slam
import slam.Robot as Robot
import slam.aruco_detector as aruco

import cv2
import cv2.aruco as cv2_aruco
import slam.Measurements as Measurements
import time


# camera calibration parameters (from M2: SLAM)
camera_matrix = np.loadtxt(
    "calibration/camera_calibration/intrinsic.txt", delimiter=","
)
dist_coeffs = np.loadtxt("calibration/camera_calibration/distCoeffs.txt", delimiter=",")
marker_length = 0.1

# wheel calibration parameters (from M2: SLAM)
wheels_scale = np.loadtxt("calibration/wheel_calibration/scale.txt", delimiter=",")
wheels_width = np.loadtxt("calibration/wheel_calibration/baseline.txt", delimiter=",")

# display window for visulisation
cv2.namedWindow("video", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("video", cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)
# font display options
font = cv2.FONT_HERSHEY_SIMPLEX
location = (0, 0)
font_scale = 1
font_col = (255, 255, 255)
line_type = 2

valid_marker_ids = [1, 3, 7, 8, 9, 11, 12, 21, 39]


# Manual SLAM
class Operate:
    def __init__(self, datadir, ppi):
        # Initialise
        self.ppi = ppi
        self.ppi.set_velocity(0, 0)
        self.img = np.zeros([240, 320, 3], dtype=np.uint8)
        self.aruco_img = np.zeros([240, 320, 3], dtype=np.uint8)

        # Keyboard teleoperation components
        self.keyboard = Keyboard.Keyboard(self.ppi)

        # Get camera / wheel calibration info for SLAM
        camera_matrix, dist_coeffs, scale, baseline = self.getCalibParams(datadir)

        # SLAM components
        self.pibot = Robot.Robot(baseline, scale, camera_matrix, dist_coeffs)
        self.aruco_det = aruco.aruco_detector(self.pibot, marker_length=0.1)
        self.slam = Slam.Slam(self.pibot)

        self.markers_travelled_to = []
        self.paths = []
        self.current_marker = None
        self.spinning = True
        self.frames = 0

    # def __del__(self):
    # self.ppi.set_velocity(0, 0)

    def getCalibParams(self, datadir):
        # Imports camera / wheel calibration parameters
        fileK = "{}camera_calibration/intrinsic.txt".format(datadir)
        camera_matrix = np.loadtxt(fileK, delimiter=",")
        fileD = "{}camera_calibration/distCoeffs.txt".format(datadir)
        dist_coeffs = np.loadtxt(fileD, delimiter=",")
        fileS = "{}wheel_calibration/scale.txt".format(datadir)
        scale = np.loadtxt(fileS, delimiter=",")
        fileB = "{}wheel_calibration/baseline.txt".format(datadir)
        baseline = np.loadtxt(fileB, delimiter=",")

        return camera_matrix, dist_coeffs, scale, baseline

    def get_camera(self):
        # get current frame
        curr = self.ppi.get_image()

        # visualise ARUCO marker detection annotations
        aruco_params = cv2_aruco.DetectorParameters_create()
        aruco_params.minDistanceToBorder = 0
        aruco_params.adaptiveThreshWinSizeMax = 1000
        aruco_dict = cv2_aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)

        corners, ids, rejected = cv2_aruco.detectMarkers(
            curr, aruco_dict, parameters=aruco_params
        )
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_length, camera_matrix, dist_coeffs
        )

        return corners, ids, rejected, rvecs, tvecs
    
    def pause(self, pause_time=0.5):
        time_start = time.time()
        self.time_prev = time.time()
        real_time_factor = 0.5

        while time.time() - time_start < pause_time:
            time_now = time.time()
            dt = time_now - self.time_prev
            dt *= real_time_factor
            self.time_prev = time_now
            self.ppi.set_velocity(0, 0)
            self.step(0, 0, dt)

    def spinOneRotation(self):
        # spinning and looking for markers at each step
        wheel_vel = 28

        self.frames += 1

        spin = True
        spin_time = 5
        fps = 30
        measurements = []
        seen_ids = set()
        moved_past_first = False

        real_time_factor = 0.5
        self.time_prev = time.time()
        initial_theta = self.slam.get_state_vector()[2]
        pause_theta = initial_theta
        while spin:
            time_now = time.time()
            dt = time_now - self.time_prev
            dt *= real_time_factor
            self.time_prev = time_now
            model_theta = self.slam.get_state_vector()[2]
            lv, rv = -wheel_vel, wheel_vel

            print(f"{seen_ids=}")

            corners, ids, rejected, rvecs, tvecs = self.get_camera()
            if ids is not None:
                ids_in_view = [ids[i, 0] for i in range(len(ids))]
                for id in ids_in_view:
                    if id in valid_marker_ids:
                        seen_ids.add(id)

            if model_theta - pause_theta > math.pi:
                self.pause(2)
                pause_theta = model_theta
            if model_theta - initial_theta > 2 * math.pi:
                spin = False

            self.step(lv, rv, dt)
        self.ppi.set_velocity(0, 0, 0.5)

        # Save the paths
        position = self.slam.robot.state[0:2]
        for marker_id in seen_ids:
            if marker_id == self.current_marker:
                continue
            try:
                marker_location = self.get_marker_location(marker_id)
            except ValueError:
                print(f"{marker_id=} value error")
                continue
            dist = (marker_location[0] - position[0]) ** 2 + (
                marker_location[1] - position[1]
            ) ** 2
            dist = dist ** 0.5
            current = (
                str(self.current_marker) if self.current_marker is not None else "start"
            )
            self.paths.append((current, str(marker_id), str(float(dist))))
        
        return seen_ids

    def get_marker_location(self, marker_id):
        x_list, y_list = self.slam.markers.tolist()
        idx = self.slam.taglist.index(marker_id)
        return x_list[idx], y_list[idx]

    def spin_to_marker(self, goal_marker_id):
        real_time_factor = 0.5

        self.time_prev = time.time()

        model_theta = self.slam.get_state_vector()[2]
        pause_theta = model_theta
        while True:
            time_now = time.time()
            dt = time_now - self.time_prev
            dt *= real_time_factor
            self.time_prev = time_now
            wheel_vel = 28
            print(f"turning to target {goal_marker_id}")
            lv, rv = -wheel_vel, wheel_vel
            self.ppi.set_velocity(lv, rv)

            # Get the ids in view
            corners, ids, rejected, rvecs, tvecs = self.get_camera()

            self.step(lv, rv, dt)

            if model_theta - pause_theta > math.pi:
                self.pause(2)
                pause_theta = model_theta

            print(ids)
            if ids is None:
                continue
            ids_in_view = [ids[i, 0] for i in range(len(ids))]

            print(ids_in_view)
            if goal_marker_id in ids_in_view:
                break

        for _ in range(10):
            self.vision()
            print(self.slam.taglist)

    def drive_to_marker(self, goal_marker_id):
        real_time_factor = 0.5

        self.time_prev = time.time()

        target_location = self.get_marker_location(goal_marker_id)

        driving = True
        while driving:
            time_now = time.time()
            dt = time_now - self.time_prev
            dt *= real_time_factor
            self.time_prev = time_now
            wheel_vel = 40
            print(f"driving to target {goal_marker_id}")

            lv, rv = wheel_vel, wheel_vel

            corners, ids, rejected, rvecs, tvecs = self.get_camera()

            if ids is not None:
                for i in range(len(ids)):
                    idi = ids[i, 0]
                    # Some markers appear multiple times but should only be handled once.
                    if idi != goal_marker_id:
                        continue

                    avg_x = corners[i][0, :, 0].mean()
                    print(f"{avg_x=}")
                    diff_from_center = avg_x - 320
                    k = 0
                    lv, rv = (
                        wheel_vel + diff_from_center * k,
                        wheel_vel - diff_from_center * k,
                    )
                    lv, rv = int(lv), int(rv)
                    lv = np.clip(lv, -50, 50)
                    rv = np.clip(rv, -50, 50)

            position = self.slam.robot.state[0:2]
            dist = (target_location[0] - position[0]) ** 2 + (
                target_location[1] - position[1]
            ) ** 2
            dist = dist ** 0.5
            print(f"{dist=}")

            if dist < 0.75:
                driving = False

            self.step(lv, rv, dt)

        self.current_marker = goal_marker_id

    def get_closest_untravelled_marker(self, ids_in_view):
        x_list, y_list = self.slam.markers.tolist()
        position = self.slam.robot.state[0:2]
        min_dist = 1e9
        min_marker = None

        for idx, (marker_x, marker_y) in enumerate(zip(x_list, y_list)):
            marker = self.slam.taglist[idx]
            if marker in self.markers_travelled_to or marker not in ids_in_view:
                continue
            dist = (marker_x - position[0]) ** 2 + (marker_y - position[1]) ** 2
            dist = dist ** 0.5
            dist = float(dist)
            if dist < min_dist:
                min_dist = dist
                min_marker = marker

        print(f"{min_marker=}, {min_dist=}")
        return min_marker, min_dist

    def control(self, lv, rv, dt):
        # Import teleoperation control signals
        self.ppi.set_velocity(lv, rv)
        drive_meas = Measurements.DriveMeasurement(lv, rv, dt=dt)
        self.slam.predict(drive_meas)

    def vision(self):
        # Import camera input and ARUCO marker info
        self.img = self.ppi.get_image()
        lms, aruco_image = self.aruco_det.detect_marker_positions(self.img)
        self.slam.add_landmarks(lms)
        # print(f'{self.slam.taglist=}, {self.slam.markers=}')
        self.slam.update(lms)

    def display(self, fig, ax):
        # Visualize SLAM
        ax[0].cla()
        self.slam.draw_slam_state(ax[0])

        ax[1].cla()
        ax[1].imshow(self.img[:, :, -1::-1])

        plt.pause(0.01)

    def step(self, lv, rv, dt):
        self.control(lv, rv, dt)
        self.vision()

        # Save SLAM map
        self.write_map(self.slam)

        # Output visualisation
        self.display(self.fig, self.ax)

    def write_map(self, slam):
        map_f = "map.txt"
        marker_list = sorted(self.slam.taglist)
        with open(map_f, "w") as f:
            f.write("id, x, y\n")
            x_list, y_list = self.slam.markers.tolist()
            position = self.slam.robot.state[0:2]
            min_dist = 1e9
            min_marker = None

            lines = []
            for idx, (marker_x, marker_y) in enumerate(zip(x_list, y_list)):
                marker = self.slam.taglist[idx]
                lines.append(f"{marker}, {marker_x}, {marker_y}\n")
            lines = sorted(lines, key=lambda x: int(x.split(",")[0]))
            f.writelines(lines)

            f.write("\ncurrent id, accessible id, distance\n")
            for path in self.paths:
                line = ", ".join(path)
                f.write(line + "\n")

    def process(self):
        # Show SLAM and camera feed side by side
        self.fig, self.ax = plt.subplots(1, 2)
        img_artist = self.ax[1].imshow(self.img)

        # Run our code
        while True:
            seen_ids = self.spinOneRotation()
            goal_marker, goal_dist = self.get_closest_untravelled_marker(seen_ids)
            self.spin_to_marker(goal_marker)
            self.drive_to_marker(goal_marker)
            self.markers_travelled_to.append(goal_marker)


if __name__ == "__main__":
    # Location of the calibration files
    currentDir = os.getcwd()
    datadir = "{}/calibration/".format(currentDir)
    # connect to the robot
    ppi = penguinPiC.PenguinPi()

    # Perform Manual SLAM
    operate = Operate(datadir, ppi)
    operate.process()
