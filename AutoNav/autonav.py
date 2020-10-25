# Manually drive the robot inside the arena and perform SLAM using ARUCO markers

# TODO:
# - increase turning speed greatly
# - maybe increase forward speed (calibration)
# - add to readme that we don't use FPS and use real time instead
# - maybe investigate glitching??

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

from yolo import YoloV5
import warnings

warnings.filterwarnings("ignore")


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

valid_marker_ids = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
ip = [(None, 3), (17, 5)]
skip_survey = []


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
        self.markers_seen_at_step = []

        self.yolo = YoloV5("./weights.pt", "cuda")  # TODO: Fix device

        self.run_start = time.time()

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

    def pause(self, pause_time=0.5, speeds=None):
        time_start = time.time()
        self.time_prev = time.time()
        real_time_factor = 0.5

        if speeds is not None:
            self.ppi.set_velocity(speeds[0], speeds[1])
        while time.time() - time_start < pause_time:
            time_now = time.time()
            dt = time_now - self.time_prev
            dt *= real_time_factor
            self.time_prev = time_now
            self.ppi.set_velocity(0, 0)
            self.step(0, 0, dt)

    def rotate(self, model_theta, pause_theta, spin_direction=1):
        wheel_vel = 40
        d_theta = abs(model_theta - pause_theta)

        lv, rv = -wheel_vel, wheel_vel
        k = 40
        break_at = 2 * math.pi / 3
        reduction = (d_theta - break_at) * k if d_theta > break_at else 0
        b_l = -(wheel_vel - reduction)
        b_r = -b_l
        b_l, b_r = int(b_l), int(b_r)

        k2 = 17
        k3 = 12
        y_int = 10
        y_int2 = wheel_vel * 2
        model_vel = (
            y_int + k2 * d_theta if d_theta < math.pi / 2 else y_int2 - k3 * d_theta
        )
        m_l = -1 * min(wheel_vel, model_vel)
        m_r = -m_l
        m_l, m_r = int(m_l), int(m_r)

        b_l *= spin_direction
        b_r *= spin_direction
        m_l *= spin_direction
        m_r *= spin_direction

        return m_l, m_r, b_l, b_r

    def spinOneRotation(self):
        # spinning and looking for markers at each step
        wheel_vel = 50

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

            m_l, m_r, b_l, b_r = self.rotate(model_theta, pause_theta)

            print(f"{seen_ids=}")

            corners, ids, rejected, rvecs, tvecs = self.get_camera()
            if ids is not None:
                ids_in_view = [ids[i, 0] for i in range(len(ids))]
                for id in ids_in_view:
                    if id in valid_marker_ids:
                        seen_ids.add(id)

            if model_theta - pause_theta > math.pi:
                self.pause(2, (wheel_vel, -wheel_vel))
                pause_theta = model_theta
            if model_theta - initial_theta > 2 * math.pi:
                spin = False

            self.step(m_l, m_r, dt, bot_input=(b_l, b_r))

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
            if current != "start":
                self.paths.append((str(marker_id), current, str(float(dist))))

        return seen_ids

    def get_marker_location(self, marker_id):
        x_list, y_list = self.slam.markers.tolist()
        idx = self.slam.taglist.index(marker_id)
        return x_list[idx], y_list[idx]

    def spin_to_marker(self, goal_marker_id):
        real_time_factor = 0.5

        self.time_prev = time.time()
        model_theta = self.slam.get_state_vector()[2]
        while model_theta > math.pi:
            model_theta -= 2 * math.pi
        while model_theta < -math.pi:
            model_theta += 2 * math.pi

        try:
            marker_pos = self.get_marker_location(goal_marker_id)
            robot_pos = self.slam.robot.state[0:2]
            relative_angle = math.atan2(
                marker_pos[1] - robot_pos[1], marker_pos[0] - robot_pos[0]
            )
            delta = relative_angle - model_theta
            spin_direction = 1 if delta > 0 else -1
        except Exception as e:
            print(e)
            spin_direction = 1

        pause_theta = model_theta
        while True:
            time_now = time.time()
            dt = time_now - self.time_prev
            dt *= real_time_factor
            self.time_prev = time_now
            print(f"turning to target {goal_marker_id}")

            model_theta = self.slam.get_state_vector()[2]
            m_l, m_r, b_l, b_r = self.rotate(model_theta, pause_theta, spin_direction)
            self.step(m_l, m_r, dt, bot_input=(b_l, b_r))

            # Get the ids in view
            corners, ids, rejected, rvecs, tvecs = self.get_camera()

            if abs(model_theta - pause_theta) > math.pi:
                self.pause(2)
                pause_theta = model_theta

            print(ids)
            if ids is None:
                continue
            ids_in_view = [ids[i, 0] for i in range(len(ids))]

            print(ids_in_view)
            if goal_marker_id in ids_in_view:
                break

        self.pause(4)

        adjusting = True
        adjusting_ticks = 0
        while adjusting and adjusting_ticks < 30:
            time_now = time.time()
            dt = time_now - self.time_prev
            dt *= real_time_factor
            self.time_prev = time_now
            wheel_vel = 40
            print(f"adjusting to target {goal_marker_id}")

            lv, rv = 0, 0

            corners, ids, rejected, rvecs, tvecs = self.get_camera()

            if ids is not None:
                for i in range(len(ids)):
                    idi = ids[i, 0]
                    # Some markers appear multiple times but should only be handled once.
                    if idi != goal_marker_id:
                        continue

                    avg_x = corners[i][0, :, 0].mean()
                    diff_from_center = avg_x - 320
                    print(f"{diff_from_center=}")
                    k = 0.4
                    lv, rv = (
                        diff_from_center * k,
                        -diff_from_center * k,
                    )
                    lv, rv = int(lv), int(rv)
                    lv = np.clip(lv, -40, 40)
                    rv = np.clip(rv, -40, 40)
                    if abs(diff_from_center) < 10:
                        adjusting = False

            self.step(lv / 4, rv / 4, dt, bot_input=(lv, rv))
            adjusting_ticks += 1

        for _ in range(10):
            self.vision()
            print(self.slam.taglist)

    def drive_to_marker(self, goal_marker_id):
        real_time_factor = 0.5
        prev_dist = 1e6

        self.time_prev = time.time()

        target_location = self.get_marker_location(goal_marker_id)

        driving = True
        while driving:
            time_now = time.time()
            dt = time_now - self.time_prev
            dt *= real_time_factor
            self.time_prev = time_now
            wheel_vel = 65
            print(f"driving to target {goal_marker_id}")

            lv, rv = wheel_vel, wheel_vel
            b_lv, b_rv = lv, rv

            corners, ids, rejected, rvecs, tvecs = self.get_camera()

            position = self.slam.robot.state[0:2]
            dist = (target_location[0] - position[0]) ** 2 + (
                target_location[1] - position[1]
            ) ** 2
            dist = dist ** 0.5
            print(f"{dist=} {ids=}")

            threshold = 2
            if dist < threshold or dist > prev_dist:
                driving = False
            elif dist > 1:
                b_lv = 70
            prev_dist = dist

            # elif dist < 1.2:
            #     if ids is None:
            #         driving = False
            #     else:
            #         ids_in_view = [ids[i, 0] for i in range(len(ids))]
            #         if goal_marker_id not in ids_in_view:
            #             driving = False

            self.step(lv, rv, dt, bot_input=(b_lv, b_rv))

        self.current_marker = goal_marker_id

    def get_next_untravelled_marker(self, ids_in_view, mode="closes", filter=False):
        x_list, y_list = self.slam.markers.tolist()
        position = self.slam.robot.state[0:2]
        min_dist = 1e9
        min_marker = None
        max_dist = -1e9
        max_marker = None
        top_y = -1e9
        top_y_marker = None

        for idx, (marker_x, marker_y) in enumerate(zip(x_list, y_list)):
            marker = self.slam.taglist[idx]
            if marker in self.markers_travelled_to or marker not in ids_in_view:
                continue
            if filter:
                f = False
                for marker_set in self.markers_seen_at_step:
                    if marker in marker_set:
                        f = True
                if f:
                    continue
            if (self.current_marker, marker) in ip:
                continue
            dist = (marker_x - position[0]) ** 2 + (marker_y - position[1]) ** 2
            dist = dist ** 0.5
            dist = float(dist)
            if dist < min_dist:
                min_dist = dist
                min_marker = marker
            if dist > max_dist:
                max_dist = dist
                max_marker = marker
            if marker_y > top_y:
                top_y = marker_y
                top_y_marker = marker

        print(f"{min_marker=}, {min_dist=}")
        print(f"{max_marker=}, {max_dist=}")
        if mode == "closest":
            return min_marker, min_dist
        elif mode == "furthest":
            return max_marker, max_dist
        elif mode == "top_y":
            return top_y_marker, top_y

    def get_next_marker_up(self, ids_in_view):
        x_list, y_list = self.slam.markers.tolist()
        position = self.slam.robot.state[0:2]
        top_y = -1e9
        top_y_marker = None

        for idx, (marker_x, marker_y) in enumerate(zip(x_list, y_list)):
            marker = self.slam.taglist[idx]
            if marker in self.markers_travelled_to or marker not in ids_in_view:
                continue
            if (self.current_marker, marker) in ip:
                continue
            if marker_y < position[1]:
                continue
            dist = (marker_x - position[0]) ** 2 + (marker_y - position[1]) ** 2
            dist = dist ** 0.5
            dist = float(dist)
            if marker_y > top_y:
                top_y = marker_y
                top_y_marker = marker

        return top_y_marker, top_y

    def spin_radians(self, radians):
        real_time_factor = 0.5

        self.time_prev = time.time()
        model_theta = self.slam.get_state_vector()[2]
        spin_direction = 1 if radians > 0 else -1

        start_theta = model_theta
        pause_theta = model_theta
        while True:
            time_now = time.time()
            dt = time_now - self.time_prev
            dt *= real_time_factor
            self.time_prev = time_now

            model_theta = self.slam.get_state_vector()[2]
            m_l, m_r, b_l, b_r = self.rotate(model_theta, pause_theta, spin_direction)
            self.step(m_l, m_r, dt, bot_input=(b_l, b_r))

            if abs(model_theta - pause_theta) > math.pi:
                self.pause(2)
                pause_theta = model_theta

            if abs(model_theta - start_theta) > abs(radians):
                self.pause(2)
                break

    def control(self, lv, rv, dt, bot_input=None):
        # Import teleoperation control signals
        drive_meas = Measurements.DriveMeasurement(lv, rv, dt=dt)
        self.slam.predict(drive_meas)
        if bot_input is not None:
            lv = bot_input[0]
            rv = bot_input[1]
        self.ppi.set_velocity(lv, rv)

    def vision(self):
        # Import camera input and ARUCO marker info
        self.img = self.ppi.get_image()
        lms, aruco_image = self.aruco_det.detect_marker_positions(self.img)
        objects = self.yolo.get_relative_locations(self.img)
        self.slam.add_landmarks(lms, objects)

        # print(f'{self.slam.taglist=}, {self.slam.markers=}')
        self.slam.update(lms)

    def display(self, fig, ax):
        # Visualize SLAM
        ax[0].cla()
        self.slam.draw_slam_state(ax[0])

        ax[1].cla()
        ax[1].imshow(self.img[:, :, -1::-1])

        plt.pause(0.01)

    def step(self, lv, rv, dt, bot_input=None):
        self.control(lv, rv, dt, bot_input)
        self.vision()

        # Save SLAM map
        self.write_map(self.slam)

        # Output visualisation
        self.display(self.fig, self.ax)

        if time.time() - self.run_start > 580:
            print("Hit 10 minutes... exiting")
            self.ppi.set_velocity(0, 0)
            self.write_map(self.slam)
            sys.exit()

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
                if marker > 0:
                    marker = f"Marker{marker}"
                elif -10 < marker <= -1:
                    marker = f"sheep"
                elif marker <= -10:
                    marker = f"Coke"
                lines.append(f"{marker}, {marker_x}, {marker_y}\n")
            lines = sorted(lines)
            f.writelines(lines)

            # f.write("\ncurrent id, accessible id, distance\n")
            # for path in self.paths:
            #     line = ", ".join(path)
            #     f.write(line + "\n")
            # f.close()

    def drive_distance(self, distance=1):
        # spinning and looking for markers at each step
        wheel_vel = 40

        start = self.slam.get_state_vector()[0:2]
        drive = True
        real_time_factor = 0.5
        self.time_prev = time.time()
        while drive:
            time_now = time.time()
            dt = time_now - self.time_prev
            dt *= real_time_factor
            self.time_prev = time_now
            current = self.slam.get_state_vector()[0:2]
            dist = (current[0] - start[0]) ** 2 + (current[1] - start[1]) ** 2
            dist = dist ** 0.5
            lv, rv = wheel_vel, wheel_vel

            if dist > distance:
                drive = False

            self.step(lv, rv, dt)

        self.ppi.set_velocity(0, 0, 0.5)

    def check_early_exit(self):
        if len(self.paths) > 12 and len(self.slam.taglist) == 8:
            return True
        else:
            return False

    def process(self):
        # Show SLAM and camera feed side by side
        self.yolo.setup()
        self.fig, self.ax = plt.subplots(1, 2)
        img_artist = self.ax[1].imshow(self.img)
        self.times_no_marker = 0

        # Run our code
        while True:
            if self.current_marker not in skip_survey:
                self.seen_ids = self.spinOneRotation()
            if self.check_early_exit():
                print("Found all required markers. Exiting...")
                break
            goal_marker, goal_dist = self.get_next_marker_up(self.seen_ids)
            if self.current_marker not in skip_survey:
                self.markers_seen_at_step.append(self.seen_ids)
            if goal_marker is None:
                print("No marker to drive to")
                rads = math.pi / 2
                if self.times_no_marker % 2 == 1:
                    rads = 3 * math.pi / 2
                self.times_no_marker += 1
                self.spin_radians(rads)
                self.drive_distance(3)
                continue
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
