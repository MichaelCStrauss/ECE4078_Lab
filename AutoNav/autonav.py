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

        self.keyboard_controlled = False
        self.manual = False

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
        wheel_vel = 30
        d_theta = abs(model_theta - pause_theta)

        lv, rv = -wheel_vel, wheel_vel
        k = 60
        break_at = 24 * math.pi / 12
        reduction = k if d_theta > break_at else 0
        b_l = -(wheel_vel - reduction)
        b_r = -b_l
        b_l, b_r = int(b_l), int(b_r)

        k2 = 10
        k3 = 0
        y_int = 7
        y_int2 = wheel_vel
        model_vel = (
            y_int + k2 * d_theta if d_theta < math.pi / 2 else y_int2 - k3 * d_theta
        )
        m_l = -1 * min(wheel_vel / 2, model_vel)
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

            # print(f"{seen_ids=}")

            corners, ids, rejected, rvecs, tvecs = self.get_camera()
            if ids is not None:
                ids_in_view = [ids[i, 0] for i in range(len(ids))]
                for id in ids_in_view:
                    seen_ids.add(id)

            self.step(m_l, m_r, dt, bot_input=(b_l, b_r))
            image = self.ppi.get_image()
            objects = self.yolo.get_relative_locations(image)
            for class_id, local_x, local_y in objects:
                world_x, world_y = self.slam.transform_local_world_space(local_x, local_y)
                tag_id = self.slam.get_tag_of_object(class_id, world_x, world_y)
                if tag_id is None:
                    continue
                seen_ids.add(tag_id)

            if model_theta - pause_theta > 2*math.pi:
                self.pause(3)
                pause_theta = model_theta
            if model_theta - initial_theta > 2 * math.pi:
                spin = False
            kboard_info = self.keyboard.get_drive_signal()
            if kboard_info[3] == True:
                spin = False

        self.pause()
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


            if abs(model_theta - pause_theta) > 2*math.pi:
                self.pause(2)
                pause_theta = model_theta

            if goal_marker_id > 0:
                # Get the ids in view
                corners, ids, rejected, rvecs, tvecs = self.get_camera()
                print(ids)
                if ids is None:
                    continue
                ids_in_view = [ids[i, 0] for i in range(len(ids))]

                print(ids_in_view)
                if goal_marker_id in ids_in_view:
                    break
            else:
                image = self.ppi.get_image()
                objects = self.yolo.get_relative_locations(image)
                print(objects)
                found = False
                for class_id, local_x, local_y in objects:
                    world_x, world_y = self.slam.transform_local_world_space(local_x, local_y)
                    tag_id = self.slam.get_tag_of_object(class_id, world_x, world_y)
                    if tag_id == goal_marker_id:
                        found = True
                        break
                if found:
                    break

        self.pause(2)

        adjusting = True
        adjusting_ticks = 0
        while adjusting and adjusting_ticks < 30:
            time_now = time.time()
            dt = time_now - self.time_prev
            dt *= real_time_factor
            self.time_prev = time_now
            wheel_vel = 30
            print(f"adjusting to target {goal_marker_id}")

            lv, rv = 0, 0

            if goal_marker_id > 0:
                corners, ids, rejected, rvecs, tvecs = self.get_camera()

                if ids is not None:
                    for i in range(len(ids)):
                        idi = ids[i, 0]
                        # Some markers appear multiple times but should only be handled once.
                        if idi == goal_marker_id:
                            avg_x = corners[i][0, :, 0].mean()
                            diff_from_center = avg_x - 320
                            print(f"{diff_from_center}")
                            k = 0.4
                            lv, rv = (
                                diff_from_center * k,
                                -diff_from_center * k,
                            )
                            lv, rv = int(lv), int(rv)
                            lv = np.clip(lv, -wheel_vel, wheel_vel)
                            rv = np.clip(rv, -wheel_vel, wheel_vel)
                            if abs(diff_from_center) < 10:
                                adjusting = False
            else:
                image = self.ppi.get_image()
                preds = self.yolo.forward(image)
                target_class = 0 if -10 < goal_marker_id <= -1 else 1
                if preds is not None:
                    for prediction in preds:
                        if prediction[5] != target_class:
                            continue
                        diff_from_center = float(prediction[2] + prediction[0]) / 2 - 320
                        print(f"{diff_from_center}")
                        k = 0.4
                        lv, rv = (
                            diff_from_center * k,
                            -diff_from_center * k,
                        )
                        lv, rv = int(lv), int(rv)
                        lv = np.clip(lv, -wheel_vel, wheel_vel)
                        rv = np.clip(rv, -wheel_vel, wheel_vel)
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
            print(f"{dist} {ids}")

            threshold = 1.5
            if dist < threshold or dist > prev_dist:
                driving = False
            elif dist > 1:
                b_lv = 75
            prev_dist = dist

            kboard_data = self.keyboard.get_drive_signal()
            stop_signal = kboard_data[3]
            if stop_signal:
                break


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

        print(f"{min_marker}, {min_dist}")
        print(f"{max_marker}, {max_dist}")
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
        self.slam.update(lms, objects)

    def display(self, fig, ax):
        # Visualize SLAM
        ax[0].cla()
        self.slam.draw_slam_state(ax[0])

        ax[1].cla()
        ax[1].imshow(self.img[:, :, -1::-1])

        plt.pause(0.01)
    
    def adjust(self):
        directions, move = self.keyboard.get_key_status()
        if not move:
            return
        current_pos = self.slam.robot.state
        dt = time.time() - self.time_prev
        speed = 0.5 * dt
        current_pos[0] += np.clip((directions[3] - directions[2]) * speed, -0.3, 0.3)
        current_pos[1] += np.clip((directions[0] - directions[1]) * speed, -0.3, 0.3)
        self.slam.robot.state = current_pos

    def step(self, lv, rv, dt, bot_input=None):
        print(self.slam.robot.state)
        print(self.slam.taglist)
        if not self.manual:
            keyboard_l, keyboard_r, adjustment, _ = self.keyboard.get_drive_signal()
            if adjustment:
                lv += int(keyboard_l / 1.5)
                rv += int(keyboard_r / 1.5)
            else:
                if keyboard_l != 0 or keyboard_r != 0:
                    lv, rv, b_lv, b_rv = self.convert_keyboard_to_slam_bot(keyboard_l, keyboard_r)
                    bot_input = b_lv, b_rv
        self.adjust()
        self.control(lv, rv, dt, bot_input)
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
            num_sheep = 0
            num_coke = 0
            for idx, (marker_x, marker_y) in enumerate(zip(x_list, y_list)):
                marker = self.slam.taglist[idx]
                if marker > 0:
                    marker = f"Marker{marker}"
                elif -10 < marker <= -1:
                    num_sheep += 1
                    marker = f"sheep{num_sheep}"
                elif marker <= -10:
                    num_coke += 1
                    marker = f"Coke{num_coke}"
                lines.append(f"{marker}, {round(marker_x, 4)}, {round(marker_y, 4)}\n")
            lines = sorted(lines)
            f.writelines(lines)
        return lines

            # f.write("\ncurrent id, accessible id, distance\n")
            # for path in self.paths:
            #     line = ", ".join(path)
            #     f.write(line + "\n")
            # f.close()

    def drive_distance(self, distance=1):
        # spinning and looking for markers at each step
        wheel_vel = 65

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
            b_lv, b_rv = lv + 5, rv

            if dist > distance:
                drive = False

            self.step(lv, rv, dt, bot_input=(b_lv, b_rv))

    def check_early_exit(self):
        if len(self.paths) > 12 and len(self.slam.taglist) == 8:
            return True
        else:
            return False
        
    def run_one_iteration(self):
        self.seen_ids = self.spinOneRotation()

        manual = False
        target = None
        while True:
            print("Current Map:")
            print("".join(self.write_map(self.slam)))
            seen_string = "\n"
            for m_id in sorted(self.seen_ids):
                seen_string += f'{m_id} '
                if m_id <= -10:
                    seen_string += "coke "
                elif m_id < 0:
                    seen_string += "sheep "
                try:
                    pos = self.get_marker_location(m_id)
                    seen_string += f"({round(pos[0], 1)}, {round(pos[1], 1)})"
                except ValueError:
                    seen_string += f"(unknown)"
                seen_string += " || "
            print("Seen IDs: " + seen_string)
            print("Enter ID to drive to, or 'drive'")
            command = input()
            if command == "drive":
                manual = True
                break
            try:
                if int(command) in self.seen_ids:
                    target = int(command)
                    break
            except:
                continue
        
        if manual:
            self.manual_control()
        else:
            self.spin_to_marker(target)
            self.drive_to_marker(target)
    
    def convert_keyboard_to_slam_bot(self, lv, rv):
        b_lv, b_rv = lv, rv
        if lv == 0 or rv == 0:
            pass
        elif lv / rv > 0:
            lv = int(lv / 2.1)
            rv = int(rv / 2.1)
            b_lv += 15
        elif lv / rv < 0:
            lv = int(lv / 5)
            rv = int(rv / 5)

        return lv, rv, b_lv, b_rv

    def manual_control(self):
        self.manual = True
        self.time_prev = time.time()
        while True:
            time_now = time.time()
            dt = time_now - self.time_prev

            lv, rv, adjust, stop = self.keyboard.get_drive_signal()
            if stop:
                break 

            lv, rv, b_lv, b_rv = self.convert_keyboard_to_slam_bot(lv, rv)
            if adjust:
                b_lv, b_rv = 0, 0

            self.step(lv, rv, dt, bot_input=(b_lv, b_rv))
            self.time_prev = time_now
        self.manual = False

    def process(self):
        # Show SLAM and camera feed side by side
        self.yolo.setup()
        self.fig, self.ax = plt.subplots(1, 2)
        img_artist = self.ax[1].imshow(self.img)
        self.times_no_marker = 0

        # Run our code
        while True:
            # self.manual_control()
            self.run_one_iteration()



if __name__ == "__main__":
    # Location of the calibration files
    currentDir = os.getcwd()
    datadir = "{}/calibration/".format(currentDir)
    # connect to the robot
    ppi = penguinPiC.PenguinPi()

    kb = False
    if len(sys.argv) > 1 and sys.argv[1] == 'keyboard':
        print("Using keyboard!")
        kb = True

    # Perform Manual SLAM
    operate = Operate(datadir, ppi)
    operate.keyboard_controlled = kb
    try:
        operate.process()
    except KeyboardInterrupt:
        operate.ppi.set_velocity(0, 0)
