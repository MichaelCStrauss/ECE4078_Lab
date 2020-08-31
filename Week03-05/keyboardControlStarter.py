# teleoperate the robot through keyboard control
# getting-started code

from pynput.keyboard import Key, Listener, KeyCode
import cv2
import numpy as np
# import OpenCV ARUCO functions
import cv2.aruco as aruco

class Keyboard:
    # feel free to change the speed, or add keys to do so
    wheel_vel_forward = 30
    wheel_vel_rotation = 20
    def __init__(self, ppi=None):
        # storage for key presses
        self.directions = [False for _ in range(4)]
        self.signal_stop = False 

        # connection to PenguinPi robot
        self.ppi = ppi
        self.wheel_vels = [0, 0]

        self.listener = Listener(on_press=self.on_press, on_release=self.on_release).start()

    def on_press(self, key):
        print(key)
        # use arrow keys to drive, space key to stop
        # feel free to add more keys
        if key == Key.up or key == 'w':
            self.directions[0] = True
        elif key == Key.down or key == 's':
            self.directions[1] = True
        elif key == Key.left or key == 'a':
            self.directions[2] = True
        elif key == Key.right or key == 'd':
            self.directions[3] = True
        elif key == Key.space:
            self.signal_stop = True
        
        self.send_drive_signal()

    def on_release(self, key):
        print(f'{key} released')
        # use arrow keys to drive, space key to stop
        # feel free to add more keys
        if key == Key.up or key == 'w':
            self.directions[0] = False
        elif key == Key.down or key == 's':
            self.directions[1] = False
        elif key == Key.left or key == 'a':
            self.directions[2] = False
        elif key == Key.right or key == 'd':
            self.directions[3] = False
        elif key == Key.space:
            self.signal_stop = False
        elif key == Key.shift:
            self.wheel_vel_forward += 20
            self.wheel_vel_rotation += 5
        elif key == Key.ctrl:
            self.wheel_vel_forward -= 20
            self.wheel_vel_rotation -= 5
        self.send_drive_signal()


    def get_drive_signal(self):           
        # translate the key presses into drive signals 
        
        # compute drive_forward and drive_rotate using wheel_vel_forward and wheel_vel_rotation
        # drive_forward = ???
        # drive_rotate = ???
        drive_forward = 0
        if self.directions[0] == True:
            drive_forward = self.wheel_vel_forward
        elif self.directions[1] == True:
            drive_forward = -self.wheel_vel_forward

        # translate drive_forward and drive_rotate into left_speed and right_speed
        left_speed = drive_forward
        right_speed = drive_forward

        #rotate left
        if self.directions[2] == True: 
            right_speed = self.wheel_vel_rotation
            left_speed = -self.wheel_vel_rotation

        if self.directions[3] == True:
            right_speed = -self.wheel_vel_rotation
            left_speed = self.wheel_vel_rotation

        if self.signal_stop:
            left_speed = 0
            right_speed = 0

        return left_speed, right_speed
    
    def send_drive_signal(self):
        if not self.ppi is None:
            lv, rv = self.get_drive_signal()
            lv, rv = self.ppi.set_velocity(lv, rv)
            self.wheel_vels = [lv, rv]
            
    def latest_drive_signal(self):
        return self.wheel_vels
    

if __name__ == "__main__":
    import penguinPiC
    ppi = penguinPiC.PenguinPi()

    keyboard_control = Keyboard(ppi)

    cv2.namedWindow('video', cv2.WINDOW_NORMAL);
    cv2.setWindowProperty('video', cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE);

    while True:
        # font display options
        font = cv2.FONT_HERSHEY_SIMPLEX
        location = (0, 0)
        font_scale = 1
        font_col = (255, 255, 255)
        line_type = 2

        # get velocity of each wheel
        wheel_vels = keyboard_control.latest_drive_signal()
        L_Wvel = wheel_vels[0]
        R_Wvel = wheel_vels[1]

        # get current camera frame
        curr = ppi.get_image()

        # uncomment to see how noises influence the accuracy of ARUCO marker detection
        #im = np.zeros(np.shape(curr), np.uint8)
        #cv2.randn(im,(0),(99))
        #curr = curr + im
        
        # show ARUCO marker detection annotations
        aruco_params = aruco.DetectorParameters_create()
        aruco_params.minDistanceToBorder = 0
        aruco_params.adaptiveThreshWinSizeMax = 1000
        aruco_dict = aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
    
        corners, ids, rejected = aruco.detectMarkers(curr, aruco_dict, parameters=aruco_params)
    
        grayscale = cv2.cvtColor(curr,cv2.COLOR_BGR2GRAY)
        for corner in corners:
            x1, y1 = corner[0][0]
            x2, y2 = corner[0][2]
            cropped = grayscale[int(y1):int(y2), int(x1):int(x2)]
            additional_corners = cv2.goodFeaturesToTrack(cropped, 27, 0.01, 10)
            for add_corner in additional_corners:
                add_corner[0][0] += x1
                add_corner[0][1] += y1
            homography, _ = cv2.findHomography(corner, np.array([[0, 0], [100, 0], [100, 100], [0, 100]]))
            normalised_additional_corners = cv2.perspectiveTransform(additional_corners, homography)
            warped = cv2.warpPerspective(grayscale, homography, (100, 100))
            print("Cropped")
            cv2.imshow('cropped', cropped)
            print("Normalised Corners:")
            print(normalised_additional_corners)
            for point in normalised_additional_corners:
                cv2.circle(warped, tuple(point[0]), 5, (100, 0, 240))
            cv2.imshow('warped', warped)

            # Create '3D points' for the Aruco markers by transforming 255 points in matrix to coordinates in 3d (note flipped y axis)
            # FOr each 3d point, find the closest detected feature as a 2d screen point
            # Pass those to code on slides

        aruco.drawDetectedMarkers(curr, corners, ids) # for detected markers show their ids
        # aruco.drawDetectedMarkers(curr, rejected, borderColor=(100, 0, 240))  # unknown squares

        # homography, _ = cv2.findHomography(corners, np.array([[0, 0], [100, 0], [100, 100], [0, 100]]))
        # normalised_additional_corners = cv2.perspectiveTransform(additional_corners, homography)

        # print("Corners:")
        # print(corners)
        # print("Homography:")
        # print(homography)
        # print("Normalised Additional Corners")
        # print(normalised_additional_corners)

        # scale to 144p
        # feel free to change the resolution
        resized = cv2.resize(curr, (960, 720), interpolation = cv2.INTER_AREA)

        # feel free to add more GUI texts
        cv2.putText(resized, 'PenguinPi', (15, 50), font, font_scale, font_col, line_type)
        cv2.putText(resized, f'Arrows or WASD to move. Shift to inc. speed, Ctrl to dec.', (15, 100), font, font_scale, font_col, line_type)
        direction = 'Stopped'
        if L_Wvel == R_Wvel:
            if L_Wvel > 0:
                direction = 'Forward'
            if L_Wvel < 0:
                direction = 'Backwards'
        elif L_Wvel > R_Wvel:
            direction = 'Right'
        elif L_Wvel < R_Wvel:
            direction = 'Left'
        cv2.putText(resized, f'L: {L_Wvel}, R: {R_Wvel}. Direction: {direction}', (15, 150), font, font_scale, font_col, line_type)

        cv2.imshow('video', resized)
        cv2.waitKey(1)

        continue
