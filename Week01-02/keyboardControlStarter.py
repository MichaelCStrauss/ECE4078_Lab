# teleoperate the robot through keyboard control
# getting-started code

from pynput.keyboard import Key, Listener, KeyCode
import cv2
import numpy as np

class Keyboard:
    # feel free to change the speed, or add keys to do so
    wheel_vel_forward = 50
    wheel_vel_rotation = 10
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
            right_speed += self.wheel_vel_rotation

        if self.directions[3] == True:
            left_speed += self.wheel_vel_rotation

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
