import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size,
    non_max_suppression,
    apply_classifier,
    scale_coords,
    xyxy2xywh,
    plot_one_box,
    strip_optimizer,
    set_logging,
)
from utils.datasets import letterbox

from utils.torch_utils import select_device, load_classifier, time_synchronized

import matplotlib.pyplot as plt


class YoloV5(object):
    def __init__(self, weights, device):
        self.weights = weights
        self.imgsz = 320
        self.model = None
        self.device = device

        self.classes = [0, 1]
        self.agnostic_nms = False
        self.conf_thres = 0.5
        self.iou_thres = 0.4

        self.coke_dimensions = [0.06, 0.06, 0.14]
        self.sheep_dimensions = [0.108, 0.223, 0.204]

    def setup(self):
        self.model = attempt_load(self.weights, map_location=self.device)
        self.model.eval()
        self.imgsz = check_img_size(
            self.imgsz, s=self.model.stride.max()
        )  # check img_size

    def forward(self, image):
        img = letterbox(image, new_shape=self.imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        # Convert
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)

        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        img = img.float() / 255.0

        pred = self.model(img)[0]
        pred = non_max_suppression(
            pred,
            self.conf_thres,
            self.iou_thres,
            classes=self.classes,
            agnostic=self.agnostic_nms,
        )[0]
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], image.shape).round()

        print(pred)
        for prediction in pred.split(1):
            prediction = prediction.squeeze()
            det_class = "sheep" if prediction[5] == 0 else "coke"
            confidence = float(prediction[4])
            print(f"Predicted a {det_class} with {confidence=}")

        return pred

    def get_relative_locations(self, image):
        camera_w, camera_h = 640, 480
        half_size_x = camera_w / 2
        half_size_y = camera_h / 2
        # h_fov = 1.0855
        h_fov = 0.8517
        focal_length = (camera_w / 2) / np.tan(h_fov / 2)

        print(f"{focal_length=}")

        pred = self.forward(image)

        # plt.imshow(image)
        # plt.show()

        # Get the locations here
        for prediction in pred.split(1):
            prediction = prediction.squeeze()

            # image_points = [
            #     prediction[0:2],
            #     prediction[[2, 1]],
            #     prediction[2:4],
            #     prediction[[0, 3]],
            # ]

            # if prediction[5] == 0:
            #     obj = self.sheep_dimensions
            #     object_points = [
            #         [0, self.sheep_dimensions[]]
            #     ]
            if prediction[5] == 0:
                true_height = self.sheep_dimensions[2]
            elif prediction[5] == 1:
                true_height = self.coke_dimensions[2]
            else:
                print("wrong class")
                continue

            pixel_height = float(prediction[3] - prediction[1])
            pixel_center = float(prediction[2] + prediction[0]) / 2 - half_size_x
            pixel_center = -pixel_center
            distance_to_camera = true_height / pixel_height * focal_length

            print(f"Object distance to camera: {distance_to_camera}")

            # # object_height_real = 2
            # # sensor_height = 0.4

            # # top_middle = box_top_right - box_top_left
            # # bottom_middle = box_bottom_left - box_bottom_left
            # # image_center = [image_top_middle, image_top_middle + (bottom_middle + top_middle)/2]

            # # object_height_image = bottom_middle - top_middle

            # # distance_to_camera = focal_length*object_height_real*camera_h/(object_height_image*sensor_height)

            theta = np.arctan(pixel_center * np.tan(h_fov / 2) / half_size_x)

            object_x = distance_to_camera * np.sin(theta)
            print(f"{pixel_center=} {theta=} {object_x=}")
