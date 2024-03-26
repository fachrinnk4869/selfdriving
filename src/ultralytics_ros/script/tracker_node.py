#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ultralytics_ros
# Copyright (C) 2023-2024  Alpaca-zip
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import cv2
import cv_bridge
import numpy as np
import roslib.packages
import rospy
from sensor_msgs.msg import Image
from ultralytics import YOLO
from ultralytics import RTDETR
import signal   
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from ultralytics_ros.msg import YoloResult
import time
from jtop import jtop
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json
import subprocess
import os
import tensorrt as trt  
import sys
import csv
from pathlib import Path

path = roslib.packages.get_pkg_dir("ultralytics_ros")

class TrackerNode:
    def __init__(self):
        cv2.cuda.setDevice(0)
        self.input_topic = rospy.get_param("~input_topic", "image_raw")
        self.result_topic = rospy.get_param("~result_topic", "yolo_result")
        self.result_image_topic = rospy.get_param(
            "~result_image_topic", "yolo_image")
        self.yolo_model = rospy.get_param("~yolo_model", "yolov8n.pt")
        self.conf_thres = rospy.get_param("~conf_thres", 0.25)
        self.iou_thres = rospy.get_param("~iou_thres", 0.45)
        self.max_det = rospy.get_param("~max_det", 300)
        self.classes = rospy.get_param("~classes", None)
        self.tracker = rospy.get_param("~tracker", "bytetrack.yaml")
        self.device = rospy.get_param("~device", None)
        self.result_conf = rospy.get_param("~result_conf", True)
        self.result_line_width = rospy.get_param("~result_line_width", None)
        self.result_font_size = rospy.get_param("~result_font_size", None)
        self.result_font = rospy.get_param("~result_font", "Arial.ttf")
        self.result_labels = rospy.get_param("~result_labels", True)
        self.result_boxes = rospy.get_param("~result_boxes", True)
        self.is_tensorrt = rospy.get_param("~is_tensorrt", False)
        
        if(self.yolo_model.startswith("rtdetr")):
            self.model = RTDETR(f"{path}/models/{self.yolo_model}")
        else:
            self.model = YOLO(f"{path}/models/{self.yolo_model}")
        self.model.fuse()


        # Load the exported TensorRT model
        if self.is_tensorrt is True:
            if(self.yolo_model.startswith("rtdetr")):
                self.model.export(format='onnx',opset=17,simplify=True,half=True)
                preprocess()
                self.model = RTDETR(f"{path}/models/rtdetr-l.engine")
            else:
                self.model.export(format='engine')  # creates 'yolov8n.engine'
                self.model = YOLO(f"{path}/models/{Path(self.yolo_model).with_suffix('.engine')}")
        
        self.sub = rospy.Subscriber(
            self.input_topic,
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2**24,
        )
        self.results_pub = rospy.Publisher(
            self.result_topic, YoloResult, queue_size=1)
        self.result_image_pub = rospy.Publisher(
            self.result_image_topic, Image, queue_size=1
        )
        self.bridge = cv_bridge.CvBridge()
        self.use_segmentation = self.yolo_model.endswith("-seg.pt")
        # Initialize font for displaying FPS
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1
        self.font_color = (0, 255, 0)  # Green color
        self.font_thickness = 2
        self.fps_position = (10, 30) 
        # Initialize FPS variables
        self.prev_time = time.time()
        self.fps = 0
        self.csv_filename = f"{path}/fps_{self.yolo_model}.csv"
        self.fps_timer = time.time()  # Timer to track FPS
        self.fps_interval = 1  # Interval to collect FPS (in seconds)
        self.no=1

        # Initialize CSV file and write headers
        with open(self.csv_filename, mode='w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['no', 'fps'])

    def save_fps_to_csv(self):
        current_time = time.time()
        elapsed_time = current_time - self.fps_timer
        if elapsed_time >= self.fps_interval:
            fps_data = [self.no, self.fps]  # Assuming 'no' and 'fps' are already defined
            self.no = self.no +1
            with open(self.csv_filename, mode='a') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(fps_data)
            self.fps_timer = current_time
    def image_callback(self, msg):
       # Calculate FPS
        current_time = time.time()
        elapsed_time = current_time - self.prev_time
        self.prev_time = current_time
        self.fps = 1.0 / elapsed_time 
        # Calculate FPS
        self.save_fps_to_csv()
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        # Display FPS on the image
        cv2.putText(
            cv_image,
            f"FPS: {self.fps:.2f}",
            self.fps_position,
            self.font,
            self.font_scale,
            self.font_color,
            self.font_thickness,
            cv2.LINE_AA,
        )
        
        results = self.model(
            source=cv_image,
            conf=self.conf_thres,
            iou=self.iou_thres,
            max_det=self.max_det,
            classes=self.classes,
            tracker=self.tracker,
            device=self.device,
            verbose=False,
            retina_masks=True,
        )

        if results is not None:
            yolo_result_msg = YoloResult()
            yolo_result_image_msg = Image()
            yolo_result_msg.header = msg.header
            yolo_result_image_msg.header = msg.header
            yolo_result_msg.detections = self.create_detections_array(results)
            yolo_result_image_msg = self.create_result_image(results)
            if self.use_segmentation:
                yolo_result_msg.masks = self.create_segmentation_masks(results)
            self.results_pub.publish(yolo_result_msg)
            self.result_image_pub.publish(yolo_result_image_msg)

    def create_detections_array(self, results):
        detections_msg = Detection2DArray()
        bounding_box = results[0].boxes.xywh
        classes = results[0].boxes.cls
        confidence_score = results[0].boxes.conf
        for bbox, cls, conf in zip(bounding_box, classes, confidence_score):
            detection = Detection2D()
            detection.bbox.center.x = float(bbox[0])
            detection.bbox.center.y = float(bbox[1])
            detection.bbox.size_x = float(bbox[2])
            detection.bbox.size_y = float(bbox[3])
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = int(cls)
            hypothesis.score = float(conf)
            detection.results.append(hypothesis)
            detections_msg.detections.append(detection)
        return detections_msg

    def create_result_image(self, results):
        plotted_image = results[0].plot(
            conf=self.result_conf,
            line_width=self.result_line_width,
            font_size=self.result_font_size,
            font=self.result_font,
            labels=self.result_labels,
            boxes=self.result_boxes,
        )
        # print(results)
        result_image_msg = self.bridge.cv2_to_imgmsg(
            plotted_image, encoding="bgr8")
        return result_image_msg

    def create_segmentation_masks(self, results):
        masks_msg = []
        for result in results:
            if hasattr(result, "masks") and result.masks is not None:
                for mask_tensor in result.masks:
                    mask_numpy = (
                        np.squeeze(mask_tensor.data.to("cpu").detach().numpy()).astype(
                            np.uint8
                        )
                        * 255
                    )
                    mask_image_msg = self.bridge.cv2_to_imgmsg(
                        mask_numpy, encoding="mono8"
                    )
                    masks_msg.append(mask_image_msg)
        return masks_msg

    def restart_jtop_service():
        subprocess.run(['sudo', 'systemctl', 'restart', 'jtop.service'])
    def update_plot(frame):
        global gpu_data, mem_data, ax1, ax2
        with jtop() as jetson:
            gpu_usage = jetson.stats['GPU']
            gpu_data.append(gpu_usage)
            mem_usage = jetson.stats['RAM']
            mem_data.append(mem_usage)
            ax1.clear()
            ax1.plot(gpu_data)
            ax1.set_title('GPU Usage (%)')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Usage (%)')
            ax2.clear()
            ax2.plot(mem_data)
            ax2.set_title('Memory Usage (%)')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Usage (%)')

def preprocess():
    f_onnx = f"{path}/models/rtdetr-l.onnx"
    file = Path(f_onnx)
    f = file.with_suffix('.engine')   

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)  
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    success = parser.parse_from_file(f_onnx)
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))

    if not success:
        pass # Error handling code here

    config = builder.create_builder_config()

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1 MiB
    config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS))

    config.set_flag(trt.BuilderFlag.FP16) # set f16

    serialized_engine = builder.build_serialized_network(network, config)


    metadata = {'description': 'Ultralytics rtdetr-l...sunny.yaml', 'author': 'Ultralytics', 'license': 'AGPL-3.0 https://ult...om/license', 'date': '2024-03-09T19:14:58.861682', 'version': '8.0.230', 'stride': 32, 'task': 'detect', 'batch': 1, 'imgsz': [640, 640], 'names': {0: 'bus', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'person', 5: 'rider', 6: 'truck'}}
    t = open(f, 'wb')
    meta = json.dumps(metadata)
    t.write(len(meta).to_bytes(4, byteorder='little', signed=True))
    t.write(meta.encode())
    # Model
    t.write(serialized_engine)

    t.close()

if __name__ == "__main__":
    start_latency = time.time()
    rospy.init_node("tracker_node")
    node = TrackerNode()
    yolo_model = node.yolo_model
    process=subprocess.run(['python3', f"{path}/cek_gpu.py", f"{path}/gpu_mem_{yolo_model}.csv", f"{path}/plot_{yolo_model}.png"])
    latency = time.time() - start_latency
    with open(f"{path}/latency_{yolo_model}_tensorrt.csv", 'w') as f:
        f.write(f"{latency}")
        f.close()
    rospy.spin()
