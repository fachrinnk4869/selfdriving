#!/usr/bin/env bash
source devel/setup.bash
roslaunch ultralytics_ros tracker.launch debug:=true yolo_model:=yolov8n.pt is_tensorrt:=True
roslaunch ultralytics_ros tracker.launch debug:=true yolo_model:=yolov8s.pt is_tensorrt:=True
roslaunch ultralytics_ros tracker.launch debug:=true yolo_model:=yolov8m.pt is_tensorrt:=True 
roslaunch ultralytics_ros tracker.launch debug:=true yolo_model:=yolov9c.pt is_tensorrt:=True
#roslaunch ultralytics_ros tracker.launch debug:=true yolo_model:=rtdetr-l.pt