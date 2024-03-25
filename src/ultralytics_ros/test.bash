#!/usr/bin/env bash
# roslaunch ultralytics_ros tracker.launch debug:=true yolo_model:=yolov8n.pt
# roslaunch ultralytics_ros tracker.launch debug:=true yolo_model:=yolov8s.pt
roslaunch ultralytics_ros tracker.launch debug:=true yolo_model:=yolov8m.pt
roslaunch ultralytics_ros tracker.launch debug:=true yolo_model:=yolov9c.pt
roslaunch ultralytics_ros tracker.launch debug:=true yolo_model:=rtdetr-l.pt