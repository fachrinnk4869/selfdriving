#!/usr/bin/env python3
import time
from pathlib import Path
import cv2
import torch
import rospy
from std_msgs.msg import Float32
import numpy as np
import roslib.packages

# Import necessary functions from utils.utils
from yolop.utils.utils_raw import (
    time_synchronized,
    select_device,
    increment_path,
    scale_coords,
    xyxy2xywh,
    non_max_suppression,
    split_for_trace_model,
    driving_area_mask,
    lane_line_mask,
    plot_one_box,
    show_seg_result,
    AverageMeter,
    LoadImages
)

path = roslib.packages.get_pkg_dir("yolop")

t1 = time_synchronized()


def detect():
    global t1
    # Fetch ROS parameters
    weights = rospy.get_param('~weights', f'src/yolop/data/weights/yolopv2.pt')
    source = rospy.get_param('~source', f'src/yolop/input/drive.mp4')
    imgsz = rospy.get_param('~img_size', 640)
    conf_thres = rospy.get_param('~conf_thres', 0.3)
    iou_thres = rospy.get_param('~iou_thres', 0.45)
    device = rospy.get_param('~device', '0')
    save_conf = rospy.get_param('~save_conf', False)
    save_txt = rospy.get_param('~save_txt', False)
    nosave = rospy.get_param('~nosave', False)
    classes = None
    agnostic_nms = rospy.get_param('~agnostic_nms', False)
    project = rospy.get_param('~project', f'src/yolop/runs/detect')
    name = rospy.get_param('~name', 'exp')
    exist_ok = rospy.get_param('~exist_ok', False)
    save_img = not nosave and not source.endswith('.txt')

    fps_pub = rospy.Publisher('fps', Float32, queue_size=10)

    # Load model
    stride = 32
    model = torch.jit.load(weights)
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    model = model.to(device)

    if half:
        model.half()  # to FP16
    model.eval()

    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
            next(model.parameters())))  # run once
    frame_count = 0
    total_fps = 0
    for path, img, im0s, vid_cap in dataset:
        frame_count += 1
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        [pred, anchor_grid], seg, ll = model(img)

        # waste time: the incompatibility of torch.jit.trace causes extra time consumption in demo version
        # but this problem will not appear in official version
        pred = split_for_trace_model(pred, anchor_grid)

        # Apply NMS
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

        fps = 1 / (time_synchronized() - t1)  # Forward pass FPS.
        t1 = time_synchronized()
        total_fps += fps

        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            s += '%gx%g ' % img.shape[2:]  # print string
            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):

                    if save_img:  # Add bbox to image
                        plot_one_box(xyxy, im0, line_thickness=3)

                    print(cls)

            # Print time (inference)
            show_seg_result(im0, (da_seg_mask, ll_seg_mask), is_demo=True)
            fps_pub.publish(fps)
            cv2.putText(
                im0,
                text=f"YOLOPv2 FPS: {fps:.1f}",
                org=(15, 35),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 0, 255),
                thickness=2,
                lineType=cv2.LINE_AA
            )
            cv2.imshow('Image', im0)
            cv2.waitKey(1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    rospy.init_node('lane_detection_raw_node')
    path = roslib.packages.get_pkg_dir("yolop")
    # opt = rospy.get_param('~lane_detection_node_params')
    # print(opt)
    with torch.no_grad():
        detect()
    rospy.spin()
    # cv2.destroyAllWindows()
