import time
from pathlib import Path
import cv2
import torch
import rospy
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import roslib.packages

# Import necessary functions from utils.utils
from yolop.utils.utils_camera import (
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
    LoadImages,
    ROITrackbar
)

path = roslib.packages.get_pkg_dir("yolop")

# Initialize cv_bridge


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # print(sem_img.shape)
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / \
            shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    return img, ratio, (dw, dh)


class Yolop():
    def __init__(self):
        self.t1 = time_synchronized()
        # Fetch ROS parameters
        self.weights = rospy.get_param(
            '~weights', f'src/yolop/data/weights/yolopv2.pt')
        self.source = rospy.get_param('~source', f'src/yolop/input/drive.mp4')
        self.imgsz = rospy.get_param('~img_size', 640)
        self.conf_thres = rospy.get_param('~conf_thres', 0.3)
        self.iou_thres = rospy.get_param('~iou_thres', 0.45)
        self.device = rospy.get_param('~device', '0')
        self.save_conf = rospy.get_param('~save_conf', False)
        self.save_txt = rospy.get_param('~save_txt', False)
        self.nosave = rospy.get_param('~nosave', False)
        self.classes = None
        self.agnostic_nms = rospy.get_param('~agnostic_nms', False)
        self.project = rospy.get_param('~project', f'src/yolop/runs/detect')
        self.name = rospy.get_param('~name', 'exp')
        self.exist_ok = rospy.get_param('~exist_ok', False)
        self.save_img = not self.nosave and not self.source.endswith('.txt')
        self.fps_pub = rospy.Publisher('fps', Float32, queue_size=10)
        self.lane_dev_pub = rospy.Publisher('lane_deviation', Float32, queue_size=10)
        self.left_curve_pub = rospy.Publisher(
            'left_lane_curve', Float32, queue_size=10)
        self.right_curve_pub = rospy.Publisher(
            'right_lane_curve', Float32, queue_size=10)
        self.center_curve_pub = rospy.Publisher(
            'center_lane_curve', Float32, queue_size=10)
        # Load model
        self.stride = 32
        self.model = torch.jit.load(self.weights)
        self.device = select_device(self.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.model = self.model.to(self.device)

        if self.half:
            self.model.half()  # to FP16
        self.model.eval()

    def detect(self, cv_image=None):

        # Set Dataloader
        # Padded resize
        img0 = cv2.resize(cv_image, (1280, 720),
                          interpolation=cv2.INTER_LINEAR)
        img = letterbox(img0, 640, stride=32)[0]

        # Convert
        # BGR to RGB, to 3x416x416
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        dataset = [(None, img, img0, None)]

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                next(self.model.parameters())))  # run once
        frame_count = 0
        for path, img, im0s, vid_cap in dataset:
            frame_count += 1
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0

            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            [pred, anchor_grid], _, ll = self.model(img)

            # waste time: the incompatibility of torch.jit.trace causes extra time consumption in demo version
            # but this problem will not appear in official version
            pred = split_for_trace_model(pred, anchor_grid)

            # Apply NMS
            pred = non_max_suppression(
                pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)

            # da_seg_mask = driving_area_mask(seg)
            ll_seg_mask = lane_line_mask(ll)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                _, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                s += '%gx%g ' % img.shape[2:]  # print string
                # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class

                    # Write results
                    for *xyxy, conf, cls in reversed(det):

                        if self.save_img:  # Add bbox to image
                            plot_one_box(xyxy, im0, line_thickness=3)

                        print(cls)

                # Print time (inference)
                output_image, left_curve, right_curve, lane_deviation = show_seg_result(
                    im0, ll_seg_mask, is_demo=True)
                try:
                    self.left_curve_pub.publish(left_curve)
                except NameError:
                    rospy.logwarn("Left curve not calculated.")

                try:
                    self.right_curve_pub.publish(right_curve)
                except NameError:
                    rospy.logwarn("Right curve not calculated.")
                try:
                    self.lane_dev_pub.publish(center_curve)
                except NameError:
                    rospy.logwarn("Lane Deviation not calculated.")
                fps = 1 / (time_synchronized() - self.t1)
                self.t1 = time_synchronized()  # Forward pass FPS.
                self.fps_pub.publish(fps)
                print("FPS: ", fps)
                cv2.putText(
                    output_image,
                    text=f"YOLOPv2 FPS: {fps:.1f}",
                    org=(15, 35),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 0, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA
                )
                cv2.imshow('Image', output_image)
                cv2.waitKey(1)

                if cv_image is not None or cv2.waitKey(1) & 0xFF == ord('q'):
                    break


class LaneDetectionNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.lane_detector = Yolop()

        # Subscribe to the camera topic
        rospy.Subscriber("/cv_camera/image_raw", Image, self.image_callback)

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with torch.no_grad():
                self.lane_detector.detect(cv_image)
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")


if __name__ == '__main__':
    rospy.init_node('lane_detection_camera')
    path = roslib.packages.get_pkg_dir("yolop")

    # Initialize the LaneDetectionNode
    lane_detection_node = LaneDetectionNode()

    rospy.spin()
    cv2.destroyAllWindows()
