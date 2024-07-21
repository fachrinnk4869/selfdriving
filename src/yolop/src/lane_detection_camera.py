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
bridge = CvBridge()


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


def detect(cv_image=None):
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
    left_curve_pub = rospy.Publisher('left_lane_curve', Float32, queue_size=10)
    right_curve_pub = rospy.Publisher(
        'right_lane_curve', Float32, queue_size=10)
    center_curve_pub = rospy.Publisher(
        'center_lane_curve', Float32, queue_size=10)
    save_dir = Path(increment_path(Path(project) / name,
                    exist_ok=exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,
                                                          exist_ok=True)  # make dir
    inf_time = AverageMeter()
    waste_time = AverageMeter()
    nms_time = AverageMeter()

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
    vid_path, vid_writer = None, None
    if cv_image is None:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
    else:
        # Padded resize
        img0 = cv2.resize(cv_image, (1280, 720),
                          interpolation=cv2.INTER_LINEAR)
        img = letterbox(img0, 640, stride=32)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        dataset = [(None, img, img0, None)]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
            next(model.parameters())))  # run once
    t0 = time.time()
    frame_count = 0
    total_fps = 0
    if cv_image is None:
        first_item = next(iter(dataset))
        _, _, initial_im0s, _ = first_item
        ROITrackbar(initial_im0s)

    for path, img, im0s, vid_cap in dataset:
        frame_count += 1
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        [pred, anchor_grid], seg, ll = model(img)
        t2 = time_synchronized()

        # waste time: the incompatibility of torch.jit.trace causes extra time consumption in demo version
        # but this problem will not appear in official version
        tw1 = time_synchronized()
        pred = split_for_trace_model(pred, anchor_grid)
        tw2 = time_synchronized()

        # Apply NMS
        t3 = time_synchronized()
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t4 = time_synchronized()

        fps = 1 / (t2 - t1)  # Forward pass FPS.
        total_fps += fps

        # da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            _, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            save_path = str(save_dir / "cv_image")  # img.jpg
            txt_path = str(save_dir / 'labels' / "cv_image")  # img.txt
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
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)
                                          ) / gn).view(-1).tolist()  # normalized xywh
                        # label format
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img:  # Add bbox to image
                        plot_one_box(xyxy, im0, line_thickness=3)

                    print(cls)

            # Print time (inference)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            output_image, left_curve, right_curve, center_curve = show_seg_result(
                im0, ll_seg_mask, is_demo=True)
            try:
                left_curve_pub.publish(left_curve)
            except NameError:
                rospy.logwarn("Left curve not calculated.")

            try:
                right_curve_pub.publish(right_curve)
            except NameError:
                rospy.logwarn("Right curve not calculated.")
            try:
                center_curve_pub.publish(center_curve)
            except NameError:
                rospy.logwarn("Center curve not calculated.")
            fps_pub.publish(fps)
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

            # Save results (image with detections)
            # if save_img:
            #       # 'video' or 'stream'
            #     if vid_path != save_path:  # new video
            #         vid_path = save_path
            #         if isinstance(vid_writer, cv2.VideoWriter):
            #             vid_writer.release()  # release previous video writer
            #         if vid_cap:  # video
            #             fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #             #w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #             #h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #             w,h = im0.shape[1], im0.shape[0]
            #         else:  # stream
            #             fps, w, h = 30, im0.shape[1], im0.shape[0]
            #             save_path += '.mp4'
            #         vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            #     vid_writer.write(output_image)

            if cv_image is not None or cv2.waitKey(1) & 0xFF == ord('q'):
                break

    inf_time.update(t2 - t1, img.size(0))
    nms_time.update(t4 - t3, img.size(0))
    waste_time.update(tw2 - tw1, img.size(0))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' %
          (inf_time.avg, nms_time.avg))
    print(f'Done. ({time.time() - t0:.3f}s)')
    print(f"Average FPS: {(total_fps / frame_count):.1f}")
    # cv2.destroyAllWindows()


def image_callback(msg):
    try:
        # Convert ROS Image message to OpenCV image
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        detect(cv_image)
    except CvBridgeError as e:
        rospy.logerr(f"CvBridge Error: {e}")


if __name__ == '__main__':
    rospy.init_node('lane_detection_camera')
    path = roslib.packages.get_pkg_dir("yolop")

    # Subscribe to the camera topic
    rospy.Subscriber("/cv_camera/image_raw", Image, image_callback)

    with torch.no_grad():
        # Initial call to detect in case of using image files
        if not rospy.has_param('/cv_camera/image_raw'):
            detect()

    rospy.spin()
    cv2.destroyAllWindows()
