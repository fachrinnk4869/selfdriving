import time
from pathlib import Path
import cv2
import torch
import rospy
from std_msgs.msg import Float32
import numpy as np
import roslib.packages

# Import necessary functions from utils.utils
from yolop.utils.utils import (
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

    left_curve_pub = rospy.Publisher('left_lane_curve', Float32, queue_size=10)
    fps_pub = rospy.Publisher('fps', Float32, queue_size=10)
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
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
            next(model.parameters())))  # run once
    t0 = time.time()
    frame_count = 0
    total_fps = 0
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

        fps = 1 / (time_synchronized() - t1)  # Forward pass FPS.
        t1 = time_synchronized()
        total_fps += fps

        # da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + \
                ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
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
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(
                        f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            # w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            # h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            w, h = im0.shape[1], im0.shape[0]
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(output_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    inf_time.update(t2 - t1, img.size(0))
    nms_time.update(t4 - t3, img.size(0))
    waste_time.update(tw2 - tw1, img.size(0))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' %
          (inf_time.avg, nms_time.avg))
    print(f'Done. ({time.time() - t0:.3f}s)')
    print(f"Average FPS: {(total_fps / frame_count):.1f}")
    # cv2.destroyALlWindows()


if __name__ == '__main__':
    rospy.init_node('lane_detection_node')
    path = roslib.packages.get_pkg_dir("yolop")
    # opt = rospy.get_param('~lane_detection_node_params')
    # print(opt)
    with torch.no_grad():
        detect()
    rospy.spin()
    # cv2.destroyAllWindows()
