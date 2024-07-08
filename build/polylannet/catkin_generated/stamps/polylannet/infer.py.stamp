import os
import sys
import random
import logging
import argparse
import subprocess
from time import time
from torchvision import transforms
import rospy
from std_msgs.msg import Float32
import cv2
import numpy as np
import torch
import time

from lib.config import Config
from utils.evaluator import Evaluator
import roslib.packages

pathall = f'{roslib.packages.get_pkg_dir("polylannet")}/src'

cv2.namedWindow('Image')  # Create trackbars for each corner point
corners = np.float32([[0, 360], [0, 0], [640, 0], [640, 360]])


def update_corners(value):
    global corners
    # Update the corners based on the trackbar values
    corners[0] = [cv2.getTrackbarPos(
        'x1', 'Image'), cv2.getTrackbarPos('y1', 'Image')]
    corners[1] = [cv2.getTrackbarPos(
        'x2', 'Image'), cv2.getTrackbarPos('y2', 'Image')]
    corners[2] = [cv2.getTrackbarPos(
        'x3', 'Image'), cv2.getTrackbarPos('y3', 'Image')]
    corners[3] = [cv2.getTrackbarPos(
        'x4', 'Image'), cv2.getTrackbarPos('y4', 'Image')]


height, width = 360, 640
cv2.createTrackbar('x1', 'Image', corners[0][0], width, update_corners)
cv2.createTrackbar('y1', 'Image', corners[0][1], height, update_corners)
cv2.createTrackbar('x2', 'Image', corners[1][0], width, update_corners)
cv2.createTrackbar('y2', 'Image', corners[1][1], height, update_corners)
cv2.createTrackbar('x3', 'Image', corners[2][0], width, update_corners)
cv2.createTrackbar('y3', 'Image', corners[2][1], height, update_corners)
cv2.createTrackbar('x4', 'Image', corners[3][0], width, update_corners)
cv2.createTrackbar('y4', 'Image', corners[3][1], height, update_corners)


def warp_perspective(img, src_points, dst_points, output_size):
    # Compute perspective transform matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    # Warp the image
    warped_img = cv2.warpPerspective(img, M, output_size)
    return warped_img


def test(model, test_loader, exp_root, cfg, view, epoch, max_batches=None, verbose=True):
    fps_pub = rospy.Publisher('fps', Float32, queue_size=10)

    if verbose:
        logging.info("Starting testing.")

    # Test the model
    if epoch > 0:
        model.load_state_dict(torch.load(os.path.join(
            pathall, exp_root, "models", "model_{:03d}.pt".format(epoch)))['model'])
        # Initialize video capture
    video_path = f'{pathall}/jalan.mp4'  # Path to the input video
    output_video_path = 'output_video.mp4'  # Path to save the output video
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps,
                          (frame_width, frame_height))

    model.eval()
    prev_time = 0
    while cap.isOpened():
        ret, frame = cap.read()
        current_time = time.time()
        elapsed_time = current_time - prev_time
        prev_time = current_time
        fps = 1 / elapsed_time if elapsed_time > 0 else 0

        with torch.no_grad():
            # img = test_loader.dataset.__getitem__(frame)
            img_cv = cv2.resize(frame, (640, 360),
                                interpolation=cv2.INTER_LINEAR)
            update_corners(None)

            top_left = np.array([corners[0, 0], 0])
            top_right = np.array([corners[3, 0], 0])
            offset = [50, 0]
            # Save source points and destination points into arrays
            src_points = np.float32(
                [corners[0], corners[1], corners[2], corners[3]])
            dst_points = np.float32(
                [corners[0] + offset, top_left + offset, top_right - offset, corners[3] - offset])
            warped_img = warp_perspective(
                img_cv, src_points, dst_points, (width, height))
            cv2.imshow('Image', warped_img)

            img = test_loader.dataset.__getitem__(warped_img)
            images = img.unsqueeze(0)
            images = images.to(device)

            outputs = model(images)

            outputs = model.decode(outputs, **cfg.get_test_parameters())

            mask = np.zeros_like(img_cv)

            # Fill the ROI with white color
            roi_corners = np.float32(
                [corners[0], corners[1], corners[2], corners[3]])
            src_pts = roi_corners.reshape((-1, 1, 2)).astype("int32")
            cv2.polylines(mask, [src_pts], True, (0, 255, 255), thickness=5)

            # Apply the ROI mask to the segmented result

            # Overlay ROI on the image
            img_with_roi = cv2.addWeighted(img_cv, 0.5, mask, 0.5, 0)
            cv2.imshow('Image_with_roi', img_with_roi)

            # if view:
            outputs, extra_outputs = outputs
            preds = test_loader.dataset.draw_annotation(
                img_path=frame,
                pred=outputs[0].cpu().numpy(),
                cls_pred=extra_outputs[0].cpu().numpy() if extra_outputs is not None else None)
            # Ensure preds is a valid image matrix
            if isinstance(preds, np.ndarray):
                # Convert to BGR format if necessary
                preds = cv2.cvtColor(preds, cv2.COLOR_RGB2BGR)

            # Overlay FPS on the preds frame
            fps_text = "FPS: {:.2f}".format(fps)
            fps_pub.publish(fps)
            print(fps_text)
            cv2.imshow('pred', preds)
            cv2.waitKey(1)
            # Overlay FPS on the frame

    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Lane regression")
    parser.add_argument("--exp_name", default="exp", help="Experiment name")
    parser.add_argument(
        "--cfg", default="cfgs/tusimple_resnet50.yaml", help="Config file")
    parser.add_argument("--epoch", type=int, default=2695,
                        help="Epoch to test the model on")
    parser.add_argument("--batch_size", type=int,
                        help="Number of images per batch")
    parser.add_argument("--view", action="store_true", help="Show predictions")

    return parser.parse_args()


if __name__ == "__main__":
    rospy.init_node('polylannet')
    args = parse_args()
    cfg = Config(args.cfg)

    # Set up seeds
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])

    # Set up logging
    exp_root = os.path.join(cfg['exps_dir'], os.path.basename(
        os.path.normpath(args.exp_name)))

    logging.info("Experiment name: {}".format(args.exp_name))
    logging.info("Config:\n" + str(cfg))
    logging.info("Args:\n" + str(args))

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Hyper parameters
    num_epochs = cfg["epochs"]
    batch_size = cfg["batch_size"] if args.batch_size is None else args.batch_size

    # Model
    model = cfg.get_model().to(device)
    test_epoch = args.epoch

    # Get data set
    test_dataset = cfg.get_dataset("test")

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size if args.view is False else 1,
                                              shuffle=False,
                                              num_workers=8)
    # Eval results
    evaluator = Evaluator(test_loader.dataset, exp_root)

    mean_loss = test(model, test_loader, exp_root, cfg,
                     epoch=test_epoch, view=args.view)
