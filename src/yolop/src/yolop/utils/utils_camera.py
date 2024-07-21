import datetime
import logging
import os
import platform
import subprocess
import time
from pathlib import Path
import re
import glob
import random
import cv2
import numpy as np
import torch
import torchvision

logger = logging.getLogger(__name__)


def git_describe(path=Path(__file__).parent):  # path must be a directory
    # return human-readable git description, i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe
    s = f'git -C {path} describe --tags --long --always'
    try:
        return subprocess.check_output(s, shell=True, stderr=subprocess.STDOUT).decode()[:-1]
    except subprocess.CalledProcessError as e:
        return ''  # not a git repository


def date_modified(path=__file__):
    # return human-readable file modification date, i.e. '2021-3-26'
    t = datetime.datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f'{t.year}-{t.month}-{t.day}'


def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    # string
    s = f'YOLOPv2 🚀 {git_describe() or date_modified()} torch {torch.__version__} '
    cpu = device.lower() == 'cpu'
    if cpu:
        # force torch.cuda.is_available() = False
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(
        ), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        n = torch.cuda.device_count()
        if n > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * len(s)
        for i, d in enumerate(device.split(',') if device else range(n)):
            p = torch.cuda.get_device_properties(i)
            # bytes to MB
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"
    else:
        s += 'CPU\n'

    logger.info(s.encode().decode('ascii', 'ignore')
                if platform.system() == 'Windows' else s)  # emoji-safe
    return torch.device('cuda:0' if cuda else 'cpu')


def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, [0, 255, 255],
                  thickness=2, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3


class SegmentationMetric(object):
    '''
    imgLabel [batch_size, height(144), width(256)]
    confusionMatrix [[0(TN),1(FP)],
                     [2(FN),3(TP)]]
    '''

    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,)*2)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def lineAccuracy(self):
        Acc = np.diag(self.confusionMatrix) / \
            (self.confusionMatrix.sum(axis=1) + 1e-12)
        return Acc[1]

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / \
            (self.confusionMatrix.sum(axis=0) + 1e-12)
        return classAcc

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + \
            np.sum(self.confusionMatrix, axis=0) - \
            np.diag(self.confusionMatrix)
        IoU = intersection / union
        IoU[np.isnan(IoU)] = 0
        mIoU = np.nanmean(IoU)
        return mIoU

    def IntersectionOverUnion(self):
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + \
            np.sum(self.confusionMatrix, axis=0) - \
            np.diag(self.confusionMatrix)
        IoU = intersection / union
        IoU[np.isnan(IoU)] = 0
        return IoU[1]

    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        # print(imgLabel.shape)
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusionMatrix, axis=1) / \
            np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
            np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
            np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def _make_grid(nx=20, ny=20):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


def split_for_trace_model(pred=None, anchor_grid=None):
    z = []
    st = [8, 16, 32]
    for i in range(3):
        bs, _, ny, nx = pred[i].shape
        pred[i] = pred[i].view(bs, 3, 85, ny, nx).permute(
            0, 1, 3, 4, 2).contiguous()
        y = pred[i].sigmoid()
        gr = _make_grid(nx, ny).to(pred[i].device)
        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + gr) * st[i]  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
        z.append(y.view(bs, -1, 85))
    pred = torch.cat(z, 1)
    return pred


def warp_perspective(img, src_points, dst_points, output_size):
    # Compute perspective transform matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    # Warp the image
    warped_img = cv2.warpPerspective(img, M, output_size)
    return warped_img


def find_nearest_lane(prediction, center, selected_y):
    # Assuming prediction is an image where blue pixels (0, 0, 255) indicate lane detections

    # Get the pixels in the selected row (selected_y) where lane detection is present (blue)
    blue_pixels = np.where(
        (prediction[selected_y] == [0, 0, 255]).all(axis=-1))

    # Extract x-coordinates of blue pixels
    nonzero_x = blue_pixels[0]

    # nonzero = prediction[selected_y].nonzero()
    # nonzero_x = np.array(nonzero)
    # Find the left and right lanes based on the center
    mask_left = nonzero_x <= center
    mask_right = nonzero_x > center

    # Extract x-coordinates of left and right lanes
    nonzero_x_left = nonzero_x[mask_left]
    nonzero_x_right = nonzero_x[mask_right]

    # Find the nearest x-coordinates for left and right lanes
    try:
        left = nonzero_x_left[-1]
    except IndexError as e:
        left = 0
    # Find the nearest x-coordinates for left and right lanes
    try:
        right = nonzero_x_right[0]
    except IndexError as e:
        right = 0
    # print(left)

    return left, right


# corners = np.float32([[0, 554], [291, 350], [954, 371], [1270, 578]])
# corners = np.float32([[253, 697], [585, 456], [700, 456], [1061, 690]])
corners = np.float32([[0, 697], [300, 325], [980, 325], [1280, 690]])


class ROITrackbar():
    def __init__(self, size):
        global corners
        cv2.namedWindow('Image')  # Create trackbars for each corner point
        height, width = size.shape[:2]
        cv2.createTrackbar('x1', 'Image', 253, width, self.update_corners)
        cv2.createTrackbar('y1', 'Image', 697, height, self.update_corners)
        cv2.createTrackbar('x2', 'Image', 585, width, self.update_corners)
        cv2.createTrackbar('y2', 'Image', 456, height, self.update_corners)
        cv2.createTrackbar('x3', 'Image', 700, width, self.update_corners)
        cv2.createTrackbar('y3', 'Image', 456, height, self.update_corners)
        cv2.createTrackbar('x4', 'Image', 1061, width, self.update_corners)
        cv2.createTrackbar('y4', 'Image', 690, height, self.update_corners)
        self.update_corners(0)
    # Callback function for the trackbar

    def update_corners(self, value):
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
        # Apply perspective transformation


def show_seg_result(img, result, palette=None, is_demo=True):
    global corners
    if palette is None:
        palette = np.random.randint(
            0, 255, size=(3, 3))
    palette[0] = [0, 0, 0]
    palette[1] = [0, 255, 0]
    palette[2] = [255, 0, 0]
    palette = np.array(palette)
    assert palette.shape[0] == 3  # len(classes)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2

    if not is_demo:
        color_seg = np.zeros(
            (result.shape[0], result.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[result == label, :] = color
        print(result.shape)
    elif type(result) != tuple:
        color_area = np.zeros(
            (result.shape[0], result.shape[1], 3), dtype=np.uint8)

        color_area[result == 1] = [0, 0, 255]  # lane line
        color_seg = color_area
        complete_lane_lines = result
    else:
        color_area = np.zeros(
            (result[0].shape[0], result[0].shape[1], 3), dtype=np.uint8)

        color_area[result[0] == 1] = [0, 255, 0]  # segmentation area
        color_area[result[1] == 1] = [0, 0, 255]  # lane line
        color_seg = color_area
        complete_lane_lines = result[1]

    # convert to BGR
    color_seg = color_seg[..., ::-1]
    color_mask = np.mean(color_seg, 2)

    # image dimensions
    height, width = img.shape[:2]

    # Save top left and right explicitly and offset
    top_left = np.array([corners[0, 0], 0])
    top_right = np.array([corners[3, 0], 0])
    offset = [50, 0]

    # Save source points and destination points into arrays
    src_points = np.float32([corners[0], corners[1], corners[2], corners[3]])
    dst_points = np.float32(
        [corners[0] + offset, top_left + offset, top_right - offset, corners[3] - offset])
    warped_img = warp_perspective(img, src_points, dst_points, (width, height))

    # cv2.imshow('bird_eye', warped_img)
    raw_image = warped_img.copy()
    img[color_mask != 0] = color_seg[color_mask != 0]

    # Create a blank mask
    mask = np.zeros_like(img)

    # Give the ROI border with yellow color
    roi_corners = np.float32([corners[0], corners[1], corners[2], corners[3]])
    src_pts = roi_corners.reshape((-1, 1, 2)).astype("int32")
    cv2.polylines(mask, [src_pts], True, (0, 255, 255), thickness=5)

    # Apply the ROI mask to the segmented result
    masked_seg = cv2.bitwise_and(color_seg, mask)

    # Overlay ROI on the image
    img_with_roi = cv2.addWeighted(img, 1, mask, 0.5, 0)

    # Overlay segmented result on the image with ROI
    img_with_seg = img_with_roi.copy()
    img_with_seg[masked_seg == 0] = 0
    # Find the center of right and left line
    center = int(width // 2)
    # Choose the nearest lane line for each side (considering a distance threshold)
    src_points = np.float32([corners[0], corners[1], corners[2], corners[3]])
    dst_points = np.float32(
        [corners[0] + offset, top_left + offset, top_right - offset, corners[3] - offset])
    color_area = np.zeros((height, width, 3), dtype=np.uint8)

    # Initialize empty lists to store pixel positions
    left_x_list = []
    left_y_list = []
    right_x_list = []
    right_y_list = []

    color_area[complete_lane_lines == 1] = [0, 0, 255]  # lane line
    bir_eye_result = warp_perspective(
        color_area, src_points, dst_points, (width, height))

    for i in range(0, height, 20):
        # Find nearest left and right lane x-coordinates for the current row
        left_nearest_x, right_nearest_x = find_nearest_lane(
            bir_eye_result, center, i)

        # Check if both left and right lane x-coordinates are found
        if left_nearest_x != 0:
            # Append x and y positions to respective lists
            left_x_list.append(left_nearest_x)
            left_y_list.append(i)
            cv2.circle(raw_image, (left_nearest_x, i), 0,
                       (0, 0, 255), 5)  # Red circle for left lane

        if right_nearest_x != 0:
            # Append x and y positions to respective lists
            right_x_list.append(right_nearest_x)
            right_y_list.append(i)
            # Draw circles on the raw image for visualization
            cv2.circle(raw_image, (right_nearest_x, i), 0,
                       (0, 0, 255), 5)  # Red circle for right lane

    # cv2.imshow('raw_image', raw_image)
    # cv2.imshow('Image_with_roi', img_with_roi)
    # cv2.waitKey(1)

    # Convert lists to NumPy arrays
    left_x = np.array(left_x_list)
    left_y = np.array(left_y_list)
    right_x = np.array(right_x_list)
    right_y = np.array(right_y_list)

    # Set a threshold for removing outliers
    threshold = 100  # Adjust this threshold as needed

    # Find the indices of elements that are within the threshold
    left_inliers = np.abs(left_x - np.mean(left_x)) < threshold
    right_inliers = np.abs(right_x - np.mean(right_x)) < threshold

    # Filter out outliers using the indices
    left_x_filtered = left_x[left_inliers]
    left_y_filtered = left_y[left_inliers]
    right_x_filtered = right_x[right_inliers]
    right_y_filtered = right_y[right_inliers]
    try:
        # Fit a second-order polynomial to the filtered data
        left_fit = np.polyfit(left_y_filtered, left_x_filtered, 2)
    except TypeError:
        print("Filtered data is empty. Unable to fit a polynomial.")
    try:
        # Fit a second-order polynomial to the filtered data
        right_fit = np.polyfit(right_y_filtered, right_x_filtered, 2)
    except TypeError:
        print("Filtered data is empty. Unable to fit a polynomial.")

    # Generate x and y values for plotting
    plot_y = np.linspace(0, warped_img.shape[0]-1, warped_img.shape[0])
    try:
        left_fit_x = left_fit[0]*plot_y**2 + left_fit[1]*plot_y + left_fit[2]
    except UnboundLocalError as e:
        print(e)
    try:
        right_fit_x = right_fit[0]*plot_y**2 + \
            right_fit[1]*plot_y + right_fit[2]
    except UnboundLocalError as e:
        print(e)
    # Get binary warped image size
    image_size = warped_img.shape

    # Get max of plot_y
    y_eval = np.max(plot_y)

    # Define conversions in x and y from pixels space to meters
    y_m_per_pix = 30/720
    x_m_per_pix = 3.7/700

    try:
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(left_y*y_m_per_pix, left_x*x_m_per_pix, 2)
    except TypeError:
        print("Filtered data is empty. Unable to fit a polynomial.")
    try:
        # Fit new polynomials to x,y in world space
        right_fit_cr = np.polyfit(right_y*y_m_per_pix, right_x*x_m_per_pix, 2)
    except TypeError or UnboundLocalError:
        print("Filtered data is empty. Unable to fit a polynomial.")
    # Calculate radius of curve
    try:
        left_curve = ((1+(2*left_fit_cr[0]*y_eval*y_m_per_pix +
                      left_fit_cr[1])**2)**1.5)/np.absolute(2*left_fit_cr[0])
    except:
        left_curve = None
        print("Filtered data is empty. Unable to fit a polynomial.")
    try:
        right_curve = ((1+(2*right_fit_cr[0]*y_eval*y_m_per_pix +
                       right_fit_cr[1])**2)**1.5)/np.absolute(2*right_fit_cr[0])
    except:
        right_curve = None
        print("Filtered data is empty. Unable to fit a polynomial.")
    # Calculate lane deviation from center of lane
    scene_height = image_size[0] * y_m_per_pix
    scene_width = image_size[1] * x_m_per_pix
    # Initialize an empty image to draw the lane information
    output_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    try:
        # Calculate the intercept points at the bottom of our image
        left_intercept = left_fit_cr[0] * scene_height ** 2 + \
            left_fit_cr[1] * scene_height + left_fit_cr[2]
        # Draw the detected lane lines on the output image
        cv2.polylines(output_img, [np.array(
            list(zip(left_fit_x, plot_y)), np.int32)], False, (255, 0, 0), thickness=5)
        cv2.putText(output_img, 'Left Lane Curvature: {:.2f} m'.format(
            left_curve), (50, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    except:
        print()
    try:
        # Calculate the intercept points at the bottom of our image
        right_intercept = right_fit_cr[0] * scene_height ** 2 + \
            right_fit_cr[1] * scene_height + right_fit_cr[2]
        # Draw the detected lane lines on the output image
        cv2.polylines(output_img, [np.array(
            list(zip(right_fit_x, plot_y)), np.int32)], False, (255, 0, 0), thickness=5)
        cv2.putText(output_img, 'Right Lane Curvature: {:.2f} m'.format(
            right_curve), (50, 150), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    except:
        print("hey")

    try:
        center = (left_intercept + right_intercept) / 2.0

    # Use intercept points to calculate the lane deviation of the vehicle
        lane_deviation = (center - scene_width / 2.0)

    # Write text to display lane curvature and deviation
        cv2.putText(output_img, 'Lane Deviation: {:.2f} m'.format(
            lane_deviation), (50, 200), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    except:
        print("")
    # equation for center
    # try:
    #     # Fit new polynomials to x,y in world space
    #     center_fit_cr = np.sum([left_fit_cr,right_fit_cr],axis=0)
    # except TypeError or UnboundLocalError:
    #     print("Filtered data is empty. Unable to fit a polynomial.")

    # try:
    #     center_curve = ((1+(2*center_fit_cr[0]*y_eval*y_m_per_pix+center_fit_cr[1])**2)**1.5)/np.absolute(2*center_fit_cr[0])
    # except:
    #     center_curve = None
    #     print("Filtered data is empty. Unable to fit a polynomial.")
    # Combine the result with the original image
    result = cv2.addWeighted(output_img, 1, raw_image, 1, 0)
    # Display the output image
    # cv2.imshow('Output Image', result)
    cv2.waitKey(1)

    # Warp the image and ROI to bird's eye perspective
    src_points = np.float32([corners[0], corners[1], corners[2], corners[3]])
    dst_points = np.float32(
        [corners[0] + offset, top_left + offset, top_right - offset, corners[3] - offset])
    warped_img = warp_perspective(
        img_with_roi, src_points, dst_points, (width, height))
    # cv2.imshow('bird_eye2', warped_img)
    # cv2.waitKey(1)
    return result, left_curve, right_curve, None


def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0],
                   img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / \
            2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def set_logging(rank=-1):
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if rank in [-1, 0] else logging.WARN)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=()):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # (pixels) minimum and maximum box width and height
    min_wh, max_wh = 2, 4096
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)
              ] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[
                conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float(
            ) / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    # iou = inter / (area1 + area2 - inter)
    return inter / (area1[:, None] + area2 - inter)


class LoadImages:  # for inference
    def __init__(self, path, img_size=640, stride=32):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff',
                       'dng', 'webp', 'mpo']  # acceptable image suffixes
        vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg',
                       'm4v', 'wmv', 'mkv']  # acceptable video suffixes
        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print(
                f'video {self.count + 1}/{self.nf} ({self.frame}/{self.nframes}) {path}: ', end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            # print(f'image {self.count}/{self.nf} {path}: ', end='')

        # Padded resize
        img0 = cv2.resize(img0, (1280, 720), interpolation=cv2.INTER_LINEAR)
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


class LoadCVImages:  # for inference
    def __init__(self, path, img_size=640, stride=32):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff',
                       'dng', 'webp', 'mpo']  # acceptable image suffixes
        vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg',
                       'm4v', 'wmv', 'mkv']  # acceptable video suffixes
        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print(
                f'video {self.count + 1}/{self.nf} ({self.frame}/{self.nframes}) {path}: ', end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            # print(f'image {self.count}/{self.nf} {path}: ', end='')

        # Padded resize
        img0 = cv2.resize(img0, (1280, 720), interpolation=cv2.INTER_LINEAR)
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf


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


def driving_area_mask(seg=None):
    da_predict = seg[:, :, 12:372, :]
    da_seg_mask = torch.nn.functional.interpolate(
        da_predict, scale_factor=2, mode='bilinear')
    _, da_seg_mask = torch.max(da_seg_mask, 1)
    da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
    return da_seg_mask


def lane_line_mask(ll=None):
    ll_predict = ll[:, :, 12:372, :]
    ll_seg_mask = torch.nn.functional.interpolate(
        ll_predict, scale_factor=2, mode='bilinear')
    ll_seg_mask = torch.round(ll_seg_mask).squeeze(1)
    ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
    return ll_seg_mask
