import cv2
import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmenters import Resize
from torchvision.transforms import ToTensor
from torch.utils.data.dataset import Dataset
from imgaug.augmentables.lines import LineString, LineStringsOnImage

from .elas import ELAS
from .llamas import LLAMAS
from .tusimple import TuSimple
from .nolabel_dataset import NoLabelDataset

GT_COLOR = (255, 0, 0)
PRED_HIT_COLOR = (0, 255, 0)
PRED_MISS_COLOR = (0, 0, 255)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


class LaneDataset(Dataset):
    def __init__(self,
                 dataset='tusimple',
                 augmentations=None,
                 normalize=False,
                 split='train',
                 img_size=(360, 640),
                 aug_chance=1.,
                 **kwargs):
        super(LaneDataset, self).__init__()

        self.img_h, self.img_w = img_size

        if augmentations is not None:
            # add augmentations
            augmentations = [getattr(iaa, aug['name'])(**aug['parameters'])
                             for aug in augmentations]  # add augmentation

        self.normalize = normalize
        transformations = iaa.Sequential([Resize({'height': self.img_h, 'width': self.img_w})])
        self.to_tensor = ToTensor()
        self.transform = iaa.Sequential([iaa.Sometimes(then_list=augmentations, p=aug_chance), transformations])


    def draw_annotation(self, img_path, pred=None, cls_pred=None):
        img = self.__getitem__(img_path)
        img = img.permute(1, 2, 0).numpy()
        # Unnormalize
        if self.normalize:
            img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
        img = (img * 255).astype(np.uint8)

        img_h, img_w, _ = img.shape

        if pred is None:
            return img

        # Draw predictions
        pred = pred[pred[:, 0] != 0]  # filter invalid lanes
        overlay = img.copy()
        for i, lane in enumerate(pred):
            print(lane)
            color = PRED_HIT_COLOR
            lane = lane[1:]  # remove conf
            lower, upper = lane[0], lane[1]
            lane = lane[2:]  # remove upper, lower positions
            # lane = lane[2:]  # elminate order 3 
            # generate points from the polynomial
            ys = np.linspace(lower, upper, num=100)
            points = np.zeros((len(ys), 2), dtype=np.int32)
            points[:, 1] = (ys * img_h).astype(int)
            points[:, 0] = (np.polyval(lane, ys) * img_w).astype(int)
            points = points[(points[:, 0] > 0) & (points[:, 0] < img_w)]

            # draw lane with a polyline on the overlay
            for current_point, next_point in zip(points[:-1], points[1:]):
                overlay = cv2.line(overlay, tuple(current_point), tuple(next_point), color=color, thickness=2)

            # draw class icon
            if cls_pred is not None and len(points) > 0:
                class_icon = self.dataset.get_class_icon(cls_pred[i])
                class_icon = cv2.resize(class_icon, (32, 32))
                mid = tuple(points[len(points) // 2] - 60)
                x, y = mid

                img[y:y + class_icon.shape[0], x:x + class_icon.shape[1]] = class_icon

        # Add lanes overlay
        w = 0.6
        img = ((1. - w) * img + w * overlay).astype(np.uint8)

        return img

    def lane_to_linestrings(self, lanes):
        lines = []
        for lane in lanes:
            lines.append(LineString(lane))

        return lines

    def linestrings_to_lanes(self, lines):
        lanes = []
        for line in lines:
            lanes.append(line.coords)

        return lanes

    def __getitem__(self, img):
        # img = cv2.imread(img_path)
        img= self.transform(image=img)

        img = img / 255.
        if self.normalize:
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = self.to_tensor(img.astype(np.float32))
        return img

    def __len__(self):
        return 1


def main():
    import torch
    from lib.config import Config
    np.random.seed(0)
    torch.manual_seed(0)
    cfg = Config('config.yaml')
    train_dataset = cfg.get_dataset('train')
    for idx in range(len(train_dataset)):
        img = train_dataset.draw_annotation(idx)
        cv2.imshow('sample', img)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
