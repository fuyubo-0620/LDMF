import os
import cv2
import glob
import torch
import numpy as np
import torch.utils.data as data
from natsort import natsorted


class TrainData(data.Dataset):
    def __init__(self):
        super().__init__()
        self.ir_path = r'/tmp/pycharm_project_659/data/m3fd/ir'
        self.vis_path = r'/tmp/pycharm_project_659/data/m3fd/vi'
        self.ir_name = natsorted(glob.glob(os.path.join(self.ir_path, '*.png')))
        self.vis_name = natsorted(glob.glob(os.path.join(self.vis_path, '*.png')))

        # 确保红外和可见光图像数量一致
        assert len(self.ir_name) == len(self.vis_name), \
            f"数量不匹配: IR={len(self.ir_name)}, VIS={len(self.vis_name)}"

    def __len__(self):
        return len(self.vis_name)

    def __getitem__(self, idx):
        try:
            # 读取红外图像
            ir_img = cv2.imread(self.ir_name[idx], cv2.IMREAD_GRAYSCALE)
            if ir_img is None:
                raise ValueError(f"Failed to read {self.ir_name[idx]}")

            # 读取可见光图像（彩色）并转换为Y通道
            vis_img = cv2.imread(self.vis_name[idx])
            if vis_img is None:
                raise ValueError(f"Failed to read {self.vis_name[idx]}")
            vis_img_ycrcb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2YCrCb)
            vis_img_y = vis_img_ycrcb[:, :, 0]  # 提取Y通道

            # 确保所有图像尺寸一致
            h, w = ir_img.shape[:2]
            if vis_img_y.shape[:2] != (h, w):
                vis_img_y = cv2.resize(vis_img_y, (w, h))

        except Exception as e:
            print(f"Skipping sample {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        # 统一处理流程
        ir_img = ir_img[:, :, np.newaxis].astype('float32') / 255.0
        vis_img_y = vis_img_y[:, :, np.newaxis].astype('float32') / 255.0

        ir_img = ir_img * 2.0 - 1.0
        vis_img_y = vis_img_y * 2.0 - 1.0

        # 转为tensor
        ir_img = torch.from_numpy(ir_img.transpose(2, 0, 1)).float()
        vis_img_y = torch.from_numpy(vis_img_y.transpose(2, 0, 1)).float()

        return vis_img_y, ir_img
