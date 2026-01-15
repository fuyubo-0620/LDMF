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

        # 获取并排序图像路径
        self.ir_name = natsorted(glob.glob(os.path.join(self.ir_path, '*.png')))
        self.vis_name = natsorted(glob.glob(os.path.join(self.vis_path, '*.png')))

        # 确保红外和可见光图像数量一致
        assert len(self.ir_name) == len(self.vis_name), \
            f"数量不匹配: IR={len(self.ir_name)}, VIS={len(self.vis_name)}"

    def __len__(self):
        return len(self.vis_name)

    def __getitem__(self, idx):
        try:
            # 读取红外图像（灰度）
            ir_img = cv2.imread(self.ir_name[idx], cv2.IMREAD_GRAYSCALE)
            if ir_img is None:
                raise ValueError(f"无法读取红外图像: {self.ir_name[idx]}")

            # 读取可见光图像（灰度）
            vis_img = cv2.imread(self.vis_name[idx], cv2.IMREAD_GRAYSCALE)
            if vis_img is None:
                raise ValueError(f"无法读取可见光图像: {self.vis_name[idx]}")

            # 确保所有图像尺寸一致
            if ir_img.shape != vis_img.shape:
                # 统一调整到相同尺寸（以红外图像尺寸为准）
                vis_img = cv2.resize(vis_img, (ir_img.shape[1], ir_img.shape[0]))

        except Exception as e:
            print(f"跳过样本 {idx}: {e}")

            next_idx = (idx + 1) % len(self)
            return self.__getitem__(next_idx)


        ir_img = np.expand_dims(ir_img, axis=2).astype('float32') / 255.0 * 2.0 - 1.0
        vis_img = np.expand_dims(vis_img, axis=2).astype('float32') / 255.0 * 2.0 - 1.0

        ir_tensor = torch.from_numpy(ir_img.transpose(2, 0, 1)).float()
        vis_tensor = torch.from_numpy(vis_img.transpose(2, 0, 1)).float()

        return vis_tensor, ir_tensor