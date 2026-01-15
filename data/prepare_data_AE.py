from PIL import Image
import os
import random


def augment_image_pairs(input_ir_folder, input_vis_folder,
                        output_ir_folder, output_vis_folder,
                        num_augmentations=5, crop_size=256, fill_color=0,
                        ir_ext='.png', vis_ext='.png'):
    """
    对红外和可见光图像对进行数据增强（保留可见光图像的色彩，仅将红外转为灰度图）

    参数：
    input_ir_folder: 输入红外图像文件夹路径
    input_vis_folder: 输入可见光图像文件夹路径
    output_ir_folder: 输出红外图像文件夹路径
    output_vis_folder: 输出可见光图像文件夹路径
    num_augmentations: 每对图像生成的增强样本数
    crop_size: 目标裁剪尺寸
    fill_color: 填充颜色值
    ir_ext: 红外图像保存格式后缀
    vis_ext: 可见光图像保存格式后缀
    """

    # 创建输出目录
    os.makedirs(output_ir_folder, exist_ok=True)
    os.makedirs(output_vis_folder, exist_ok=True)

    # 支持的图像格式
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')

    # 获取红外图像列表
    ir_files = [f for f in os.listdir(input_ir_folder) if f.lower().endswith(valid_exts)]

    for ir_file in ir_files:
        # 构建文件路径
        base_name = os.path.splitext(ir_file)[0]
        ir_path = os.path.join(input_ir_folder, ir_file)
        vis_path = os.path.join(input_vis_folder, base_name + vis_ext)

        # 跳过不存在的可见光图像
        if not os.path.exists(vis_path):
            print(f"Visible image {vis_path} not found, skipping...")
            continue

        try:
            # 打开图像对
            ir_img = Image.open(ir_path)
            vis_img = Image.open(vis_path)
        except Exception as e:
            print(f"Error opening image pair: {e}")
            continue

        # 统一图像尺寸
        ir_size = ir_img.size
        if vis_img.size != ir_size:
            print(f"Resizing visible image to match infrared size {ir_size}")
            vis_img = vis_img.resize(ir_size)

        # 生成多个增强样本
        for aug_idx in range(num_augmentations):
            # 复制原始图像
            ir = ir_img.copy()
            vis = vis_img.copy()

            # 填充和随机裁剪
            width, height = ir.size
            new_width = max(width, crop_size)
            new_height = max(height, crop_size)

            # 计算填充参数
            pad_w = max(new_width - width, 0)
            pad_h = max(new_height - height, 0)
            padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)

            # 应用填充
            ir_padded = Image.new(ir.mode, (new_width, new_height), fill_color)
            ir_padded.paste(ir, (padding[0], padding[1]))

            vis_padded = Image.new(vis.mode, (new_width, new_height), fill_color)
            vis_padded.paste(vis, (padding[0], padding[1]))

            # 随机裁剪位置（确保所有图像裁剪位置相同）
            x = random.randint(0, max(new_width - crop_size, 0))
            y = random.randint(0, max(new_height - crop_size, 0))

            # 执行裁剪
            ir_crop = ir_padded.crop((x, y, x + crop_size, y + crop_size))
            vis_crop = vis_padded.crop((x, y, x + crop_size, y + crop_size))

            # 随机旋转（0, 90, 180, 270度）
            angle = random.choice([0, 90, 180, 270])
            ir_rot = ir_crop.rotate(angle)
            vis_rot = vis_crop.rotate(angle)

            # 随机翻转
            if random.random() > 0.5:
                ir_rot = ir_rot.transpose(Image.FLIP_LEFT_RIGHT)
                vis_rot = vis_rot.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                ir_rot = ir_rot.transpose(Image.FLIP_TOP_BOTTOM)
                vis_rot = vis_rot.transpose(Image.FLIP_TOP_BOTTOM)

            # 仅将红外图像转换为灰度图，保留可见光图像的原始色彩
            ir_gray = ir_rot.convert('L')
            # 可见光图像不做灰度转换，保持原始模式（RGB或其他）
            vis_color = vis_rot

            # 保存结果
            ir_output = os.path.join(output_ir_folder, f"{base_name}_aug{aug_idx}{ir_ext}")
            vis_output = os.path.join(output_vis_folder, f"{base_name}_aug{aug_idx}{vis_ext}")

            ir_gray.save(ir_output)
            vis_color.save(vis_output)


if __name__ == "__main__":
    # 使用示例
    augment_image_pairs(
        input_ir_folder=r'/tmp/pycharm_project_659/Havard-Medical-Image-Fusion-Datasets-main/SPECT-MRI/MRI',
        input_vis_folder=r'/tmp/pycharm_project_659/Havard-Medical-Image-Fusion-Datasets-main/SPECT-MRI/SPECT',
        output_ir_folder=r'/tmp/pycharm_project_659/data/mrispect/mri',
        output_vis_folder=r'/tmp/pycharm_project_659/data/mrispect/spect',
        num_augmentations=20,
        ir_ext='.png',
        vis_ext='.png'
    )

