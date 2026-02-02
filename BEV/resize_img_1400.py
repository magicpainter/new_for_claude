import os
import cv2
import numpy as np

def resize_image_1400(folder: str, filename: str, out_folder: str | None = None, overwrite: bool = False) -> str:
    """
    从 folder 读取 filename，对图像强制缩放到 1400x1400 并保存。
    - folder:     输入图片所在文件夹
    - filename:   输入图片文件名（如 'img.png'）
    - out_folder: 输出文件夹；为 None 时保存到原文件夹
    - overwrite:  是否覆盖原图；False 时在文件名后加 _1400x1400 后缀

    返回：保存后的文件完整路径
    """
    src_path = os.path.join(folder, filename)
    if not os.path.isfile(src_path):
        raise FileNotFoundError(f"输入文件不存在: {src_path}")

    # 读图
    img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"无法读取图像: {src_path}")

    h, w = img.shape[:2]
    target_size = (1400, 1400)

    # 选择插值方式：缩小时用 AREA，放大用 CUBIC
    if target_size[0] < w and target_size[1] < h:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_CUBIC

    resized = cv2.resize(img, target_size, interpolation=interp)

    # 输出路径
    out_folder = out_folder or folder
    os.makedirs(out_folder, exist_ok=True)
    name, ext = os.path.splitext(filename)
    out_name = f"{name}{ext}" if overwrite else f"{name}_1400x1400{ext}"
    out_path = os.path.join(out_folder, out_name)

    # 保存
    ok = cv2.imwrite(out_path, resized)
    if not ok:
        raise IOError(f"保存失败: {out_path}")

    return out_path

# 示例：
# saved = resize_image_1400("/path/to/images", "example.jpg")
# print("已保存到:", saved)
if __name__ == '__main__':
    saved = resize_image_1400("/home/ubuntu2/yhe/Projects/UniMatch-V2/irrelevant/", "RBEV_14k.png")
    print("已保存到:", saved)