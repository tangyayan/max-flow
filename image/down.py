import cv2
import numpy as np
from pathlib import Path
import argparse
from PIL import Image

def downsample_image(image, max_dimension=500, method='lanczos'):
    """
    降采样图像到指定最大尺寸
    
    Args:
        image: numpy array 或 PIL Image
        max_dimension: 最大边长
        method: 降采样方法 ('lanczos', 'bilinear', 'nearest', 'area')
    
    Returns:
        downsampled_image: 降采样后的图像
        scale_factor: 缩放比例
    """
    # 转换为 numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    height, width = image.shape[:2]
    
    # 计算缩放比例
    if max(height, width) <= max_dimension:
        print(f"图像尺寸 {width}x{height} 已在范围内，无需降采样")
        return image, 1.0
    
    scale_factor = max_dimension / max(height, width)
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # 选择插值方法
    interpolation_methods = {
        'lanczos': cv2.INTER_LANCZOS4,
        'bilinear': cv2.INTER_LINEAR,
        'nearest': cv2.INTER_NEAREST,
        'area': cv2.INTER_AREA,
        'cubic': cv2.INTER_CUBIC
    }
    
    interp = interpolation_methods.get(method.lower(), cv2.INTER_LANCZOS4)
    
    # 降采样
    downsampled = cv2.resize(image, (new_width, new_height), interpolation=interp)
    
    print(f"降采样: {width}x{height} -> {new_width}x{new_height} (缩放比例: {scale_factor:.2%})")
    
    return downsampled, scale_factor


def downsample_file(input_path, output_path=None, max_dimension=500, method='lanczos', quality=95):
    """
    降采样单个图像文件
    
    Args:
        input_path: 输入图像路径
        output_path: 输出路径（None 则覆盖原文件）
        max_dimension: 最大边长
        method: 降采样方法
        quality: JPEG 质量 (1-100)
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        print(f"错误: 文件不存在 {input_path}")
        return False
    
    # 读取图像
    try:
        image = cv2.imread(str(input_path))
        if image is None:
            raise ValueError("无法读取图像")
    except Exception as e:
        print(f"错误: 读取图像失败 {input_path}: {e}")
        return False
    
    # 降采样
    downsampled, scale = downsample_image(image, max_dimension, method)
    
    if scale == 1.0:
        return True  # 无需处理
    
    # 确定输出路径
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_down{input_path.suffix}"
    else:
        output_path = Path(output_path)
    
    # 保存
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix.lower() in ['.jpg', '.jpeg']:
            cv2.imwrite(str(output_path), downsampled, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif output_path.suffix.lower() == '.png':
            cv2.imwrite(str(output_path), downsampled, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        else:
            cv2.imwrite(str(output_path), downsampled)
        
        print(f"保存到: {output_path}")
        return True
    except Exception as e:
        print(f"错误: 保存失败 {output_path}: {e}")
        return False


def downsample_folder(input_folder, output_folder=None, max_dimension=500, method='lanczos', 
                      extensions=None, recursive=False):
    """
    批量降采样文件夹中的图像
    
    Args:
        input_folder: 输入文件夹
        output_folder: 输出文件夹（None 则在原文件夹生成）
        max_dimension: 最大边长
        method: 降采样方法
        extensions: 文件扩展名列表
        recursive: 是否递归处理子文件夹
    """
    input_folder = Path(input_folder)
    
    if not input_folder.exists():
        print(f"错误: 文件夹不存在 {input_folder}")
        return
    
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    # 查找图像文件
    if recursive:
        image_files = []
        for ext in extensions:
            image_files.extend(input_folder.rglob(f"*{ext}"))
            image_files.extend(input_folder.rglob(f"*{ext.upper()}"))
    else:
        image_files = []
        for ext in extensions:
            image_files.extend(input_folder.glob(f"*{ext}"))
            image_files.extend(input_folder.glob(f"*{ext.upper()}"))
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 处理每个文件
    success_count = 0
    for i, img_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] 处理: {img_path.name}")
        
        # 确定输出路径
        if output_folder:
            output_folder = Path(output_folder)
            relative_path = img_path.relative_to(input_folder)
            out_path = output_folder / relative_path
        else:
            out_path = None
        
        if downsample_file(img_path, out_path, max_dimension, method):
            success_count += 1
    
    print(f"\n完成! 成功处理 {success_count}/{len(image_files)} 个文件")


def compare_images(original_path, downsampled_path):
    """
    对比原始图像和降采样后的图像
    """
    orig = cv2.imread(str(original_path))
    down = cv2.imread(str(downsampled_path))
    
    if orig is None or down is None:
        print("错误: 无法读取图像")
        return
    
    print(f"原始图像: {orig.shape[1]}x{orig.shape[0]}, 大小: {Path(original_path).stat().st_size / 1024:.1f} KB")
    print(f"降采样图像: {down.shape[1]}x{down.shape[0]}, 大小: {Path(downsampled_path).stat().st_size / 1024:.1f} KB")
    
    # 计算 PSNR 和 SSIM (需要尺寸一致)
    if orig.shape == down.shape:
        psnr = cv2.PSNR(orig, down)
        print(f"PSNR: {psnr:.2f} dB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="图像降采样工具")
    parser.add_argument("input", help="输入图像文件或文件夹路径")
    parser.add_argument("-o", "--output", help="输出路径（可选）")
    parser.add_argument("-m", "--max-dimension", type=int, default=500, 
                        help="最大边长 (默认: 500)")
    parser.add_argument("--method", choices=['lanczos', 'bilinear', 'nearest', 'area', 'cubic'],
                        default='lanczos', help="降采样方法 (默认: lanczos)")
    parser.add_argument("-q", "--quality", type=int, default=95,
                        help="JPEG 质量 1-100 (默认: 95)")
    parser.add_argument("-r", "--recursive", action="store_true",
                        help="递归处理子文件夹")
    parser.add_argument("--compare", help="对比两个图像")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_images(args.input, args.compare)
    else:
        input_path = Path(args.input)
        
        if input_path.is_file():
            # 处理单个文件
            downsample_file(input_path, args.output, args.max_dimension, args.method, args.quality)
        elif input_path.is_dir():
            # 处理文件夹
            downsample_folder(input_path, args.output, args.max_dimension, args.method, 
                            recursive=args.recursive)
        else:
            print(f"错误: 无效的路径 {input_path}")