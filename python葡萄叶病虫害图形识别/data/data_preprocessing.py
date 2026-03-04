import os
import shutil
import random
import numpy as np
import cv2
import requests
from tqdm import tqdm
from PIL import Image
import zipfile
import tarfile
import glob

# 配置参数
class Config:
    def __init__(self):
        self.raw_data_dir = 'data/raw'
        self.processed_data_dir = 'data/dataset'
        self.image_size = (224, 224)
        self.augmentation_factor = 5  # 每张原始图像生成的增强图像数量
        self.train_ratio = 0.7
        self.val_ratio = 0.2
        self.test_ratio = 0.1
        self.class_names = ['健康叶片', '霜霉病', '白粉病', '黑腐病', '叶斑病']
        self.dataset_urls = {
            'plantvillage': 'https://data.mendeley.com/public-files/datasets/tywbtsjrjv/files/d5652a28-c1d8-4b76-97f3-72fb80f94efc/file_downloaded',
            # 可以添加更多数据集的URL
        }

# 创建目录结构
def create_directory_structure(config):
    """创建必要的目录结构"""
    # 创建原始数据目录
    os.makedirs(config.raw_data_dir, exist_ok=True)
    
    # 创建处理后的数据目录及子目录
    os.makedirs(config.processed_data_dir, exist_ok=True)
    
    # 创建训练、验证和测试集目录
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(config.processed_data_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        # 为每个类别创建子目录
        for class_name in config.class_names:
            class_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

# 下载数据集
def download_dataset(url, save_path):
    """从指定URL下载数据集"""
    try:
        print(f"正在下载数据集: {url}")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1MB
        
        with open(save_path, 'wb') as file, tqdm(
            desc=os.path.basename(save_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(block_size):
                file.write(data)
                progress_bar.update(len(data))
        
        print(f"数据集下载完成，保存至: {save_path}")
        return True
    except Exception as e:
        print(f"下载数据集失败: {e}")
        return False

# 解压数据集
def extract_dataset(file_path, extract_dir):
    """解压数据集文件"""
    try:
        print(f"正在解压数据集: {file_path}")
        
        if file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        elif file_path.endswith('.tar.gz') or file_path.endswith('.tgz'):
            with tarfile.open(file_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_dir)
        elif file_path.endswith('.tar'):
            with tarfile.open(file_path, 'r') as tar_ref:
                tar_ref.extractall(extract_dir)
        
        print(f"数据集解压完成，提取至: {extract_dir}")
        return True
    except Exception as e:
        print(f"解压数据集失败: {e}")
        return False

# 数据清洗
def clean_data(raw_data_dir, min_size=50):
    """清洗数据，移除损坏或过小的图像"""
    print("正在清洗数据...")
    
    # 遍历所有子目录
    for root, dirs, files in os.walk(raw_data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # 尝试打开图像
                with Image.open(file_path) as img:
                    # 检查图像尺寸
                    if img.width < min_size or img.height < min_size:
                        print(f"移除过小的图像: {file_path}")
                        os.remove(file_path)
            except Exception as e:
                print(f"移除损坏的图像: {file_path}")
                os.remove(file_path)
    
    print("数据清洗完成")

# 图像增强函数
def augment_image(img_path, output_dir, filename, config):
    """对图像进行数据增强"""
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图像: {img_path}")
        return []
    
    # 转换为RGB格式
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 获取图像名称和扩展名
    name, ext = os.path.splitext(filename)
    
    augmented_images = []
    
    # 原始图像（调整大小）
    resized_img = cv2.resize(img, config.image_size)
    original_output_path = os.path.join(output_dir, f'{name}_original{ext}')
    cv2.imwrite(original_output_path, cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR))
    augmented_images.append(original_output_path)
    
    # 生成增强图像
    for i in range(config.augmentation_factor):
        augmented_img = resized_img.copy()
        
        # 随机旋转（-15到15度）
        angle = random.randint(-15, 15)
        M = cv2.getRotationMatrix2D((config.image_size[0]//2, config.image_size[1]//2), angle, 1)
        augmented_img = cv2.warpAffine(augmented_img, M, config.image_size)
        
        # 随机水平翻转
        if random.random() > 0.5:
            augmented_img = cv2.flip(augmented_img, 1)
        
        # 随机垂直翻转
        if random.random() > 0.5:
            augmented_img = cv2.flip(augmented_img, 0)
        
        # 随机亮度调整
        brightness_factor = random.uniform(0.8, 1.2)
        hsv = cv2.cvtColor(augmented_img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, int(50 * (brightness_factor - 1)))
        v = np.clip(v, 0, 255)
        hsv_bright = cv2.merge((h, s, v))
        augmented_img = cv2.cvtColor(hsv_bright, cv2.COLOR_HSV2RGB)
        
        # 随机对比度调整
        contrast_factor = random.uniform(0.8, 1.2)
        augmented_img = cv2.convertScaleAbs(augmented_img, alpha=contrast_factor, beta=0)
        
        # 随机裁剪和调整大小
        if random.random() > 0.5:
            crop_size = int(min(config.image_size) * random.uniform(0.8, 1.0))
            x = random.randint(0, config.image_size[0] - crop_size)
            y = random.randint(0, config.image_size[1] - crop_size)
            augmented_img = augmented_img[y:y+crop_size, x:x+crop_size]
            augmented_img = cv2.resize(augmented_img, config.image_size)
        
        # 添加高斯噪声
        if random.random() > 0.7:
            noise = np.random.normal(0, 15, augmented_img.shape).astype(np.uint8)
            augmented_img = cv2.add(augmented_img, noise)
        
        # 保存增强后的图像
        output_path = os.path.join(output_dir, f'{name}_augmented_{i}{ext}')
        cv2.imwrite(output_path, cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR))
        augmented_images.append(output_path)
    
    return augmented_images

# 划分数据集
def split_dataset(config):
    """划分数据集为训练集、验证集和测试集"""
    print("正在划分数据集...")
    
    # 遍历每个类别
    for class_name in config.class_names:
        class_dir = os.path.join(config.raw_data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"警告: 类别目录不存在: {class_dir}")
            continue
        
        # 获取该类别下的所有图像文件
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(class_dir, ext)))
        
        # 打乱文件列表
        random.shuffle(image_files)
        
        # 计算划分索引
        total_files = len(image_files)
        train_split = int(total_files * config.train_ratio)
        val_split = int(total_files * (config.train_ratio + config.val_ratio))
        
        # 划分文件
        train_files = image_files[:train_split]
        val_files = image_files[train_split:val_split]
        test_files = image_files[val_split:]
        
        # 处理训练集
        for file_path in tqdm(train_files, desc=f"处理训练集 - {class_name}"):
            filename = os.path.basename(file_path)
            output_dir = os.path.join(config.processed_data_dir, 'train', class_name)
            augment_image(file_path, output_dir, filename, config)
        
        # 处理验证集
        for file_path in tqdm(val_files, desc=f"处理验证集 - {class_name}"):
            filename = os.path.basename(file_path)
            output_dir = os.path.join(config.processed_data_dir, 'val', class_name)
            augment_image(file_path, output_dir, filename, config)
        
        # 处理测试集（不进行数据增强，只调整大小）
        for file_path in tqdm(test_files, desc=f"处理测试集 - {class_name}"):
            filename = os.path.basename(file_path)
            output_dir = os.path.join(config.processed_data_dir, 'test', class_name)
            
            # 只调整大小，不进行其他增强
            img = cv2.imread(file_path)
            if img is not None:
                img = cv2.resize(img, config.image_size)
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, img)
    
    print("数据集划分完成")

# 统计数据集信息
def count_dataset_info(config):
    """统计数据集的基本信息"""
    print("\n数据集统计信息:")
    
    # 统计每个分割集的样本数量
    for split in ['train', 'val', 'test']:
        total_count = 0
        print(f"\n{split}集:")
        
        for class_name in config.class_names:
            class_dir = os.path.join(config.processed_data_dir, split, class_name)
            if os.path.exists(class_dir):
                count = len([f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))])
                print(f"  {class_name}: {count}个样本")
                total_count += count
        
        print(f"  总计: {total_count}个样本")

# 创建样本数据集（用于测试）
def create_sample_dataset(config):
    """创建一个小型的样本数据集，用于演示和测试"""
    print("\n正在创建样本数据集...")
    
    # 创建样本数据目录
    sample_dir = os.path.join(config.processed_data_dir, 'sample')
    os.makedirs(sample_dir, exist_ok=True)
    
    # 为每个类别创建一些样本图像
    for class_idx, class_name in enumerate(config.class_names):
        class_sample_dir = os.path.join(sample_dir, class_name)
        os.makedirs(class_sample_dir, exist_ok=True)
        
        # 创建5个样本图像
        for i in range(5):
            # 创建一个随机的RGB图像
            img = np.ones((*config.image_size, 3), dtype=np.uint8) * 255  # 白色背景
            
            # 根据类别添加不同的特征
            if class_name == '健康叶片':
                # 健康叶片 - 绿色
                cv2.rectangle(img, (20, 20), (config.image_size[0]-20, config.image_size[1]-20), (0, 200, 0), -1)
                cv2.circle(img, (config.image_size[0]//2, config.image_size[1]//2), 50, (0, 255, 0), -1)
            elif class_name == '霜霉病':
                # 霜霉病 - 黄色背景加白色斑点
                cv2.rectangle(img, (0, 0), config.image_size, (255, 255, 0), -1)
                for _ in range(30):
                    x = random.randint(10, config.image_size[0]-10)
                    y = random.randint(10, config.image_size[1]-10)
                    r = random.randint(2, 8)
                    cv2.circle(img, (x, y), r, (255, 255, 255), -1)
            elif class_name == '白粉病':
                # 白粉病 - 绿色背景加白色粉末状区域
                cv2.rectangle(img, (0, 0), config.image_size, (0, 200, 0), -1)
                for _ in range(20):
                    x = random.randint(10, config.image_size[0]-50)
                    y = random.randint(10, config.image_size[1]-50)
                    w = random.randint(10, 50)
                    h = random.randint(10, 50)
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), -1)
            elif class_name == '黑腐病':
                # 黑腐病 - 绿色背景加黑色斑点
                cv2.rectangle(img, (0, 0), config.image_size, (0, 200, 0), -1)
                for _ in range(15):
                    x = random.randint(10, config.image_size[0]-10)
                    y = random.randint(10, config.image_size[1]-10)
                    r = random.randint(5, 15)
                    cv2.circle(img, (x, y), r, (0, 0, 0), -1)
            else:
                # 叶斑病 - 绿色背景加褐色斑点
                cv2.rectangle(img, (0, 0), config.image_size, (0, 200, 0), -1)
                for _ in range(20):
                    x = random.randint(10, config.image_size[0]-10)
                    y = random.randint(10, config.image_size[1]-10)
                    r = random.randint(3, 10)
                    cv2.circle(img, (x, y), r, (165, 42, 42), -1)
            
            # 保存样本图像
            sample_path = os.path.join(class_sample_dir, f'sample_{i}.jpg')
            cv2.imwrite(sample_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    print("样本数据集创建完成")

# 主函数
def main():
    """主函数"""
    # 创建配置对象
    config = Config()
    
    # 创建目录结构
    create_directory_structure(config)
    
    # 下载数据集（可选，根据实际情况取消注释）
    # for name, url in config.dataset_urls.items():
    #     save_path = os.path.join(config.raw_data_dir, f'{name}.zip')
    #     if download_dataset(url, save_path):
    #         extract_dataset(save_path, config.raw_data_dir)
    
    # 提示用户准备数据
    print("\n请确保在以下目录中按类别组织好您的葡萄叶图像数据:")
    print(f"{config.raw_data_dir}/健康叶片/")
    print(f"{config.raw_data_dir}/霜霉病/")
    print(f"{config.raw_data_dir}/白粉病/")
    print(f"{config.raw_data_dir}/黑腐病/")
    print(f"{config.raw_data_dir}/叶斑病/")
    print("\n如果您没有实际数据，可以使用样本数据集进行测试。")
    
    # 询问用户是否创建样本数据集
    create_sample = input("\n是否创建样本数据集用于测试？(y/n): ").lower()
    if create_sample == 'y':
        create_sample_dataset(config)
        print("\n样本数据集已创建完成，位于: data/dataset/sample/")
    else:
        # 清洗数据
        clean_data(config.raw_data_dir)
        
        # 划分数据集
        split_dataset(config)
        
        # 统计数据集信息
        count_dataset_info(config)
    
    print("\n数据预处理完成！")

if __name__ == '__main__':
    main()