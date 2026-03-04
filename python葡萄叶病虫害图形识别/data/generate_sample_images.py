import os
import numpy as np
import cv2
import random
from PIL import Image, ImageDraw

# 配置参数
class Config:
    def __init__(self):
        self.output_dir = '../static/images/samples'
        self.image_size = (224, 224)
        self.num_samples_per_class = 2
        self.class_names = ['健康叶片', '霜霉病', '白粉病', '黑腐病', '叶斑病']

# 创建目录
def create_directories(config):
    """创建输出目录"""
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 为每个类别创建子目录
    for class_name in config.class_names:
        class_dir = os.path.join(config.output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

# 生成健康叶片图像
def generate_healthy_leaf(size):
    """生成健康的葡萄叶图像"""
    # 创建白色背景
    img = np.ones((*size, 3), dtype=np.uint8) * 255
    
    # 创建PIL图像用于绘制
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    
    # 随机生成叶片形状
    width, height = size
    center_x, center_y = width // 2, height // 2
    leaf_length = int(min(size) * 0.8)
    leaf_width = int(leaf_length * 0.6)
    
    # 绘制叶片轮廓
    points = []
    # 上半部分轮廓
    for i in range(0, 181, 5):
        angle = np.radians(i)
        # 使用正弦函数生成波浪状轮廓
        r = leaf_width * 0.5 * (1 + 0.2 * np.sin(5 * angle))
        x = center_x + r * np.cos(angle)
        y = center_y - leaf_length * 0.5 * (1 - angle / np.pi)
        points.append((x, y))
    # 下半部分轮廓
    for i in range(180, 361, 5):
        angle = np.radians(i)
        # 使用正弦函数生成波浪状轮廓
        r = leaf_width * 0.5 * (1 + 0.2 * np.sin(5 * angle))
        x = center_x + r * np.cos(angle)
        y = center_y + leaf_length * 0.5 * (1 - (np.pi * 2 - angle) / np.pi)
        points.append((x, y))
    
    # 绘制叶片主体
    draw.polygon(points, fill=(34, 139, 34))  # 深绿色
    
    # 添加叶脉
    # 主叶脉
    draw.line([(center_x, center_y - leaf_length * 0.4), 
               (center_x, center_y + leaf_length * 0.4)], 
              fill=(0, 100, 0), width=3)
    
    # 侧叶脉
    for i in range(-5, 6):
        if i == 0:  # 跳过主叶脉
            continue
        # 左侧叶脉
        start_x = center_x + i * 5
        start_y = center_y - leaf_length * 0.4 + abs(i) * 3
        end_x = center_x - leaf_width * 0.4 * (i / abs(i))
        end_y = center_y - leaf_length * 0.2 + abs(i) * 5
        draw.line([(start_x, start_y), (end_x, end_y)], 
                  fill=(0, 100, 0), width=1)
        
        # 右侧叶脉
        start_x = center_x + i * 5
        start_y = center_y - leaf_length * 0.4 + abs(i) * 3
        end_x = center_x + leaf_width * 0.4 * (i / abs(i))
        end_y = center_y - leaf_length * 0.2 + abs(i) * 5
        draw.line([(start_x, start_y), (end_x, end_y)], 
                  fill=(0, 100, 0), width=1)
    
    # 转换回numpy数组
    img = np.array(pil_img)
    
    return img

# 生成霜霉病叶片图像
def generate_downy_mildew_leaf(size):
    """生成感染霜霉病的葡萄叶图像"""
    # 先生成健康叶片
    img = generate_healthy_leaf(size)
    
    # 在叶片上添加霜霉病症状
    height, width = size
    
    # 生成黄色斑点
    for _ in range(20):
        x = random.randint(width // 4, width * 3 // 4)
        y = random.randint(height // 4, height * 3 // 4)
        radius = random.randint(5, 15)
        cv2.circle(img, (x, y), radius, (255, 255, 0), -1)
    
    # 生成白色霜霉层（主要在叶片背面，但这里简化处理）
    for _ in range(15):
        x = random.randint(width // 4, width * 3 // 4)
        y = random.randint(height // 4, height * 3 // 4)
        radius = random.randint(8, 20)
        # 使用白色半透明覆盖
        overlay = img.copy()
        cv2.circle(overlay, (x, y), radius, (255, 255, 255), -1)
        # 混合图像
        img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
    
    return img

# 生成白粉病叶片图像
def generate_powdery_mildew_leaf(size):
    """生成感染白粉病的葡萄叶图像"""
    # 先生成健康叶片
    img = generate_healthy_leaf(size)
    
    # 在叶片上添加白粉病症状
    height, width = size
    
    # 生成白色粉末状区域
    for _ in range(10):
        x1 = random.randint(width // 5, width * 4 // 5)
        y1 = random.randint(height // 5, height * 4 // 5)
        x2 = min(x1 + random.randint(20, 50), width)
        y2 = min(y1 + random.randint(20, 50), height)
        # 使用白色半透明覆盖
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), -1)
        # 混合图像
        img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
    
    # 添加一些更小的白色斑点
    for _ in range(30):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        radius = random.randint(1, 3)
        cv2.circle(img, (x, y), radius, (255, 255, 255), -1)
    
    return img

# 生成黑腐病叶片图像
def generate_black_rot_leaf(size):
    """生成感染黑腐病的葡萄叶图像"""
    # 先生成健康叶片
    img = generate_healthy_leaf(size)
    
    # 在叶片上添加黑腐病症状
    height, width = size
    
    # 生成褐色病斑
    for _ in range(8):
        x = random.randint(width // 4, width * 3 // 4)
        y = random.randint(height // 4, height * 3 // 4)
        radius = random.randint(10, 30)
        # 绘制同心圆环来模拟病斑的发展过程
        for i in range(3):
            r = radius - i * 5
            if r <= 0:
                break
            # 外层是褐色，内层逐渐变黑
            color = (139 - i * 40, 69 - i * 20, 19 - i * 10)  # 从褐色到深褐色
            cv2.circle(img, (x, y), r, color, -1)
    
    # 添加黑色坏死区域
    for _ in range(5):
        x = random.randint(width // 5, width * 4 // 5)
        y = random.randint(height // 5, height * 4 // 5)
        radius = random.randint(5, 15)
        cv2.circle(img, (x, y), radius, (0, 0, 0), -1)
    
    return img

# 生成叶斑病叶片图像
def generate_leaf_spot_leaf(size):
    """生成感染叶斑病的葡萄叶图像"""
    # 先生成健康叶片
    img = generate_healthy_leaf(size)
    
    # 在叶片上添加叶斑病症状
    height, width = size
    
    # 生成不同大小的褐色斑点
    for _ in range(25):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        radius = random.randint(3, 8)
        # 斑点中心颜色较深
        cv2.circle(img, (x, y), radius, (139, 69, 19), -1)  # 褐色
        # 斑点边缘颜色较浅
        if radius > 4:
            cv2.circle(img, (x, y), radius + 1, (205, 133, 63), 1)  # 浅褐色边框
    
    # 添加一些黄色晕圈
    for _ in range(10):
        x = random.randint(width // 4, width * 3 // 4)
        y = random.randint(height // 4, height * 3 // 4)
        radius = random.randint(10, 20)
        # 使用黄色半透明覆盖
        overlay = img.copy()
        cv2.circle(overlay, (x, y), radius, (255, 255, 0), -1)
        # 混合图像
        img = cv2.addWeighted(overlay, 0.2, img, 0.8, 0)
    
    return img

# 生成所有类别的示例图像
def generate_all_samples(config):
    """生成所有类别的示例图像"""
    # 创建目录
    create_directories(config)
    
    # 生成每个类别的示例图像
    for class_name in config.class_names:
        print(f"正在生成 {class_name} 的示例图像...")
        
        # 获取对应的生成函数
        if class_name == '健康叶片':
            generate_func = generate_healthy_leaf
        elif class_name == '霜霉病':
            generate_func = generate_downy_mildew_leaf
        elif class_name == '白粉病':
            generate_func = generate_powdery_mildew_leaf
        elif class_name == '黑腐病':
            generate_func = generate_black_rot_leaf
        elif class_name == '叶斑病':
            generate_func = generate_leaf_spot_leaf
        else:
            print(f"警告: 未知类别 {class_name}")
            continue
        
        # 生成指定数量的示例图像
        for i in range(config.num_samples_per_class):
            # 生成图像
            img = generate_func(config.image_size)
            
            # 添加一些随机噪声以增加真实感
            noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)
            
            # 随机调整亮度和对比度
            alpha = random.uniform(0.9, 1.1)  # 对比度调整因子
            beta = random.randint(-10, 10)     # 亮度调整因子
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            
            # 保存图像
            output_path = os.path.join(config.output_dir, class_name, f'sample_{i+1}.jpg')
            cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"  已保存: {output_path}")
    
    print("所有示例图像生成完成！")

# 主函数
if __name__ == '__main__':
    # 创建配置对象
    config = Config()
    
    # 生成所有示例图像
    generate_all_samples(config)