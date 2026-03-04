import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import time
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("GrapeLeafDiseaseDetector")

# 图像预处理函数
def preprocess_image(image_path, target_size=(224, 224), normalize=True):
    """预处理图像以适应模型输入"""
    try:
        # 使用PIL读取图像
        img = Image.open(image_path).convert('RGB')
        
        # 调整大小
        img = img.resize(target_size)
        
        # 转换为numpy数组
        img_array = np.array(img)
        
        # 归一化
        if normalize:
            img_array = img_array / 255.0
        
        # 扩展维度以匹配模型输入形状
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        logger.error(f"预处理图像时出错: {e}")
        return None

# 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """绘制混淆矩阵热图"""
    try:
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 绘制热图
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('混淆矩阵')
        plt.xlabel('预测类别')
        plt.ylabel('真实类别')
        
        # 保存图像
        if save_path:
            plt.savefig(save_path)
            logger.info(f"混淆矩阵已保存至: {save_path}")
        
        plt.close()
        return cm
    except Exception as e:
        logger.error(f"绘制混淆矩阵时出错: {e}")
        return None

# 评估模型性能
def evaluate_model_performance(model, test_generator, class_names, save_dir=None):
    """评估模型性能并生成报告"""
    try:
        # 记录开始时间
        start_time = time.time()
        
        # 获取预测结果
        y_pred = model.predict(test_generator)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = test_generator.classes
        
        # 计算评估指标
        report = classification_report(y_true, y_pred_classes, target_names=class_names)
        
        # 记录结束时间
        end_time = time.time()
        inference_time = end_time - start_time
        
        # 计算平均推理时间
        avg_inference_time = inference_time / len(y_pred)
        
        # 打印评估结果
        logger.info(f"模型评估完成")
        logger.info(f"总推理时间: {inference_time:.2f}秒")
        logger.info(f"平均每张图像推理时间: {avg_inference_time:.4f}秒")
        logger.info(f"分类报告:\n{report}")
        
        # 绘制混淆矩阵
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            cm_save_path = os.path.join(save_dir, 'confusion_matrix.png')
            plot_confusion_matrix(y_true, y_pred_classes, class_names, cm_save_path)
        
        return {
            'report': report,
            'confusion_matrix': confusion_matrix(y_true, y_pred_classes),
            'inference_time': inference_time,
            'avg_inference_time': avg_inference_time
        }
    except Exception as e:
        logger.error(f"评估模型性能时出错: {e}")
        return None

# 图像增强函数
def augment_single_image(image, augmentation_factor=5):
    """对单个图像进行数据增强"""
    try:
        augmented_images = [image]  # 包含原始图像
        
        # 获取图像尺寸
        height, width = image.shape[:2]
        
        for i in range(augmentation_factor):
            augmented_img = image.copy()
            
            # 随机旋转（-15到15度）
            angle = np.random.randint(-15, 15)
            M = cv2.getRotationMatrix2D((width//2, height//2), angle, 1)
            augmented_img = cv2.warpAffine(augmented_img, M, (width, height))
            
            # 随机水平翻转
            if np.random.random() > 0.5:
                augmented_img = cv2.flip(augmented_img, 1)
            
            # 随机垂直翻转
            if np.random.random() > 0.5:
                augmented_img = cv2.flip(augmented_img, 0)
            
            # 随机亮度调整
            brightness_factor = np.random.uniform(0.8, 1.2)
            hsv = cv2.cvtColor(augmented_img, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            v = cv2.add(v, int(50 * (brightness_factor - 1)))
            v = np.clip(v, 0, 255)
            hsv_bright = cv2.merge((h, s, v))
            augmented_img = cv2.cvtColor(hsv_bright, cv2.COLOR_HSV2RGB)
            
            # 添加到结果列表
            augmented_images.append(augmented_img)
        
        return augmented_images
    except Exception as e:
        logger.error(f"增强图像时出错: {e}")
        return [image]  # 返回原始图像

# 加载和保存模型
def load_model(model_path):
    """加载模型"""
    try:
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            logger.info(f"成功加载模型: {model_path}")
            return model
        else:
            logger.error(f"模型文件不存在: {model_path}")
            return None
    except Exception as e:
        logger.error(f"加载模型时出错: {e}")
        return None

# 保存模型
def save_model(model, model_path):
    """保存模型"""
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # 保存模型
        model.save(model_path)
        logger.info(f"模型已保存至: {model_path}")
        return True
    except Exception as e:
        logger.error(f"保存模型时出错: {e}")
        return False

# 计算准确率、精确率、召回率和F1分数
def calculate_metrics(y_true, y_pred):
    """计算分类模型的评估指标"""
    try:
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 计算总体准确率
        accuracy = np.trace(cm) / np.sum(cm)
        
        # 计算每个类别的精确率、召回率和F1分数
        precision = []
        recall = []
        f1_score = []
        
        for i in range(len(cm)):
            # 精确率 = TP / (TP + FP)
            if np.sum(cm[:, i]) > 0:
                p = cm[i, i] / np.sum(cm[:, i])
            else:
                p = 0
            
            # 召回率 = TP / (TP + FN)
            if np.sum(cm[i, :]) > 0:
                r = cm[i, i] / np.sum(cm[i, :])
            else:
                r = 0
            
            # F1分数 = 2 * (精确率 * 召回率) / (精确率 + 召回率)
            if p + r > 0:
                f1 = 2 * (p * r) / (p + r)
            else:
                f1 = 0
            
            precision.append(p)
            recall.append(r)
            f1_score.append(f1)
        
        # 计算宏平均
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1_score)
        
        # 计算加权平均
        support = np.sum(cm, axis=1)
        weighted_precision = np.average(precision, weights=support)
        weighted_recall = np.average(recall, weights=support)
        weighted_f1 = np.average(f1_score, weights=support)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'confusion_matrix': cm
        }
    except Exception as e:
        logger.error(f"计算评估指标时出错: {e}")
        return None

# 调整图像对比度和亮度
def adjust_brightness_contrast(image, brightness=0, contrast=0):
    """调整图像的亮度和对比度"""
    try:
        # 转换为浮点型以避免溢出
        img = image.astype(float)
        
        # 调整对比度
        if contrast != 0:
            factor = (259 * (contrast + 255)) / (255 * (259 - contrast))
            img = factor * (img - 128) + 128
        
        # 调整亮度
        if brightness != 0:
            img = img + brightness
        
        # 裁剪到有效范围
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        return img
    except Exception as e:
        logger.error(f"调整图像亮度和对比度时出错: {e}")
        return image

# 去除图像噪声
def remove_noise(image, method='gaussian', kernel_size=(5, 5)):
    """去除图像中的噪声"""
    try:
        if method == 'gaussian':
            return cv2.GaussianBlur(image, kernel_size, 0)
        elif method == 'median':
            return cv2.medianBlur(image, kernel_size[0])
        elif method == 'bilateral':
            return cv2.bilateralFilter(image, kernel_size[0], 75, 75)
        else:
            logger.warning(f"未知的去噪方法: {method}，使用原始图像")
            return image
    except Exception as e:
        logger.error(f"去除图像噪声时出错: {e}")
        return image

# 边缘检测
def detect_edges(image, low_threshold=100, high_threshold=200):
    """使用Canny算法检测图像边缘"""
    try:
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny边缘检测
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        
        return edges
    except Exception as e:
        logger.error(f"检测图像边缘时出错: {e}")
        return image

# 计算图像的颜色直方图
def calculate_color_histogram(image, bins=256):
    """计算图像的颜色直方图"""
    try:
        hist = []
        
        # 分别计算RGB三个通道的直方图
        for i in range(3):
            channel_hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
            cv2.normalize(channel_hist, channel_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            hist.extend(channel_hist.flatten())
        
        return np.array(hist)
    except Exception as e:
        logger.error(f"计算图像颜色直方图时出错: {e}")
        return None

# 比较两个图像的相似度
def compare_images(image1, image2, metric='histogram'):
    """比较两个图像的相似度"""
    try:
        if metric == 'histogram':
            # 计算直方图
            hist1 = calculate_color_histogram(image1)
            hist2 = calculate_color_histogram(image2)
            
            if hist1 is None or hist2 is None:
                return 0
            
            # 使用相关性计算相似度
            similarity = cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_CORREL)
            
            # 相关性范围是[-1, 1]，转换为[0, 1]
            return (similarity + 1) / 2
        else:
            logger.warning(f"未知的相似度度量方法: {metric}")
            return 0
    except Exception as e:
        logger.error(f"比较图像相似度时出错: {e}")
        return 0

# 保存图像
def save_image(image, save_path):
    """保存图像"""
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存图像
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        logger.info(f"图像已保存至: {save_path}")
        return True
    except Exception as e:
        logger.error(f"保存图像时出错: {e}")
        return False

# 读取图像
def read_image(image_path):
    """读取图像"""
    try:
        # 使用cv2读取图像
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"无法读取图像: {image_path}")
            return None
        
        # 转换为RGB格式
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img
    except Exception as e:
        logger.error(f"读取图像时出错: {e}")
        return None

# 批量处理图像
def batch_process_images(input_dir, output_dir, process_func, **kwargs):
    """批量处理图像"""
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取输入目录中的所有图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        image_files = []
        
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))
        
        # 处理每个图像
        for i, image_path in enumerate(image_files):
            # 读取图像
            img = read_image(image_path)
            if img is None:
                continue
            
            # 处理图像
            processed_img = process_func(img, **kwargs)
            
            # 获取相对路径以保持目录结构
            rel_path = os.path.relpath(image_path, input_dir)
            save_path = os.path.join(output_dir, rel_path)
            
            # 确保保存目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 保存处理后的图像
            save_image(processed_img, save_path)
            
            # 打印进度
            if (i + 1) % 100 == 0:
                logger.info(f"已处理 {i + 1}/{len(image_files)} 张图像")
        
        logger.info(f"批量处理完成，共处理 {len(image_files)} 张图像")
        return True
    except Exception as e:
        logger.error(f"批量处理图像时出错: {e}")
        return False

# 主函数（用于测试）
if __name__ == '__main__':
    # 测试工具函数
    print("工具函数模块加载成功")
    # 这里可以添加测试代码
    pass