import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, MobileNetV3Small
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# 配置参数
class Config:
    def __init__(self):
        self.model_save_path = '../models'
        self.image_size = (224, 224)
        self.num_classes = 5

# 创建占位模型
def create_placeholder_model(model_type, config):
    """创建占位模型并保存"""
    print(f"正在创建{model_type}占位模型...")
    
    if model_type == 'efficientnet':
        # 创建EfficientNet占位模型
        base_model = EfficientNetB0(
            include_top=False,
            weights=None,  # 不使用预训练权重
            input_shape=(*config.image_size, 3)
        )
        
        # 添加自定义分类头
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(config.num_classes, activation='softmax')(x)
        
        # 构建完整模型
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # 编译模型
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 保存模型
        model_path = os.path.join(config.model_save_path, 'efficientnet_model.h5')
        
    elif model_type == 'mobilenet':
        # 创建MobileNetV3占位模型
        base_model = MobileNetV3Small(
            include_top=False,
            weights=None,  # 不使用预训练权重
            input_shape=(*config.image_size, 3)
        )
        
        # 添加自定义分类头
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(config.num_classes, activation='softmax')(x)
        
        # 构建完整模型
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # 编译模型
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 保存模型
        model_path = os.path.join(config.model_save_path, 'mobilenet_model.h5')
    
    else:
        print(f"未知的模型类型: {model_type}")
        return False
    
    # 保存模型
    model.save(model_path)
    print(f"{model_type}占位模型已保存至: {model_path}")
    
    # 创建一个简单的模型权重，使模型能够产生一些合理的预测
    # 这只是为了演示目的，实际应用中应该使用训练好的权重
    create_demo_weights(model, config.num_classes)
    
    # 再次保存带有演示权重的模型
    model.save(model_path)
    
    return True

# 创建演示权重
def create_demo_weights(model, num_classes):
    """创建简单的演示权重"""
    # 为了简单起见，我们只是初始化权重，不进行实际训练
    # 在实际应用中，这里应该加载训练好的权重
    print("正在创建演示权重...")
    
    # 遍历模型的所有层
    for layer in model.layers:
        # 获取层的权重
        weights = layer.get_weights()
        
        # 如果层有权重
        if len(weights) > 0:
            # 创建新的随机权重
            new_weights = []
            for w in weights:
                # 保持相同的形状，但用随机值填充
                # 为了使预测更有趣，我们使用一个固定的随机种子
                np.random.seed(42)
                new_w = np.random.normal(0, 0.01, w.shape)
                new_weights.append(new_w)
            
            # 设置新的权重
            layer.set_weights(new_weights)

# 主函数
if __name__ == '__main__':
    # 创建配置对象
    config = Config()
    
    # 确保模型保存目录存在
    os.makedirs(config.model_save_path, exist_ok=True)
    
    # 创建EfficientNet占位模型
    create_placeholder_model('efficientnet', config)
    
    # 创建MobileNetV3占位模型
    create_placeholder_model('mobilenet', config)
    
    print("所有占位模型创建完成！")
    print("注意：这些只是占位模型，不包含实际训练的权重。要获得准确的预测结果，请使用train_model.py脚本训练模型。")