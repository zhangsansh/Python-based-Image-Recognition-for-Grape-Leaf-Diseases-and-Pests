import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0, MobileNetV3Small
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 配置参数
class Config:
    def __init__(self):
        self.data_dir = 'data/dataset'
        self.image_size = (224, 224)
        self.batch_size = 32
        self.epochs = 50
        self.learning_rate = 0.001
        self.validation_split = 0.2
        self.test_split = 0.1
        self.num_classes = 5
        self.model_save_path = '../models'
        self.class_names = ['健康叶片', '霜霉病', '白粉病', '黑腐病', '叶斑病']

# 数据加载和预处理
def load_data(config):
    """加载并预处理数据集"""
    # 创建数据生成器，包含数据增强
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=config.validation_split
    )
    
    # 测试数据生成器（不进行数据增强）
    test_datagen = ImageDataGenerator(
        rescale=1./255
    )
    
    # 创建训练数据集
    train_generator = train_datagen.flow_from_directory(
        config.data_dir,
        target_size=config.image_size,
        batch_size=config.batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # 创建验证数据集
    validation_generator = train_datagen.flow_from_directory(
        config.data_dir,
        target_size=config.image_size,
        batch_size=config.batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # 如果有单独的测试集，可以在这里加载
    # test_generator = test_datagen.flow_from_directory(...)
    
    return train_generator, validation_generator, None

# 创建EfficientNet模型
def create_efficientnet_model(config):
    """创建并配置EfficientNet模型"""
    # 加载预训练的EfficientNetB0模型，不包含顶层分类器
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(*config.image_size, 3)
    )
    
    # 冻结基础模型的层
    for layer in base_model.layers:
        layer.trainable = False
    
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
        optimizer=Adam(learning_rate=config.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# 创建MobileNetV3模型
def create_mobilenet_model(config):
    """创建并配置MobileNetV3模型"""
    # 加载预训练的MobileNetV3Small模型，不包含顶层分类器
    base_model = MobileNetV3Small(
        weights='imagenet',
        include_top=False,
        input_shape=(*config.image_size, 3)
    )
    
    # 冻结基础模型的层
    for layer in base_model.layers:
        layer.trainable = False
    
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
        optimizer=Adam(learning_rate=config.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# 微调模型
def fine_tune_model(model, config):
    """微调模型以提高性能"""
    # 解冻基础模型的部分层
    for layer in model.layers[:-20]:  # 保留最后20层可训练
        layer.trainable = False
    for layer in model.layers[-20:]:
        layer.trainable = True
    
    # 降低学习率并重新编译模型
    model.compile(
        optimizer=Adam(learning_rate=config.learning_rate / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# 训练模型
def train_model(model, train_generator, validation_generator, config, model_name):
    """训练模型并保存最佳权重"""
    # 确保模型保存目录存在
    os.makedirs(config.model_save_path, exist_ok=True)
    
    # 创建回调
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(config.model_save_path, f'{model_name}_best.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6
    )
    
    # 训练模型
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // config.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // config.batch_size,
        epochs=config.epochs,
        callbacks=[early_stopping, model_checkpoint, reduce_lr]
    )
    
    # 保存最终模型
    model.save(os.path.join(config.model_save_path, f'{model_name}_final.h5'))
    
    return model, history

# 评估模型
def evaluate_model(model, validation_generator, config):
    """评估模型性能"""
    # 获取预测结果
    Y_pred = model.predict(validation_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = validation_generator.classes
    
    # 计算分类报告
    report = classification_report(y_true, y_pred, target_names=config.class_names)
    print("分类报告:")
    print(report)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 绘制混淆矩阵热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=config.class_names, yticklabels=config.class_names)
    plt.title('混淆矩阵')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    return report, cm

# 绘制训练历史
def plot_training_history(history, model_name):
    """绘制训练过程中的准确率和损失曲线"""
    # 创建图形
    plt.figure(figsize=(12, 5))
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.title(f'{model_name} 准确率曲线')
    plt.xlabel('轮次')
    plt.ylabel('准确率')
    plt.legend()
    
    # 绘制损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title(f'{model_name} 损失曲线')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    
    # 保存图形
    plt.tight_layout()
    plt.savefig(f'{model_name}_training_history.png')
    plt.show()

# 主函数
def main():
    """主函数"""
    # 创建配置对象
    config = Config()
    
    # 打印配置信息
    print("配置信息:")
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    
    try:
        # 加载数据
        print("\n正在加载数据...")
        train_generator, validation_generator, test_generator = load_data(config)
        print(f"训练样本数: {train_generator.samples}")
        print(f"验证样本数: {validation_generator.samples}")
        
        # 创建并训练EfficientNet模型
        print("\n正在创建EfficientNet模型...")
        efficientnet_model = create_efficientnet_model(config)
        
        print("\n正在训练EfficientNet模型...")
        efficientnet_model, efficientnet_history = train_model(
            efficientnet_model, train_generator, validation_generator, config, 'efficientnet'
        )
        
        print("\n正在微调EfficientNet模型...")
        efficientnet_model = fine_tune_model(efficientnet_model, config)
        efficientnet_model, efficientnet_history_fine = train_model(
            efficientnet_model, train_generator, validation_generator, config, 'efficientnet_fine_tuned'
        )
        
        # 评估EfficientNet模型
        print("\n正在评估EfficientNet模型...")
        evaluate_model(efficientnet_model, validation_generator, config)
        
        # 绘制EfficientNet训练历史
        plot_training_history(efficientnet_history_fine, 'EfficientNet')
        
        # 创建并训练MobileNetV3模型
        print("\n正在创建MobileNetV3模型...")
        mobilenet_model = create_mobilenet_model(config)
        
        print("\n正在训练MobileNetV3模型...")
        mobilenet_model, mobilenet_history = train_model(
            mobilenet_model, train_generator, validation_generator, config, 'mobilenet'
        )
        
        print("\n正在微调MobileNetV3模型...")
        mobilenet_model = fine_tune_model(mobilenet_model, config)
        mobilenet_model, mobilenet_history_fine = train_model(
            mobilenet_model, train_generator, validation_generator, config, 'mobilenet_fine_tuned'
        )
        
        # 评估MobileNetV3模型
        print("\n正在评估MobileNetV3模型...")
        evaluate_model(mobilenet_model, validation_generator, config)
        
        # 绘制MobileNetV3训练历史
        plot_training_history(mobilenet_history_fine, 'MobileNetV3')
        
        print("\n模型训练和评估完成！")
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()