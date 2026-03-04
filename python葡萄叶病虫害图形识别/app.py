from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_preprocess
from PIL import Image
import cv2

# 初始化Flask应用
app = Flask(__name__)
CORS(app)

# 配置上传文件夹和允许的文件类型
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 确保上传文件夹存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 类别标签（根据实际训练的模型调整）
CLASS_NAMES = [
    '健康叶片',
    '霜霉病',
    '白粉病',
    '黑腐病',
    '叶斑病'
]

# 防治建议
PREVENTION_ADVICE = {
    '健康叶片': '叶片状态良好，建议继续保持当前的种植管理措施。定期检查，确保早期发现任何潜在问题。',
    '霜霉病': '1. 及时清除病叶、病果，减少病菌来源；\n2. 加强果园通风透光，降低湿度；\n3. 发病初期可使用波尔多液、代森锰锌等药剂防治；\n4. 选择抗病品种。',
    '白粉病': '1. 冬季清园，减少越冬菌源；\n2. 合理密植，保持通风透光；\n3. 发病初期可使用三唑酮、戊唑醇等药剂防治；\n4. 增施磷钾肥，提高植株抗病能力。',
    '黑腐病': '1. 及时清除病果、病叶，集中烧毁；\n2. 加强栽培管理，避免果实受伤；\n3. 发病初期可使用甲基硫菌灵、多菌灵等药剂防治；\n4. 套袋栽培可有效减少病害发生。',
    '叶斑病': '1. 及时清除病叶，减少病菌传播；\n2. 合理施肥，增强植株抗病能力；\n3. 发病初期可使用百菌清、代森锰锌等药剂防治；\n4. 避免在阴雨天进行田间作业。'
}

# 加载模型（这里使用占位符，实际应用中需要加载预训练模型）
def load_model(model_name='efficientnet'):
    """加载预训练的深度学习模型"""
    try:
        if model_name == 'efficientnet':
            # 加载EfficientNet模型
            model = tf.keras.applications.EfficientNetB0(
                include_top=True,
                weights=None,
                classes=len(CLASS_NAMES)
            )
            # 如果有预训练权重，加载它们
            model_path = 'models/efficientnet_model.h5'
            if os.path.exists(model_path):
                model.load_weights(model_path)
            return model, efficientnet_preprocess
        else:
            # 加载MobileNetV3模型
            model = tf.keras.applications.MobileNetV3Small(
                include_top=True,
                weights=None,
                classes=len(CLASS_NAMES)
            )
            # 如果有预训练权重，加载它们
            model_path = 'models/mobilenet_model.h5'
            if os.path.exists(model_path):
                model.load_weights(model_path)
            return model, mobilenet_preprocess
    except Exception as e:
        print(f"加载模型时出错: {e}")
        # 返回None表示模型加载失败
        return None, None

# 检查文件类型是否允许
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 预处理图像
def preprocess_image(img_path, target_size=(224, 224), preprocess_func=None):
    """预处理上传的图像以适应模型输入"""
    # 读取图像
    img = Image.open(img_path).convert('RGB')
    # 调整大小
    img = img.resize(target_size)
    # 转换为数组
    img_array = image.img_to_array(img)
    # 扩展维度以匹配模型输入形状
    img_array = np.expand_dims(img_array, axis=0)
    # 应用预处理器
    if preprocess_func:
        img_array = preprocess_func(img_array)
    return img_array

# 预测图像类别
def predict_image(model, img_array):
    """使用模型预测图像类别"""
    try:
        # 进行预测
        predictions = model.predict(img_array)
        # 获取预测结果
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        return predicted_class, confidence
    except Exception as e:
        print(f"预测时出错: {e}")
        return None, None

# 图像增强（用于数据预处理阶段）
def augment_image(img_path, output_dir):
    """对图像进行数据增强"""
    # 读取图像
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 获取图像名称和扩展名
    img_name = os.path.basename(img_path)
    name, ext = os.path.splitext(img_name)
    
    # 增强操作1: 水平翻转
    horizontal_flip = cv2.flip(img, 1)
    cv2.imwrite(os.path.join(output_dir, f'{name}_horizontal{ext}'), cv2.cvtColor(horizontal_flip, cv2.COLOR_RGB2BGR))
    
    # 增强操作2: 垂直翻转
    vertical_flip = cv2.flip(img, 0)
    cv2.imwrite(os.path.join(output_dir, f'{name}_vertical{ext}'), cv2.cvtColor(vertical_flip, cv2.COLOR_RGB2BGR))
    
    # 增强操作3: 旋转90度
    rotate_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(os.path.join(output_dir, f'{name}_rotate90{ext}'), cv2.cvtColor(rotate_90, cv2.COLOR_RGB2BGR))
    
    # 增强操作4: 旋转180度
    rotate_180 = cv2.rotate(img, cv2.ROTATE_180)
    cv2.imwrite(os.path.join(output_dir, f'{name}_rotate180{ext}'), cv2.cvtColor(rotate_180, cv2.COLOR_RGB2BGR))
    
    # 增强操作5: 亮度调整（增加）
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, 50)
    v = np.clip(v, 0, 255)
    hsv_bright = cv2.merge((h, s, v))
    bright_img = cv2.cvtColor(hsv_bright, cv2.COLOR_HSV2RGB)
    cv2.imwrite(os.path.join(output_dir, f'{name}_bright{ext}'), cv2.cvtColor(bright_img, cv2.COLOR_RGB2BGR))
    
    # 增强操作6: 亮度调整（降低）
    v = cv2.subtract(v, 100)
    v = np.clip(v, 0, 255)
    hsv_dark = cv2.merge((h, s, v))
    dark_img = cv2.cvtColor(hsv_dark, cv2.COLOR_HSV2RGB)
    cv2.imwrite(os.path.join(output_dir, f'{name}_dark{ext}'), cv2.cvtColor(dark_img, cv2.COLOR_RGB2BGR))

# 主页路由
@app.route('/')
def home():
    return render_template('index.html')

# 上传和预测路由
@app.route('/predict', methods=['POST'])
def predict():
    # 检查请求中是否有文件部分
    if 'file' not in request.files:
        return jsonify({'error': '没有文件部分'})
    
    file = request.files['file']
    
    # 检查是否选择了文件
    if file.filename == '':
        return jsonify({'error': '没有选择文件'})
    
    # 检查文件类型是否允许
    if file and allowed_file(file.filename):
        # 保存文件
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 获取用户选择的模型
        model_name = request.form.get('model', 'efficientnet')
        
        # 加载模型
        model, preprocess_func = load_model(model_name)
        
        if model is None:
            return jsonify({
                'error': '模型加载失败，请确保模型文件存在',
                'file_path': f'/static/uploads/{filename}'
            })
        
        # 预处理图像
        img_array = preprocess_image(filepath, preprocess_func=preprocess_func)
        
        # 预测
        predicted_class_idx, confidence = predict_image(model, img_array)
        
        if predicted_class_idx is None:
            return jsonify({
                'error': '预测失败',
                'file_path': f'/static/uploads/{filename}'
            })
        
        # 获取预测类别名称
        predicted_class = CLASS_NAMES[predicted_class_idx]
        
        # 获取防治建议
        advice = PREVENTION_ADVICE.get(predicted_class, '暂无防治建议')
        
        # 返回预测结果
        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'advice': advice,
            'file_path': f'/static/uploads/{filename}'
        })
    
    return jsonify({'error': '不允许的文件类型'})

# 训练模型的路由（仅用于演示，实际环境中可能需要更安全的方式）
@app.route('/train', methods=['POST'])
def train_model_route():
    # 这里可以添加训练模型的代码
    # 实际应用中，这可能需要更复杂的实现，包括数据加载、预处理、模型训练等
    return jsonify({'message': '模型训练功能正在开发中'})

if __name__ == '__main__':
    # 创建必要的目录
    for dir_path in ['models', 'static', 'templates', 'utils', 'data']:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    # 启动Flask应用
    app.run(debug=True, host='0.0.0.0')