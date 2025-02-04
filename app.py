from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import logging

# 初始化 Flask 应用
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载模型
try:
    pipeline = joblib.load('forest_model.pkl')
    logger.info("模型已成功加载")
except FileNotFoundError:
    logger.error("模型文件未找到，请检查路径 forest_model.pkl 是否正确")
    pipeline = None
scaler = joblib.load('scaler.pkl')
# 自定义评分等级
def assign_grade(score):
    if score > 70:
        return 'High'
    elif score >= 30:
        return 'Medium'
    else:
        return 'Low'

# 数据替换和 PCA 插入函数（若有需要）
def replace_with_pca_and_insert(data, feature_columns, new_feature_name, coefficient1, coefficient2):
    if feature_columns[0] in data and feature_columns[1] in data:
        pca_value = coefficient1 * data[feature_columns[0]] + coefficient2 * data[feature_columns[1]]
        data[new_feature_name] = pca_value
    return data

# 定义预测路由
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 检查模型是否加载
        if pipeline is None:
            return jsonify({'error': '模型未加载，请检查服务器配置'}), 500

        # 获取传入的 JSON 数据
        data = request.json
        logger.info(f"Received data: {data}")

        # 必要的特征列表
        features = ['ALT_AST_pca', 'HBcAb', 'albumin', 'PT', 'GGT', 'AFP', 'INR', 'globulin', 'APTT', 'HBVDNA', 'HBeAg', 'fibrinogen', 'PLT']

        # 自动生成 ALT_AST_pca 特征
        if 'ALT' in data and 'AST' in data:
            data = replace_with_pca_and_insert(data, ['ALT', 'AST'], 'ALT_AST_pca', 0.8715, 0.4904)
        else:
            missing = [key for key in ['ALT', 'AST'] if key not in data]
            return jsonify({'error': f'缺少生成 ALT_AST_pca 的必要输入: {missing}'}), 400

        # 检查缺失的特征
        missing_features = [feature for feature in features if feature not in data]
        if missing_features:
            return jsonify({'error': f'缺少必要的输入数据: {missing_features}'}), 400

        # 数据转换为模型所需的格式
        input_data = np.array([data[feature] for feature in features]).reshape(1, -1)

        # 模型预测
        probability = pipeline.predict_proba(input_data)[:, 1][0]
        score = probability * 100
        grade = assign_grade(score)

        return jsonify({'score': score, 'grade': grade})

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': '服务器错误，请检查输入数据或模型'}), 500


# 主页路由
import os

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # 默认端口为 5000
    app.run(host='0.0.0.0', port=port, debug=True)
