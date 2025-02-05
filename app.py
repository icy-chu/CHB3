from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import logging

# 初始化 Flask 应用
app = Flask(__name__)
CORS(app)

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

# 处理输入数据，确保包含 ALT_AST_pca 特征，并只传递必须的特征
def preprocess_input_data(data):
    # 定义所有必需的特征列（13个特征）
    features = [
        'albumin', 'globulin', 'GGT', 'PLT', 'AFP', 'PT', 'INR', 'APTT',
        'fibrinogen', 'HBcAb', 'HBeAg', 'HBVDNA', 'ALT_AST_pca'
    ]

    # 生成 ALT_AST_pca 特征
    data = replace_with_pca_and_insert(data, ['ALT', 'AST'], 'ALT_AST_pca', 0.8715, 0.4904)
    logger.info(f"Data after PCA feature insertion: {data}")

    # 检查所有必要特征是否存在
    missing_features = [feature for feature in features if feature not in data]
    if missing_features:
        return None, missing_features  # 返回缺少的特征

    # 只保留必需的特征列（13个）
    input_data = np.array([data[feature] for feature in features]).reshape(1, -1)
    return input_data, None

# 定义预测路由
@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info("Predict route accessed")
        data = request.json
        logger.info(f"Received data: {data}")

        # 处理输入数据
        input_data, missing_features = preprocess_input_data(data)
        if missing_features:
            return jsonify({'error': f'缺少特征：{missing_features}'}), 400

        # 进行标准化
        input_data_scaled = scaler.transform(input_data)
        logger.info(f"Normalized input data: {input_data_scaled}")

        # 预测
        probability = pipeline.predict_proba(input_data_scaled)[:, 1][0]
        logger.info(f"Predicted probability: {probability}")

        score = probability * 100
        grade = assign_grade(score)

        return jsonify({'score': score, 'grade': grade})
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({'error': '服务器错误，请检查输入数据或模型'}), 500

# 主页路由
import os

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))  # 默认端口为 5001
    app.run(host='127.0.0.1', port=5001, debug=True)

