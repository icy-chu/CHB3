from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import logging

# 初始化 Flask 应用
app = Flask(__name__)#__name__代表目前执行的模组
CORS(app, resources={r"/predict": {"origins": "*"}})

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载模型
pipeline = joblib.load('random_forest_model2.pkl')
logger.info("模型已成功加载")

# 自定义评分等级
def assign_grade(score):
    if score > 40:
        return 'High'
    elif score > 20:
        return 'Medium'
    else:
        return 'Low'

# 定义预测路由
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 获取传入的 JSON 数据
        data = request.json
        logger.info(f"Received data: {data}")

        features = ['AST', 'HBcAb' ,'ALT', 'albumin', 'GGT', 'HBVDNA', 'INR']
        if not all(feature in data for feature in features):
            return jsonify({'error': '缺少必要的输入数据'}), 400

        # 数据转换
        input_data = np.array([data[feature] for feature in features]).reshape(1, -1)

        # 模型预测
        probability = pipeline.predict_proba(input_data)[:, 1][0]
        score = probability * 100
        grade = assign_grade(score)

        return jsonify({'score': score, 'grade': grade})

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': '服务器错误'}), 500

# 主页路由
import os

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # 默认端口为 5000
    app.run(host='0.0.0.0', port=port, debug=True)
