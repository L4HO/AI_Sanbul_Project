import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib
from flask import Flask, render_template
from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

np.random.seed(42)
app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
Bootstrap5(app)

model = keras.models.load_model('fires_model.keras')
full_pipeline = joblib.load('fires_pipeline.pkl')

class LabForm(FlaskForm):
    longitude = StringField('longitude', validators=[DataRequired()])
    latitude = StringField('latitude', validators=[DataRequired()])
    month = StringField('month (예: 03-Mar)', validators=[DataRequired()])
    day = StringField('day (예: 05-fri)', validators=[DataRequired()])
    avg_temp = StringField('avg_temp', validators=[DataRequired()])
    max_temp = StringField('max_temp', validators=[DataRequired()])
    max_wind_speed = StringField('max_wind_speed', validators=[DataRequired()])
    avg_wind = StringField('avg_wind', validators=[DataRequired()])
    submit = SubmitField('산불 예측하기')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    form = LabForm()
    if form.validate_on_submit():
        input_data = pd.DataFrame([{
            'longitude': float(form.longitude.data),
            'latitude': float(form.latitude.data),
            'month': form.month.data,
            'day': form.day.data,
            'avg_temp': float(form.avg_temp.data),
            'max_temp': float(form.max_temp.data),
            'max_wind_speed': float(form.max_wind_speed.data),
            'avg_wind': float(form.avg_wind.data)
        }])
        
        prepared = full_pipeline.transform(input_data)
        pred_log = model.predict(prepared, verbose=0)[0][0]
        
        # ←←← 핵심 수정 부분 (이걸 그대로 복사해서 넣으세요)
        print(f"🔍 모델 예측 log 값: {pred_log:.4f}")   # 확인용
        
        # log 값을 현실적인 범위로 강제 제한
        pred_log = np.clip(pred_log, -2.0, 6.0)      # 대략 0.1 ~ 2980 m²
        pred_area = max(0.0, np.exp(pred_log) - 1)
        pred_area = min(pred_area, 5000.0)           # 최대 5000m² 제한
        
        print(f"→ 최종 예측 면적: {pred_area:.2f} m²")   # 확인용
        # ←←← 여기까지
        return render_template('result.html', area=round(pred_area, 2))
    return render_template('prediction.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)
