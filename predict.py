"""
Module dự đoán mức tiêu thụ điện cho nhà thông minh từ dữ liệu cảm biến đầu vào.
Tải mô hình đã được huấn luyện tốt nhất từ outputs/best_energy_model.pkl.
"""

import os
import sys

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "best_energy_model.pkl")


def load_model_bundle():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Chưa tìm thấy mô hình tại {MODEL_PATH}. Vui lòng chạy 'python train.py' trước!")
    return joblib.load(MODEL_PATH)


def predict_sample(sample_dict):
    """
    Dự đoán mức tiêu thụ điện (Wh) từ một từ điển (dict) chứa các thông số cảm biến.
    """
    bundle = load_model_bundle()
    model = bundle['model']
    feature_cols = bundle['feature_cols']

    df_in = pd.DataFrame([sample_dict])

    # Tự động tính các đặc trưng phái sinh nếu chưa có sẵn
    if 'hour' not in df_in.columns and 'date' in df_in.columns:
        df_in['date'] = pd.to_datetime(df_in['date'])
        h = df_in['date'].dt.hour + df_in['date'].dt.minute / 60.0
        df_in['hour'] = df_in['date'].dt.hour
        df_in['dayofweek'] = df_in['date'].dt.day_of_week
        df_in['is_weekend'] = (df_in['dayofweek'] >= 5).astype(int)
        df_in['is_night'] = ((df_in['hour'] >= 22) | (df_in['hour'] <= 6)).astype(int)
        df_in['sin_hour'] = np.sin(2 * np.pi * h / 24.0)
        df_in['cos_hour'] = np.cos(2 * np.pi * h / 24.0)
        df_in['sin_dayofweek'] = np.sin(2 * np.pi * df_in['dayofweek'] / 7.0)
        df_in['cos_dayofweek'] = np.cos(2 * np.pi * df_in['dayofweek'] / 7.0)

    # Tính trung bình nhiệt độ & độ ẩm nếu chưa có
    temp_cols = [f'T{i}' for i in range(1, 10) if f'T{i}' in df_in.columns]
    rh_cols = [f'RH_{i}' for i in range(1, 10) if f'RH_{i}' in df_in.columns]

    if 'T_indoor_mean' not in df_in.columns and len(temp_cols) > 0:
        df_in['T_indoor_mean'] = df_in[temp_cols].mean(axis=1)
        df_in['T_indoor_std'] = df_in[temp_cols].std(axis=1).fillna(0)

    if 'RH_indoor_mean' not in df_in.columns and len(rh_cols) > 0:
        df_in['RH_indoor_mean'] = df_in[rh_cols].mean(axis=1)
        df_in['RH_indoor_std'] = df_in[rh_cols].std(axis=1).fillna(0)

    if 'T_diff_out' not in df_in.columns and 'T_out' in df_in.columns:
        df_in['T_diff_out'] = df_in['T_indoor_mean'] - df_in['T_out']

    if 'RH_diff_out' not in df_in.columns and 'RH_out' in df_in.columns:
        df_in['RH_diff_out'] = df_in['RH_indoor_mean'] - df_in['RH_out']

    # Bổ sung các cột còn thiếu bằng giá trị mặc định 0 nếu cần
    for col in feature_cols:
        if col not in df_in.columns:
            df_in[col] = 0.0

    X = df_in[feature_cols]
    pred = model.predict(X)[0]
    return float(max(0.0, pred))


if __name__ == "__main__":
    print("=" * 60)
    print(" DEMO DỰ ĐOÁN MỨC TIÊU THỤ ĐIỆN TỨC THỜI (WH)")
    print("=" * 60)

    # Mẫu thử nghiệm giả định (thời điểm 18:30 chiều tan tầm về nhà)
    sample_input = {
        'date': '2016-05-20 18:30:00',
        'lights': 30,
        # Nhiệt độ các phòng (°C)
        'T1': 21.5, 'T2': 20.8, 'T3': 22.0, 'T4': 20.5, 'T5': 19.8,
        'T6': 18.5, 'T7': 20.2, 'T8': 21.0, 'T9': 20.0,
        # Độ ẩm các phòng (%)
        'RH_1': 45.2, 'RH_2': 42.1, 'RH_3': 43.5, 'RH_4': 41.0, 'RH_5': 50.2,
        'RH_6': 60.5, 'RH_7': 44.0, 'RH_8': 46.2, 'RH_9': 43.8,
        # Thời tiết ngoài trời
        'T_out': 16.5, 'Press_mm_hg': 755.0, 'RH_out': 70.0,
        'Windspeed': 4.5, 'Visibility': 40.0, 'Tdewpoint': 11.0,
        'T_out_lag1': 16.3, 'RH_out_lag1': 71.0, 'T_indoor_lag1': 20.7
    }

    try:
        predicted_energy = predict_sample(sample_input)
        print(f"\n[+] Kết quả dự đoán lượng điện tiêu thụ: {predicted_energy:.2f} Wh")
        print(f"    (Tương đương: {predicted_energy / 1000.0:.4f} kWh trong chu kỳ 10 phút)")
    except Exception as e:
        print(f"Lỗi: {e}")
