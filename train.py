"""
Hệ thống dự đoán mức tiêu thụ điện trong nhà thông minh (Smart Home Energy Prediction)
Tập dữ liệu: UCI Appliances Energy Prediction
Tác giả / Nhóm thực hiện: Sinh viên thực hiện
"""

import os
import sys

# Đảm bảo in tiếng Việt chuẩn trên Windows console
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Kiểm tra thư viện LightGBM (tùy chọn mở rộng nếu có)
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# Đường dẫn & cấu hình
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET_CSV = os.path.join(DATA_DIR, "energydata_complete.csv")
UCI_DATA_URL = "https://raw.githubusercontent.com/LuisM78/Appliances-energy-prediction-data/master/energydata_complete.csv"


def download_dataset():
    """Tự động tải tập dữ liệu UCI Appliances Energy Prediction nếu chưa có."""
    if not os.path.exists(DATASET_CSV):
        print(f"[*] Đang tải tập dữ liệu từ: {UCI_DATA_URL} ...")
        response = requests.get(UCI_DATA_URL, timeout=30)
        response.raise_for_status()
        with open(DATASET_CSV, "wb") as f:
            f.write(response.content)
        print(f"[+] Tải thành công! Đã lưu tại: {DATASET_CSV}")
    else:
        print(f"[i] Dữ liệu đã có sẵn tại: {DATASET_CSV}")


def load_and_preprocess_data():
    """Đọc dữ liệu, làm sạch và xử lý các biến nhiễu theo khuyến nghị của UCI."""
    print("\n--- 1. ĐỌC VÀ LÀM SẠCH DỮ LIỆU ---")
    df = pd.read_csv(DATASET_CSV)
    print(f"Kích thước dữ liệu gốc: {df.shape[0]} dòng, {df.shape[1]} cột")

    # Chuyển đổi cột date sang datetime
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Loại bỏ 2 biến ngẫu nhiên nhiễu (rv1, rv2) do UCI cố ý đưa vào để thử thách mô hình
    cols_to_drop = [c for c in ['rv1', 'rv2'] if c in df.columns]
    if cols_to_drop:
        print(f"[+] Loại bỏ biến nhiễu kiểm thử của UCI: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    # Kiểm tra giá trị rỗng (Missing Values)
    null_counts = df.isnull().sum().sum()
    print(f"Tổng số giá trị bị thiếu (Null/NaN): {null_counts}")
    if null_counts > 0:
        df = df.dropna()

    return df


def engineer_features(df):
    """
    Kỹ thuật đặc trưng (Feature Engineering):
    - Chu kỳ thời gian (sin/cos hour, dayofweek, is_weekend, is_night)
    - Thống kê tổng hợp nhiệt độ & độ ẩm trong nhà
    - Độ chênh lệch môi trường trong nhà và ngoài trời
    - Độ trễ môi trường (Lag features)
    """
    print("\n--- 2. KỸ THUẬT ĐẶC TRƯNG (FEATURE ENGINEERING) ---")
    df = df.copy()

    # 1. Đặc trưng thời gian & chu kỳ
    hour = df['date'].dt.hour + df['date'].dt.minute / 60.0
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.day_of_week
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)

    # Biến đổi tuần hoàn sin/cos cho giờ (23h và 0h liền kề nhau)
    df['sin_hour'] = np.sin(2 * np.pi * hour / 24.0)
    df['cos_hour'] = np.cos(2 * np.pi * hour / 24.0)
    df['sin_dayofweek'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
    df['cos_dayofweek'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)

    # 2. Đặc trưng tổng hợp trong nhà (Nhiệt độ T1-T9, Độ ẩm RH_1-RH_9)
    temp_cols = [f'T{i}' for i in range(1, 10)]
    rh_cols = [f'RH_{i}' for i in range(1, 10)]

    df['T_indoor_mean'] = df[temp_cols].mean(axis=1)
    df['T_indoor_std'] = df[temp_cols].std(axis=1)
    df['RH_indoor_mean'] = df[rh_cols].mean(axis=1)
    df['RH_indoor_std'] = df[rh_cols].std(axis=1)

    # 3. Chênh lệch nhiệt độ & độ ẩm giữa trong nhà và bên ngoài (Trạm thời tiết)
    df['T_diff_out'] = df['T_indoor_mean'] - df['T_out']
    df['RH_diff_out'] = df['RH_indoor_mean'] - df['RH_out']

    # 4. Đặc trưng trễ môi trường & quán tính tiêu thụ điện (Lag features)
    df['T_out_lag1'] = df['T_out'].shift(1).bfill()
    df['RH_out_lag1'] = df['RH_out'].shift(1).bfill()
    df['T_indoor_lag1'] = df['T_indoor_mean'].shift(1).bfill()
    
    # Mức tiêu thụ điện ở chu kỳ 10 phút và 20 phút trước (rất quan trọng vì máy móc chạy theo chu kỳ)
    df['Appliances_lag1'] = df['Appliances'].shift(1).bfill()
    df['Appliances_lag2'] = df['Appliances'].shift(2).bfill()

    print(f"[+] Đã tạo các nhóm đặc trưng mới. Kích thước bảng dữ liệu hiện tại: {df.shape}")
    return df


def split_time_series(df, test_ratio=0.2):
    """
    Chia tập Train/Test theo mốc thời gian để tránh RÒ RỈ DỮ LIỆU (Data Leakage).
    80% thời gian đầu dùng để Huấn luyện, 20% thời gian cuối dùng để Kiểm thử độc lập.
    """
    print("\n--- 3. PHÂN CHIA TẬP DỮ LIỆU (TIME-SERIES SPLIT) ---")
    split_idx = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    feature_cols = [
        c for c in df.columns 
        if c not in ['date', 'Appliances']
    ]
    target_col = 'Appliances'

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    print(f"Khoảng thời gian Train: từ {train_df['date'].min()} đến {train_df['date'].max()} ({len(train_df)} mẫu)")
    print(f"Khoảng thời gian Test : từ {test_df['date'].min()} đến {test_df['date'].max()} ({len(test_df)} mẫu)")
    print(f"Số lượng đặc trưng đầu vào: {len(feature_cols)}")

    return X_train, y_train, X_test, y_test, test_df, feature_cols


def evaluate_predictions(y_true, y_pred, model_name=""):
    """Tính toán MAE, RMSE, R2, MAPE."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-5))) * 100
    return {
        "Model": model_name,
        "MAE (Wh)": round(mae, 2),
        "RMSE (Wh)": round(rmse, 2),
        "R2 Score": round(r2, 4),
        "MAPE (%)": round(mape, 2)
    }


def train_and_compare_models(X_train, y_train, X_test, y_test):
    """
    Huấn luyện và so sánh:
    1. Baseline: Ridge Regression (có chuẩn hóa StandardScaler)
    2. Model 1: Random Forest Regressor
    3. Model 2 (Champion): Gradient Boosting (HistGradientBoosting hoặc LightGBM)
    """
    print("\n--- 4. HUẤN LUYỆN VÀ SO SÁNH CÁC MÔ HÌNH HỒI QUY ---")

    models = {}

    # 1. Baseline Model: Ridge (tuyến tính)
    models["Ridge Regression (Baseline)"] = Pipeline([
        ('scaler', StandardScaler()),
        ('reg', Ridge(alpha=100.0, random_state=42))
    ])

    # 2. Random Forest Regressor
    models["Random Forest"] = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42
    )

    # 3. Gradient Boosting Regressor (State-of-the-Art Tabular)
    if HAS_LIGHTGBM:
        models["LightGBM Regressor"] = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
    else:
        models["HistGradientBoosting (SOTA)"] = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=42
        )

    results = []
    predictions_dict = {}

    for name, model in models.items():
        print(f"[*] Đang huấn luyện: {name} ...")
        model.fit(X_train, y_train)
        
        pred = model.predict(X_test)
        pred = np.clip(pred, a_min=0, a_max=None) # Lượng điện không âm

        metrics = evaluate_predictions(y_test, pred, model_name=name)
        results.append(metrics)
        predictions_dict[name] = pred
        print(f"    -> MAE: {metrics['MAE (Wh)']} Wh | RMSE: {metrics['RMSE (Wh)']} Wh | R2: {metrics['R2 Score']}")

    results_df = pd.DataFrame(results)
    
    # Chọn mô hình tốt nhất dựa trên RMSE thấp nhất
    best_row = results_df.sort_values(by="RMSE (Wh)").iloc[0]
    best_model_name = best_row['Model']
    best_model = models[best_model_name]
    print(f"\n[★] MÔ HÌNH TỐI ƯU NHẤT: {best_model_name} (RMSE: {best_row['RMSE (Wh)']} Wh, R2: {best_row['R2 Score']})")

    return results_df, models, predictions_dict, best_model, best_model_name


def plot_visualizations(results_df, test_df, y_test, predictions_dict, best_model, best_model_name, feature_cols):
    """Vẽ và lưu trữ các biểu đồ minh chứng thực nghiệm."""
    print("\n--- 5. TRỰC QUAN HÓA KẾT QUẢ & XUẤT MINH CHỨNG ---")
    
    # 1. Biểu đồ so sánh Benchmark các mô hình
    plt.figure(figsize=(10, 5))
    x = np.arange(len(results_df))
    width = 0.25

    plt.bar(x - width, results_df['MAE (Wh)'], width, label='MAE (Wh)', color='#4C72B0')
    plt.bar(x, results_df['RMSE (Wh)'], width, label='RMSE (Wh)', color='#DD8452')
    plt.bar(x + width, results_df['R2 Score'] * 100, width, label='R² Score (x100)', color='#55A868')

    plt.xticks(x, results_df['Model'], rotation=10, fontsize=10)
    plt.ylabel('Giá trị chỉ số')
    plt.title('So sánh hiệu năng giữa các thuật toán hồi quy', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    comp_path = os.path.join(OUTPUT_DIR, "metrics_comparison.png")
    plt.savefig(comp_path, dpi=200)
    plt.close()
    print(f"[+] Đã lưu biểu đồ so sánh: {comp_path}")

    # 2. Biểu đồ Dự đoán vs Thực tế (Actual vs Predicted) trên mẫu 4 ngày liên tiếp
    sample_window = 144 * 4  # Mỗi ngày có 144 mẫu (10 phút/mẫu)
    plot_sub_df = test_df.iloc[:sample_window].copy()
    y_true_sub = y_test.iloc[:sample_window]
    y_pred_sub = predictions_dict[best_model_name][:sample_window]

    plt.figure(figsize=(14, 5))
    plt.plot(plot_sub_df['date'], y_true_sub, label='Thực tế (Actual)', color='black', alpha=0.7, linewidth=1.5)
    plt.plot(plot_sub_df['date'], y_pred_sub, label=f'Dự đoán ({best_model_name})', color='#E24A33', linewidth=1.5, linestyle='--')
    plt.fill_between(plot_sub_df['date'], y_true_sub, y_pred_sub, color='#E24A33', alpha=0.1)

    plt.title(f'So sánh Giá trị Thực tế vs Dự đoán theo chuỗi thời gian (4 ngày kiểm thử)', fontsize=13, fontweight='bold')
    plt.xlabel('Thời gian')
    plt.ylabel('Mức tiêu thụ điện (Wh)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    ts_path = os.path.join(OUTPUT_DIR, "actual_vs_predicted.png")
    plt.savefig(ts_path, dpi=200)
    plt.close()
    print(f"[+] Đã lưu biểu đồ dự đoán chuỗi thời gian: {ts_path}")

    # 3. Biểu đồ Mức độ quan trọng của đặc trưng (Feature Importance)
    importances = None
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    elif hasattr(best_model, 'named_steps') and hasattr(best_model.named_steps['reg'], 'coef_'):
        importances = np.abs(best_model.named_steps['reg'].coef_)

    if importances is not None:
        feat_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
        feat_df = feat_df.sort_values('Importance', ascending=False).head(15)

        plt.figure(figsize=(10, 6))
        plt.barh(feat_df['Feature'][::-1], feat_df['Importance'][::-1], color='#2b5c8f')
        plt.title(f'Top 15 đặc trưng quan trọng nhất ({best_model_name})', fontsize=13, fontweight='bold')
        plt.xlabel('Trọng số quan trọng (Feature Importance)')
        plt.tight_layout()
        feat_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
        plt.savefig(feat_path, dpi=200)
        plt.close()
        print(f"[+] Đã lưu biểu đồ đặc trưng quan trọng: {feat_path}")


def main():
    print("=" * 65)
    print(" HỆ THỐNG DỰ ĐOÁN TIÊU THỤ ĐIỆN NHÀ THÔNG MINH (UCI DATASET)")
    print("=" * 65)

    # 1. Tải dữ liệu
    download_dataset()

    # 2. Đọc và tiền xử lý
    df = load_and_preprocess_data()

    # 3. Kỹ thuật đặc trưng
    df = engineer_features(df)

    # 4. Phân chia tập dữ liệu
    X_train, y_train, X_test, y_test, test_df, feature_cols = split_time_series(df)

    # 5. Huấn luyện và đánh giá
    results_df, models, predictions_dict, best_model, best_model_name = train_and_compare_models(
        X_train, y_train, X_test, y_test
    )

    print("\n" + "=" * 55)
    print(" BẢNG TỔNG HỢP SO SÁNH HIỆU NĂNG MÔ HÌNH")
    print("=" * 55)
    print(results_df.to_string(index=False))

    # Lưu bảng kết quả ra file csv
    benchmark_csv = os.path.join(OUTPUT_DIR, "benchmark_results.csv")
    results_df.to_csv(benchmark_csv, index=False)
    print(f"\n[+] Đã lưu bảng benchmark ra: {benchmark_csv}")

    # 6. Vẽ đồ thị trực quan hóa
    plot_visualizations(results_df, test_df, y_test, predictions_dict, best_model, best_model_name, feature_cols)

    # 7. Lưu mô hình tốt nhất
    model_bundle = {
        'model': best_model,
        'model_name': best_model_name,
        'feature_cols': feature_cols,
        'metrics': results_df.to_dict(orient='records')
    }
    model_save_path = os.path.join(OUTPUT_DIR, "best_energy_model.pkl")
    joblib.dump(model_bundle, model_save_path)
    print(f"[+] Đã đóng gói và lưu mô hình tối ưu ra: {model_save_path}")
    print("\n[✓] HOÀN TẤT THỰC NGHIỆM THÀNH CÔNG!")


if __name__ == "__main__":
    main()
