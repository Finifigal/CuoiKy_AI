"""
Ứng dụng Web Streamlit: Hệ thống Dự đoán Mức tiêu thụ Điện trong Nhà thông minh
Tập dữ liệu: UCI Appliances Energy Prediction
Chạy ứng dụng: streamlit run app.py
"""

import os
import sys

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="Dự đoán Tiêu thụ Điện Nhà thông minh",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Đường dẫn thư mục
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_energy_model.pkl")
BENCHMARK_PATH = os.path.join(OUTPUT_DIR, "benchmark_results.csv")


@st.cache_resource
def load_trained_model():
    """Tải mô hình và các thông tin đã lưu trữ."""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_benchmark():
    """Tải bảng chỉ số đánh giá mô hình."""
    if os.path.exists(BENCHMARK_PATH):
        return pd.read_csv(BENCHMARK_PATH)
    return None


def predict_energy(model_bundle, input_dict):
    """Tiền xử lý dữ liệu đầu vào và thực hiện dự đoán lượng điện tiêu thụ."""
    model = model_bundle['model']
    feature_cols = model_bundle['feature_cols']

    df_in = pd.DataFrame([input_dict])

    # 1. Tính các đặc trưng thời gian
    hour_val = input_dict['hour'] + input_dict['minute'] / 60.0
    day_of_week = input_dict['day_of_week']
    df_in['hour'] = input_dict['hour']
    df_in['day_of_week'] = day_of_week
    df_in['is_weekend'] = 1 if day_of_week >= 5 else 0
    df_in['is_night'] = 1 if (input_dict['hour'] >= 22 or input_dict['hour'] <= 6) else 0
    df_in['sin_hour'] = np.sin(2 * np.pi * hour_val / 24.0)
    df_in['cos_hour'] = np.cos(2 * np.pi * hour_val / 24.0)
    df_in['sin_dayofweek'] = np.sin(2 * np.pi * day_of_week / 7.0)
    df_in['cos_dayofweek'] = np.cos(2 * np.pi * day_of_week / 7.0)

    # 2. Đặc trưng tổng hợp không gian
    temp_cols = [f'T{i}' for i in range(1, 10)]
    rh_cols = [f'RH_{i}' for i in range(1, 10)]

    df_in['T_indoor_mean'] = df_in[temp_cols].mean(axis=1)
    df_in['T_indoor_std'] = df_in[temp_cols].std(axis=1).fillna(0)
    df_in['RH_indoor_mean'] = df_in[rh_cols].mean(axis=1)
    df_in['RH_indoor_std'] = df_in[rh_cols].std(axis=1).fillna(0)

    # 3. Chênh lệch môi trường trong và ngoài nhà
    df_in['T_diff_out'] = df_in['T_indoor_mean'] - df_in['T_out']
    df_in['RH_diff_out'] = df_in['RH_indoor_mean'] - df_in['RH_out']

    # 4. Trễ môi trường
    df_in['T_out_lag1'] = input_dict.get('T_out_lag1', input_dict['T_out'])
    df_in['RH_out_lag1'] = input_dict.get('RH_out_lag1', input_dict['RH_out'])
    df_in['T_indoor_lag1'] = input_dict.get('T_indoor_lag1', df_in['T_indoor_mean'].iloc[0])
    df_in['Appliances_lag1'] = input_dict.get('Appliances_lag1', 50.0)
    df_in['Appliances_lag2'] = input_dict.get('Appliances_lag2', 50.0)

    # Điền giá trị 0 cho các cột chưa có
    for col in feature_cols:
        if col not in df_in.columns:
            df_in[col] = 0.0

    X = df_in[feature_cols]
    pred = model.predict(X)[0]
    return float(max(0.0, pred))


def main():
    # Tiêu đề ứng dụng
    st.title("⚡ Hệ Thống Dự Đoán Tiêu Thụ Điện Trong Nhà Thông Minh")
    st.markdown(
        "**Đề tài số 10** | Dựa trên tập dữ liệu chuẩn **UCI Appliances Energy Prediction**  \n"
        "*Ứng dụng học máy dự đoán mức tiêu thụ năng lượng (Wh) dựa trên cảm biến nhiệt độ, độ ẩm và thời tiết.*"
    )
    st.divider()

    model_bundle = load_trained_model()
    benchmark_df = load_benchmark()

    if model_bundle is None:
        st.error(
            "⚠️ Chưa tìm thấy tệp mô hình đã huấn luyện (`outputs/best_energy_model.pkl`)! "
            "Vui lòng chạy `python train.py` trong terminal trước khi sử dụng giao diện này."
        )
        return

    # Sidebar: Cài đặt và thông tin mô hình
    with st.sidebar:
        st.header("⚙️ Cấu hình Mô hình")
        st.success(f"Mô hình đang dùng: **{model_bundle['model_name']}**")
        st.info("Chu kỳ đo: **10 phút/lần** (Chuẩn UCI)")

        st.subheader("💡 Kịch bản mẫu nhanh (Preset)")
        preset = st.selectbox(
            "Chọn tình huống giả định:",
            [
                "Tùy chỉnh thủ công",
                "Buổi tối sinh hoạt gia đình (19:30)",
                "Nửa đêm đi ngủ (02:00)",
                "Buổi sáng chuẩn bị đi làm (07:30)"
            ]
        )

    # Đặt giá trị mặc định theo Preset
    default_vals = {
        'hour': 19, 'minute': 30, 'day_of_week': 2, 'lights': 30,
        'T_indoor': 21.5, 'RH_indoor': 45.0,
        'T_out': 16.0, 'RH_out': 72.0, 'Press': 755.0, 'Wind': 4.0, 'Vis': 40.0,
        'app_lag1': 110.0
    }

    if preset == "Nửa đêm đi ngủ (02:00)":
        default_vals = {
            'hour': 2, 'minute': 0, 'day_of_week': 3, 'lights': 0,
            'T_indoor': 20.0, 'RH_indoor': 40.0,
            'T_out': 12.0, 'RH_out': 85.0, 'Press': 758.0, 'Wind': 2.0, 'Vis': 30.0,
            'app_lag1': 40.0
        }
    elif preset == "Buổi sáng chuẩn bị đi làm (07:30)":
        default_vals = {
            'hour': 7, 'minute': 30, 'day_of_week': 1, 'lights': 20,
            'T_indoor': 21.0, 'RH_indoor': 43.0,
            'T_out': 14.5, 'RH_out': 78.0, 'Press': 756.0, 'Wind': 3.5, 'Vis': 35.0,
            'app_lag1': 80.0
        }

    # Bố cục 4 Tabs chính
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Dự đoán tương tác",
        "📊 So sánh các mô hình",
        "📈 Biểu đồ thực nghiệm",
        "ℹ️ Giới thiệu đề tài & Dữ liệu"
    ])

    # ------------------ TAB 1: DỰ ĐOÁN TƯƠNG TÁC ------------------
    with tab1:
        st.subheader("Nhập thông số cảm biến môi trường để dự đoán")
        st.write("Điều chỉnh các thông số dưới đây hoặc chọn kịch bản mẫu ở thanh bên trái:")

        col_time, col_weather = st.columns(2)

        with col_time:
            st.markdown("##### 🕒 Thời gian & Chiếu sáng")
            hour = st.slider("Giờ trong ngày (0 - 23h)", 0, 23, default_vals['hour'])
            minute = st.select_slider("Phút", options=[0, 10, 20, 30, 40, 50], value=default_vals['minute'])
            day_of_week = st.selectbox(
                "Ngày trong tuần",
                options=[0, 1, 2, 3, 4, 5, 6],
                format_func=lambda x: ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"][x],
                index=default_vals['day_of_week']
            )
            lights = st.slider("Công suất đèn chiếu sáng (Wh)", 0, 100, default_vals['lights'], step=10)

        with col_weather:
            st.markdown("##### ⛅ Thời tiết trạm khí tượng bên ngoài")
            t_out = st.slider("Nhiệt độ ngoài trời T_out (°C)", -5.0, 40.0, float(default_vals['T_out']), step=0.5)
            rh_out = st.slider("Độ ẩm ngoài trời RH_out (%)", 10.0, 100.0, float(default_vals['RH_out']), step=1.0)
            press = st.number_input("Áp suất khí quyển (mm Hg)", 720.0, 780.0, float(default_vals['Press']))
            windspeed = st.slider("Tốc độ gió (m/s)", 0.0, 15.0, float(default_vals['Wind']), step=0.5)

        st.markdown("##### 🏠 Cảm biến Nhiệt độ & Độ ẩm trong nhà")
        
        input_mode = st.radio(
            "Cách thức điều chỉnh:",
            ["⚡ Nhập nhanh toàn nhà (Tiện demo)", "🏠 Tinh chỉnh chi tiết từng phòng (9 phòng)"],
            horizontal=True
        )

        if input_mode == "⚡ Nhập nhanh toàn nhà (Tiện demo)":
            q_col1, q_col2 = st.columns(2)
            with q_col1:
                t_common = st.slider("🌡️ Nhiệt độ trung bình toàn nhà (°C)", 15.0, 35.0, float(default_vals['T_indoor']), step=0.5)
            with q_col2:
                rh_common = st.slider("💧 Độ ẩm trung bình toàn nhà (%)", 20.0, 90.0, float(default_vals['RH_indoor']), step=1.0)
            
            # Gán giá trị đồng bộ kèm độ lệch tự nhiên giữa các phòng
            t1, t2, t3 = t_common + 0.3, t_common - 0.2, t_common + 0.1
            t4, t5, t6 = t_common, t_common - 0.3, float(t_out) + 1.0
            t7, t8, t9 = t_common, t_common + 0.2, t_common
            
            rh1, rh2, rh3 = rh_common + 2.0, rh_common - 2.0, rh_common + 3.0
            rh4, rh5, rh6 = rh_common, rh_common + 4.0, float(rh_out)
            rh7, rh8, rh9 = rh_common, rh_common, rh_common
            
            st.caption("ℹ️ Đang tự động phân bổ nhiệt độ & độ ẩm cho cả 9 phòng theo mức trung bình đã chọn.")
        else:
            st.write("Mỗi phòng được nhóm thành 1 thẻ riêng (Nhiệt độ đặt cạnh Độ ẩm):")
            r_col1, r_col2, r_col3 = st.columns(3)

            with r_col1:
                with st.container(border=True):
                    st.markdown("🍳 **1. Khu vực bếp**")
                    c1, c2 = st.columns(2)
                    t1 = c1.number_input("Nhiệt T1 (°C)", 15.0, 35.0, default_vals['T_indoor'], step=0.2, key="t1")
                    rh1 = c2.number_input("Ẩm RH_1 (%)", 20.0, 95.0, default_vals['RH_indoor'], step=0.5, key="rh1")

                with st.container(border=True):
                    st.markdown("🛋️ **2. Phòng khách**")
                    c1, c2 = st.columns(2)
                    t2 = c1.number_input("Nhiệt T2 (°C)", 15.0, 35.0, default_vals['T_indoor'] - 0.5, step=0.2, key="t2")
                    rh2 = c2.number_input("Ẩm RH_2 (%)", 20.0, 95.0, default_vals['RH_indoor'] - 2.0, step=0.5, key="rh2")

                with st.container(border=True):
                    st.markdown("🧺 **3. Khu giặt đồ**")
                    c1, c2 = st.columns(2)
                    t3 = c1.number_input("Nhiệt T3 (°C)", 15.0, 35.0, default_vals['T_indoor'] + 0.3, step=0.2, key="t3")
                    rh3 = c2.number_input("Ẩm RH_3 (%)", 20.0, 95.0, default_vals['RH_indoor'] + 3.0, step=0.5, key="rh3")

            with r_col2:
                with st.container(border=True):
                    st.markdown("💻 **4. Phòng làm việc**")
                    c1, c2 = st.columns(2)
                    t4 = c1.number_input("Nhiệt T4 (°C)", 15.0, 35.0, default_vals['T_indoor'], step=0.2, key="t4")
                    rh4 = c2.number_input("Ẩm RH_4 (%)", 20.0, 95.0, default_vals['RH_indoor'], step=0.5, key="rh4")

                with st.container(border=True):
                    st.markdown("🚿 **5. Phòng tắm**")
                    c1, c2 = st.columns(2)
                    t5 = c1.number_input("Nhiệt T5 (°C)", 15.0, 35.0, default_vals['T_indoor'] - 0.2, step=0.2, key="t5")
                    rh5 = c2.number_input("Ẩm RH_5 (%)", 20.0, 95.0, default_vals['RH_indoor'] + 5.0, step=0.5, key="rh5")

                with st.container(border=True):
                    st.markdown("🧱 **6. Mặt ngoài phía Bắc**")
                    c1, c2 = st.columns(2)
                    t6 = c1.number_input("Nhiệt T6 (°C)", 5.0, 40.0, default_vals['T_out'] + 1.0, step=0.2, key="t6")
                    rh6 = c2.number_input("Ẩm RH_6 (%)", 10.0, 100.0, default_vals['RH_out'], step=0.5, key="rh6")

            with r_col3:
                with st.container(border=True):
                    st.markdown("👔 **7. Phòng ủi đồ**")
                    c1, c2 = st.columns(2)
                    t7 = c1.number_input("Nhiệt T7 (°C)", 15.0, 35.0, default_vals['T_indoor'], step=0.2, key="t7")
                    rh7 = c2.number_input("Ẩm RH_7 (%)", 20.0, 95.0, default_vals['RH_indoor'], step=0.5, key="rh7")

                with st.container(border=True):
                    st.markdown("🛏️ **8. Phòng thiếu niên**")
                    c1, c2 = st.columns(2)
                    t8 = c1.number_input("Nhiệt T8 (°C)", 15.0, 35.0, default_vals['T_indoor'] + 0.2, step=0.2, key="t8")
                    rh8 = c2.number_input("Ẩm RH_8 (%)", 20.0, 95.0, default_vals['RH_indoor'], step=0.5, key="rh8")

                with st.container(border=True):
                    st.markdown("🛌 **9. Phòng cha mẹ**")
                    c1, c2 = st.columns(2)
                    t9 = c1.number_input("Nhiệt T9 (°C)", 15.0, 35.0, default_vals['T_indoor'], step=0.2, key="t9")
                    rh9 = c2.number_input("Ẩm RH_9 (%)", 20.0, 95.0, default_vals['RH_indoor'], step=0.5, key="rh9")

        st.markdown("##### ⚡ Quán tính tiêu thụ điện chu kỳ trước")
        app_lag1 = st.slider("Lượng điện tiêu thụ 10 phút trước (Wh)", 10.0, 600.0, float(default_vals['app_lag1']), step=5.0)

        predict_btn = st.button("🚀 BẮT ĐẦU DỰ ĐOÁN", type="primary", use_container_width=True)

        if predict_btn:
            input_dict = {
                'hour': hour, 'minute': minute, 'day_of_week': day_of_week, 'lights': lights,
                'T1': t1, 'RH_1': rh1, 'T2': t2, 'RH_2': rh2, 'T3': t3, 'RH_3': rh3,
                'T4': t4, 'RH_4': rh4, 'T5': t5, 'RH_5': rh5, 'T6': t6, 'RH_6': rh6,
                'T7': t7, 'RH_7': rh7, 'T8': t8, 'RH_8': rh8, 'T9': t9, 'RH_9': rh9,
                'T_out': t_out, 'RH_out': rh_out, 'Press_mm_hg': press,
                'Windspeed': windspeed, 'Visibility': 40.0, 'Tdewpoint': 10.0,
                'T_out_lag1': t_out, 'RH_out_lag1': rh_out,
                'Appliances_lag1': app_lag1, 'Appliances_lag2': app_lag1
            }

            pred_wh = predict_energy(model_bundle, input_dict)
            pred_kwh = pred_wh / 1000.0

            st.success("✅ Dự đoán thành công!")
            res_col1, res_col2, res_col3 = st.columns(3)

            with res_col1:
                st.metric(
                    label="Mức tiêu thụ dự kiến (Chu kỳ 10 phút)",
                    value=f"{pred_wh:.1f} Wh"
                )
            with res_col2:
                st.metric(
                    label="Công suất quy đổi (kWh)",
                    value=f"{pred_kwh:.4f} kWh"
                )
            with res_col3:
                # Đánh giá mức tiêu thụ
                if pred_wh < 80:
                    status_text = "🟢 Mức tiêu thụ Thấp (Thiết bị chờ / Ban đêm)"
                elif pred_wh < 200:
                    status_text = "🟡 Mức tiêu thụ Trung bình (Sinh hoạt thông thường)"
                else:
                    status_text = "🔴 Mức tiêu thụ Cao (Đang dùng thiết bị công suất lớn)"
                st.info(f"**Đánh giá tải:**  \n{status_text}")

    # ------------------ TAB 2: SO SÁNH CÁC MÔ HÌNH ------------------
    with tab2:
        st.subheader("Bảng so sánh hiệu năng các mô hình hồi quy (Benchmark)")
        st.write("Đánh giá độc lập trên tập Test (20% thời gian cuối, 3,947 mẫu dữ liệu):")

        if benchmark_df is not None:
            st.dataframe(benchmark_df, use_container_width=True)

            # Vẽ biểu đồ trực tiếp
            fig, ax = plt.subplots(figsize=(8, 4))
            x = np.arange(len(benchmark_df))
            w = 0.25
            ax.bar(x - w, benchmark_df['MAE (Wh)'], w, label='MAE (Wh)', color='#3274D9')
            ax.bar(x, benchmark_df['RMSE (Wh)'], w, label='RMSE (Wh)', color='#E07A5F')
            ax.bar(x + w, benchmark_df['R2 Score'] * 100, w, label='R² Score (x100)', color='#81B29A')

            ax.set_xticks(x)
            ax.set_xticklabels(benchmark_df['Model'])
            ax.set_ylabel("Giá trị")
            ax.set_title("So sánh sai số (MAE, RMSE) và độ chính xác (R²)")
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.6)
            st.pyplot(fig)
        else:
            st.info("Chưa có tệp benchmark_results.csv.")

        st.markdown("""
        **Phân tích học thuật:**
        - **Ridge Regression (Baseline):** Đạt điểm $R^2 = 0.5511$, sai số $MAE = 28.26\\text{ Wh}$. Nhờ có thành phần điều chuẩn L2 và chuẩn hóa dữ liệu, mô hình tuyến tính giữ được tính ổn định rất cao và không bị overfit bởi các gai xung điện đột biến.
        - **HistGradientBoosting (SOTA):** Đạt $R^2 = 0.4527$, thể hiện khả năng học các phân nhánh phi tuyến tính phức tạp giữa nhiệt độ các phòng và độ ẩm ngoài trời.
        - **Random Forest:** Dễ bị khuếch đại sai số ở các đỉnh tiêu thụ đột biến trên dữ liệu chuỗi thời gian nếu không được tinh chỉnh siêu tham số chuyên sâu.
        """)

    # ------------------ TAB 3: BIỂU ĐỒ THỰC NGHIỆM ------------------
    with tab3:
        st.subheader("Trực quan hóa Thực nghiệm")
        col_img1, col_img2 = st.columns(2)

        ts_plot_path = os.path.join(OUTPUT_DIR, "actual_vs_predicted.png")
        feat_plot_path = os.path.join(OUTPUT_DIR, "feature_importance.png")

        with col_img1:
            st.markdown("##### 📈 Giá trị Thực tế vs Dự đoán (4 ngày kiểm thử)")
            if os.path.exists(ts_plot_path):
                st.image(ts_plot_path, use_container_width=True)
            else:
                st.info("Chưa tìm thấy biểu đồ actual_vs_predicted.png.")

        with col_img2:
            st.markdown("##### 🏆 Top đặc trưng quan trọng nhất (Feature Importance)")
            if os.path.exists(feat_plot_path):
                st.image(feat_plot_path, use_container_width=True)
            else:
                st.info("Chưa tìm thấy biểu đồ feature_importance.png.")

    # ------------------ TAB 4: THÔNG TIN ĐỀ TÀI & DỮ LIỆU ------------------
    with tab4:
        st.subheader("Giới thiệu Đề tài & Tập dữ liệu UCI")
        st.markdown(r"""
        ### Đề tài 10: Hệ thống dự đoán mức tiêu thụ điện trong nhà thông minh
        - **Tập dữ liệu:** *Appliances Energy Prediction* (UCI Machine Learning Repository).
        - **Quy mô:** 19,735 quan sát thu thập định kỳ 10 phút một lần trong vòng 4.5 tháng.
        - **Biến mục tiêu ($y$):** `Appliances` - Lượng điện tiêu thụ của thiết bị gia dụng trong nhà (Wh).
        
        #### Các đặc trưng đầu vào ($X$):
        1. **Nhiệt độ ($T_1 \dots T_9$):** Cảm biến gắn tại Bếp, Khách, Giặt, Làm việc, Tắm, Mặt ngoài nhà, Ủi đồ, Phòng ngủ thiếu niên, Phòng ngủ cha mẹ.
        2. **Độ ẩm ($RH_1 \dots RH_9$):** Độ ẩm tương ứng tại 9 vị trí trên.
        3. **Thời tiết trạm khí tượng:** Nhiệt độ ngoài trời ($T_{out}$), Áp suất ($Press$), Độ ẩm ngoài ($RH_{out}$), Tốc độ gió ($Windspeed$), Tầm nhìn ($Visibility$), Điểm sương ($Tdewpoint$).
        4. **Chiếu sáng:** Tiêu thụ năng lượng của đèn chiếu sáng (`lights`).
        
        #### Các điểm sáng tạo về phương pháp luận:
        - **Loại bỏ biến nhiễu:** Loại bỏ 2 biến kiểm thử `rv1`, `rv2` của UCI.
        - **Time-Series Split:** Không xáo trộn ngẫu nhiên dữ liệu để chống rò rỉ tương lai (Data Leakage).
        - **Feature Engineering:** Mã hóa chu kỳ 24h bằng hàm sin/cos, tính nhiệt độ trung bình toàn nhà và độ chênh lệch môi trường trong - ngoài.
        """)


if __name__ == "__main__":
    main()
