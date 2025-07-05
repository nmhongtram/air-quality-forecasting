import streamlit as st
import requests
import pandas as pd
from matplotlib import cm
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import time


# Config
API_URL = os.getenv("API_URL", "http://localhost:8000")
STATION_NAME = "Hà Nội - Trạm An Khánh"
COLORS = {
    'pm25': '#FF5722',
    'co': '#4CAF50',
    'no2': '#2196F3', 
    'aqi': '#9C27B0'
}
WHO_PM25_GUIDELINE = 5

# Breakpoints cho PM2.5 (µg/m³)
PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]


# Layout
st.set_page_config(page_title="Air Quality Dashboard", layout="wide")
st.title(f"🌎 GIÁM SÁT CHẤT LƯỢNG KHÔNG KHÍ 🌎")
st.subheader(f"{STATION_NAME}")

st.markdown("""
    <style>
        /* Đổi màu nền sidebar */
        section[data-testid="stSidebar"] {
            background-color: #B3E5FC;
        }

        /* Màu chữ trong sidebar */
        section[data-testid="stSidebar"] * {
            color: #01579B;
        }
    </style>
""", unsafe_allow_html=True)



# Sidebar controls
with st.sidebar:
    st.header("Cài đặt")
    auto_refresh = st.checkbox("Tự động cập nhật", value=True, key="auto_refresh")
    refresh_interval = st.slider("Khoảng thời gian cập nhật (phút)", 1, 60, 30, key="refresh_interval")

    model_options = ["RNN", "LSTM", "GRU"]
    selected_model = st.selectbox("Chọn mô hình dự đoán", model_options, index=0, key="model_select")

    num_days = st.slider("Hiển thị dữ liệu quá khứ (ngày)", min_value=0, max_value=4, value=2, step=1, key="num_days")


# Main content
placeholder = st.empty()

def get_aqi_message_color(aqi):
    if aqi <= 50:
        return "🟢 Tốt: Không ảnh hưởng đến sức khỏe.", "success"
    elif aqi <= 100:
        return "🟡 Vừa phải: Có thể ảnh hưởng nhẹ đến nhóm nhạy cảm.", "info"
    elif aqi <= 150:
        return "🟠 Kém: Nhóm nhạy cảm nên hạn chế hoạt động ngoài trời.", "warning"
    elif aqi <= 200:
        return "🔴 Xấu: Nhóm nhạy cảm tránh ra ngoài. Người khỏe mạnh nên hạn chế.", "error"
    elif aqi <= 300:
        return "🟣 Rất xấu: Hạn chế tối đa hoạt động ngoài trời.", "error"
    else:
        return "🟤 Nguy hại: Tình trạng khẩn cấp về sức khỏe. Mọi người nên ở trong nhà.", "error"

# Hàm tính AQI
def compute_pm25_aqi(pm25):
    for Clow, Chigh, Ilow, Ihigh in PM25_BREAKPOINTS:
        if Clow <= pm25 <= Chigh:
            return round(((Ihigh - Ilow) / (Chigh - Clow)) * (pm25 - Clow) + Ilow)
    return None  

def calculate_aqi(row):
    if pd.notna(row['pm25']):
        return compute_pm25_aqi(row['pm25'])
    return None


# Cache dữ liệu lịch sử
@st.cache_data(ttl=600)  # cache trong 10 phút
def fetch_historical():
    return requests.get(f"{API_URL}/historical").json()

# Cache dự đoán theo mô hình
@st.cache_data(ttl=600)
def fetch_prediction(model_name):
    return requests.get(f"{API_URL}/predict?model={model_name}").json()


def update_dashboard():
    with placeholder.container():
        try:
            # Lấy dữ liệu
            historical = fetch_historical()
            prediction = fetch_prediction(selected_model.lower())

            # Tạo DataFrames
            if num_days == 0:
                df_hist = pd.DataFrame(historical).iloc[-1:]
            else:
                hours_hist = num_days * 24
                df_hist = pd.DataFrame(historical).iloc[-hours_hist:]
            df_pred = pd.DataFrame(prediction['prediction'])

            df_hist['aqi'] = df_hist.apply(calculate_aqi, axis=1)
            df_pred['aqi'] = df_pred.apply(calculate_aqi, axis=1)

            # Dán nhãn kiểu dữ liệu
            df_hist['type'] = 'hist'
            df_pred['type'] = 'pred'

            # Thông tin hiện tại
            latest = df_hist.iloc[-1]
            latest_aqi = latest['aqi']
            latest_pm25 = latest['pm25']
            # col1, col2, col3, col4 = st.columns(4)
            # col1.metric("AQI", f"{latest_aqi:.0f}")
            # col2.metric("PM2.5", f"{latest['pm25']:.1f} µg/m³")
            # col3.metric("NO2", f"{latest['no2']:.1f} µg/m³")
            # col4.metric("CO", f"{latest['co']:.1f} µg/m³")
           
            col1, col2, col3, col4 = st.columns(4, gap="large")

            def bordered_metric(col, label, value):
                # Mô tả các nhãn
                descriptions = {
                    "AQI": "Air Quality Index",
                    "PM2.5": "Hạt mịn (≤2.5 µm)",
                    "NO2": "Nitơ Dioxit",
                    "CO": "Cacbon Monoxit"
                }
                description = descriptions.get(label, "")

                col.markdown(
                    f"""
                    <div style="
                        border: 2px solid #00aaff; 
                        border-radius: 8px; 
                        padding: 10px; 
                        text-align: left;
                        width: 100%;
                    ">
                        <h4 style="margin: 0; font-size: 20px; color: gray;">{label}</h4>
                        <p style="margin: 0; font-size: 14px; color: #888;">{description}</p>
                        <p style="font-size: 32px; margin: 2px 0 0; font-weight: bold;">{value}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            bordered_metric(col1, "AQI", f"{latest_aqi:.0f}")
            bordered_metric(col2, "PM2.5", f"{latest['pm25']:.1f} µg/m³")
            bordered_metric(col3, "NO2", f"{latest['no2']:.1f} µg/m³")
            bordered_metric(col4, "CO", f"{latest['co']:.1f} µg/m³")

            st.write("\n")  # Thêm khoảng trắng giữa các phần


            # Hiển thị cảnh báo AQI
            message, level = get_aqi_message_color(latest_aqi)
            if level == "success":
                st.success(message)
            elif level == "info":
                st.info(message)
            elif level == "warning":
                st.warning(message)
            elif level == "error":
                st.error(message)

            # Cảnh báo riêng cho PM2.5
            pm25_multiple = latest_pm25 / WHO_PM25_GUIDELINE


            if pm25_multiple >= 3.0:
                st.error(
                    f"‼️ Nồng độ PM2.5 hiện tại gấp **{pm25_multiple:.1f} lần** so với "
                    f"giá trị hướng dẫn hàng năm của WHO (PM2.5 = {WHO_PM25_GUIDELINE})."
                )
            elif pm25_multiple >= 1.5 and pm25_multiple < 3.0:
                st.warning(
                    f"⚠️ Nồng độ PM2.5 hiện tại gấp **{pm25_multiple:.1f} lần** so với "
                    f"giá trị hướng dẫn hàng năm của WHO (PM2.5 = {WHO_PM25_GUIDELINE})."
                )
            elif pm25_multiple >= 1.0:
                st.info(
                    f"🟡 Nồng độ PM2.5 hiện tại vượt **giá trị khuyến nghị của WHO** "
                    f"(PM2.5 = {WHO_PM25_GUIDELINE})."
                )

            st.write("\n")  # Thêm khoảng trắng giữa các phần
            st.subheader("Biểu đồ chất lượng không khí")

            # Tạo subplot
            fig = make_subplots(
                rows=4, cols=1, 
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=("AQI", "PM2.5", "NO2", "CO")
            )

            # ===== VẼ AQI gradient =====
            def get_color(aqi):
                if aqi <= 50: return "green"
                elif aqi <= 100: return "yellow"
                elif aqi <= 150: return "orange"
                elif aqi <= 200: return "red"
                elif aqi <= 300: return "purple"
                else: return "maroon"

            df_hist['color'] = df_hist['aqi'].apply(get_color)
            df_pred['color'] = df_pred['aqi'].apply(get_color)
            df_hist['message'] = df_hist['aqi'].apply(lambda aqi: get_aqi_message_color(aqi)[0])
            df_pred['message'] = df_pred['aqi'].apply(lambda aqi: get_aqi_message_color(aqi)[0])

            # Vẽ AQI quá khứ
            fig.add_trace(go.Scattergl(
                x=df_hist['datetimeFrom_local'],
                y=df_hist['aqi'],
                mode='lines+markers',
                marker=dict(color=df_hist['color'], size=6),
                line=dict(color='gray'),
                hovertemplate="%{y}<br>%{customdata}",
                customdata=df_hist['message'],
                name='AQI (hist)'
            ), row=1, col=1)

            # Vẽ AQI dự đoán
            fig.add_trace(go.Scattergl(
                x=df_pred['datetimeFrom_local'],
                y=df_pred['aqi'],
                mode='lines+markers',
                marker=dict(color=df_pred['color'], size=6),
                line=dict(color='gray', dash='dot'),
                hovertemplate="%{y}<br>%{customdata}",
                customdata=df_pred['message'],
                name='AQI (pred)'
            ), row=1, col=1)

            # Nối AQI
            fig.add_trace(go.Scatter(
                x=[df_hist['datetimeFrom_local'].iloc[-1], df_pred['datetimeFrom_local'].iloc[0]],
                y=[df_hist['aqi'].iloc[-1], df_pred['aqi'].iloc[0]],
                mode='lines',
                line=dict(color='gray', dash='dot'),
                showlegend=False
            ), row=1, col=1)

            # ===== VẼ PM2.5, NO2, CO =====
            def hex_to_rgba(hex_color, alpha=0.2):
                hex_color = hex_color.lstrip('#')
                r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                return f'rgba({r}, {g}, {b}, {alpha})'

            def add_dual_trace(row, var, color):
                rgba_fill = hex_to_rgba(color, alpha=0.2)

                # Vẽ vùng fill cho dữ liệu quá khứ
                fig.add_trace(go.Scatter(
                    x=list(df_hist['datetimeFrom_local']) + list(df_hist['datetimeFrom_local'][::-1]),
                    y=list(df_hist[var]) + [0]*len(df_hist),
                    fill='toself',
                    fillcolor=rgba_fill,
                    line=dict(color='rgba(0,0,0,0)'),
                    hoverinfo='skip',
                    showlegend=False
                ), row=row, col=1)

                # Vẽ đường line cho dữ liệu quá khứ
                fig.add_trace(go.Scatter(
                    x=df_hist['datetimeFrom_local'], y=df_hist[var],
                    mode='lines', name=f'{var.upper()} (hist)',
                    line=dict(color=color),
                ), row=row, col=1)

                # Vẽ vùng fill cho dự đoán
                fig.add_trace(go.Scatter(
                    x=list(df_pred['datetimeFrom_local']) + list(df_pred['datetimeFrom_local'][::-1]),
                    y=list(df_pred[var]) + [0]*len(df_pred),
                    fill='toself',
                    fillcolor=rgba_fill,
                    line=dict(color='rgba(0,0,0,0)'),
                    hoverinfo='skip',
                    showlegend=False
                ), row=row, col=1)

                # Vẽ đường line dự đoán
                fig.add_trace(go.Scatter(
                    x=df_pred['datetimeFrom_local'], y=df_pred[var],
                    mode='lines', name=f'{var.upper()} (pred)',
                    line=dict(color=color, dash='dot')
                ), row=row, col=1)

                # Đường nối giữa quá khứ và dự đoán (line)
                x_connect = [df_hist['datetimeFrom_local'].iloc[-1], df_pred['datetimeFrom_local'].iloc[0]]
                y_connect = [df_hist[var].iloc[-1], df_pred[var].iloc[0]]

                # Fill dưới đoạn nối
                fig.add_trace(go.Scatter(
                    x=x_connect + x_connect[::-1],
                    y=y_connect + [0, 0],
                    fill='toself',
                    fillcolor=rgba_fill,
                    line=dict(color='rgba(0,0,0,0)'),
                    hoverinfo='skip',
                    showlegend=False
                ), row=row, col=1)

                # Line đoạn nối
                fig.add_trace(go.Scatter(
                    x=x_connect, y=y_connect,
                    mode='lines',
                    line=dict(color=color, dash='dot'),
                    showlegend=False
                ), row=row, col=1)


            add_dual_trace(2, 'pm25', COLORS['pm25'])
            add_dual_trace(3, 'no2', COLORS['no2'])
            add_dual_trace(4, 'co', COLORS['co'])

            # Layout
            fig.update_layout(
                height=900,
                title_text=f"Chất lượng không khí: Quan sát & Dự đoán ({selected_model.upper()})",
                hovermode="x unified",
                showlegend=True,
                plot_bgcolor="#f8f8f8"
            )

            fig.update_yaxes(title_text="AQI", row=1, col=1)
            for i in range(2, 5):
                fig.update_yaxes(title_text="µg/m³", row=i, col=1)
            fig.update_xaxes(title_text="Thời gian", row=4, col=1)

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Lỗi kết nối: {str(e)}")



# Auto-refresh logic
if auto_refresh:
    while True:
        update_dashboard()
        time.sleep(refresh_interval * 60)
else:
    update_dashboard()


