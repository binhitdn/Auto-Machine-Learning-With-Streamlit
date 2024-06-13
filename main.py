import streamlit as st
from streamlit_option_menu import option_menu
import os
import pandas as pd

from pagess.upload_page import upload_page
from pagess.data_analysis_page import trang_phan_tich
from pagess.prediction_model_page import hoi_quy
from pagess.data_cleaning_page import phan_tich_va_lam_sach_du_lieu
from pagess.recommendation_system import recommendation_system

st.title("Machine Learning")
st.write("Trước tiên hãy up file lên")
st.write("Chọn một tùy chọn từ thanh bên trái")
st.set_option('deprecation.showPyplotGlobalUse', False)

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)
    st.session_state.df = df
    st.session_state.columns = df.columns.tolist()

css_style = {
    "icon": {"color": "black"},
    "icon_selected": {"color": "white"},
    "nav-link": {"--hover-color": "grey"},
    "nav-link-selected": {"background-color": "#FF4C1B", "color": "white", "font-weight": "400"},
}

# Thiết lập thanh menu lựa chọn (sidebar)
with st.sidebar:
    st.info("Abc")
    selected = option_menu(
        menu_title=None,
        options=["Trang Tải Lên Dữ Liệu", "Trang Phân Tích Dữ Liệu", "Mô Hình Dự Đoán", "Phân Tích và Làm Sạch Dữ Liệu", "Hệ Thống Đề Xuất"],
        icons=["cloud-upload", "clipboard-data", "cpu", "download", "calculator", "chart-line", "chart-bar", "chart-pie"],
        styles=css_style
    )

if selected == "Trang Tải Lên Dữ Liệu":
    upload_page()

elif selected == "Trang Phân Tích Dữ Liệu":
    trang_phan_tich()

elif selected == "Mô Hình Dự Đoán":
    hoi_quy()

elif selected == "Phân Tích và Làm Sạch Dữ Liệu":
    phan_tich_va_lam_sach_du_lieu()
elif selected == "Hệ Thống Đề Xuất":
    recommendation_system()
