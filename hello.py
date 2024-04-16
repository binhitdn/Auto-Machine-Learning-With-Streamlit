import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
import pickle
from streamlit_option_menu import option_menu
from streamlit_pandas_profiling import st_profile_report
import os
from pycaret.classification import setup, compare_models, pull, save_model
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


st.title("Machine Learnin ")
st.write("Trước tiên hãy up file lên")
st.write("Chọn một tùy chọn từ thanh bên trái")


# Khai báo biến df
df = None
css_style = {
    "icon": {"color": "black"},
    "icon_selected": {"color": "white"},
    "nav-link": {"--hover-color": "grey"},
    "nav-link-selected": {"background-color": "#FF4C1B"},
    "title": {"background-color": "#FF4C1B", "color": "white"},
    "header": {"background-color": "#FF4C1B"},
    "header-info": {"background-color": "#FF4C1B"},
    "header-text": {"color": "white"},
}

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

auto_url = "./images/AUTOML.png"

def home_page():
    st.image(auto_url, use_column_width=True)

def upload_page():
    st.subheader("TẢI LÊN DỮ LIỆU CỦA BẠN ")
    st.info("Trong phần này, người dùng có thể tải lên tập dữ liệu của mình để phân tích và xây dựng mô hình học máy tự động.")
    data = st.file_uploader("Vui lòng tải lên tập dữ liệu của bạn ở đây")
    if data:
        df = pd.read_csv(data, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)

def trang_phan_tich():
    st.subheader("Phân Tích Dữ Liệu")
    st.info("Trong phần này, ứng dụng thực hiện một phân tích dữ liệu tự động trên dữ liệu. Điều này giúp người dùng hiểu biết và nắm bắt dữ liệu của mình hơn")
    st.info("GHI CHÚ: Nếu không có dữ liệu được tải lên ở Trang Tải Lên Dữ Liệu, trang này sẽ hiển thị một thông báo lỗi.")
    bao_cao = ProfileReport(df)
    st_profile_report(bao_cao)


def trang_hoc():
    st.subheader("TÍNH TOÁN MÁY HỌC TỰ ĐỘNG")
    st.info("Trong phần này, ứng dụng xây dựng và huấn luyện các mô hình máy học khác nhau với dữ liệu huấn luyện. Người dùng chỉ cần xác định và nhập biến mục tiêu")
    st.info("GHI CHÚ: Nếu không có dữ liệu được tải lên ở Trang Tải Lên Dữ Liệu, trang này sẽ hiển thị một thông báo lỗi.")
    muc_tieu = st.selectbox("Vui lòng chọn đặc trưng mục tiêu của bạn", df.columns)
    if st.button("Huấn luyện mô hình"):
        setup(df, target=muc_tieu)
        setup_df = pull()
        st.info("Đây là các cài đặt máy học tự động")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("Đây là hiệu suất của các mô hình máy học")
        st.dataframe(compare_df)
        best_model


def trang_tai_xuong():
    st.subheader("DOWNLOAD BEST MODEL")
    st.info("In this section, the app allows users to download and save their best performing models to their local computers")
    st.info("NOTE : If no data is uploaded at the Data Upload Page, this page will show an error message.")
    best_model = compare_models()
    compare_df = pull()
    st.info("This is the Performance of the machine learning models")
    st.dataframe(compare_df)
    best_model
    save_model(best_model, "best_model")
    with open("best_model.pkl", 'rb') as f:
        st.download_button("Please download best model", f, "auto_trained_model.pkl")


def trang_hoi_quy_tuyen_tinh():
    st.subheader("HỒI QUY TUYẾN TÍNH")
    st.info("Trong phần này, ứng dụng thực hiện hồi quy tuyến tính trên dữ liệu được chọn.")
    st.info("GHI CHÚ: Nếu không có dữ liệu được tải lên ở Trang Tải Lên Dữ Liệu, trang này sẽ hiển thị một thông báo lỗi.")
    
    if df is not None:
        # Chọn biến độc lập (X) và biến mục tiêu (Y)
        options = [''] + df.columns.tolist()  # Thêm một lựa chọn trống vào danh sách các biến
        X = st.selectbox("Chọn biến độc lập (X)", options)
        Y = st.selectbox("Chọn biến mục tiêu (Y)", options)
        
        # Kiểm tra xem có biến nào được chọn không
        if X != '' and Y != '':
            # Xử lý dữ liệu đầu vào nếu cần
            # Ví dụ: df = df.dropna() để loại bỏ các hàng có giá trị thiếu
            
            # Huấn luyện mô hình hồi quy tuyến tính
            X_data = df[[X]]  # Chọn các biến độc lập
            y_data = df[Y]  # Chọn biến mục tiêu
            model = LinearRegression()
            model.fit(X_data, y_data)
            
            # Dự đoán
            y_pred = model.predict(X_data)
            
            # Hiển thị kết quả
            mse = mean_squared_error(y_data, y_pred)
            r_squared = r2_score(y_data, y_pred)
            st.write(f"Mean Squared Error: {mse}")
            st.write(f"R-squared: {r_squared}")
            
            # Hiển thị biểu đồ
            plt.figure(figsize=(10, 6))
            plt.scatter(X_data, y_data, color='blue', label='Actual')
            plt.plot(X_data, y_pred, color='red', linewidth=2, label='Predicted')
            plt.xlabel(X)
            plt.ylabel(Y)
            plt.title('Actual vs Predicted')
            plt.legend()
            st.pyplot(plt)  # Hiển thị biểu đồ trong ứng dụng



# Thiết lập thanh menu lựa chọn (sidebar)
with st.sidebar:
    st.image(auto_url, use_column_width=True)
    st.info("Ứng dụng này cho phép người dùng xây dựng và tải xuống một mô hình học máy tự động bằng cách sử dụng streamlit, pandas profiling và pycaret")
    selected = option_menu(
        menu_title=None,
        options=["Trang Chủ", "Trang Tải Lên Dữ Liệu", "Trang Phân Tích Dữ Liệu", "Trang Máy Học Tự Động", "Trang Tải Xuống Mô Hình",  "Hồi Quy Tuyến Tính"],
        icons=["home", "cloud-upload", "clipboard-data", "cpu", "download",  "calculator"],
        styles=css_style
   )
    
if selected == "Trang Chủ":
    home_page()

elif selected == "Trang Tải Lên Dữ Liệu":
    upload_page()

elif selected == "Trang Phân Tích Dữ Liệu":
    trang_phan_tich()

elif selected == "Trang Máy Học Tự Động":
    trang_hoc()

elif selected == "Trang Tải Xuống Mô Hình":
    trang_tai_xuong(df)

elif selected == "Trang Phát Triển":
    trang_phat_trien()
elif selected == "Hồi Quy Tuyến Tính":
    trang_hoi_quy_tuyen_tinh()