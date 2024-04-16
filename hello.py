import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_option_menu import option_menu
from streamlit_pandas_profiling import st_profile_report
import os
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



st.title("Machine Learning")
st.write("Trước tiên hãy up file lên")
st.write("Chọn một tùy chọn từ thanh bên trái")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Khai báo biến df
df = None
css_style = {
    "icon": {"color": "black"},
    "icon_selected": {"color": "white"},
    "nav-link": {"--hover-color": "grey"},
    "nav-link-selected": {"background-color": "#FF4C1B", "color": "white","font-weight": "400"},
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

def trang_phan_tich(df):
    st.subheader("Phân Tích Dữ Liệu")
    st.info("Trong phần này, ứng dụng thực hiện một phân tích dữ liệu tự động trên dữ liệu. Điều này giúp người dùng hiểu biết và nắm bắt dữ liệu của mình hơn")
    st.info("GHI CHÚ: Nếu không có dữ liệu được tải lên ở Trang Tải Lên Dữ Liệu, trang này sẽ hiển thị một thông báo lỗi.")
    
    st.write("Thông tin cơ bản về dữ liệu:")
    st.write(df.info())

    st.write("Mô tả thống kê cơ bản:")
    st.write(df.describe())
    
    st.write("Mối quan hệ giữa các biến:")
    sns.pairplot(df)
    st.pyplot()
    
    # Phân tích phân phối của các biến
    st.write("Phân phối của các biến:")
    for column in df.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[column], kde=True)
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        st.pyplot()
   # Tạo báo cáo phân tích dữ liệu
    profile = ProfileReport(df, explorative=True)
    st_profile_report(profile)

def trang_hoi_quy_tuyen_tinh():
    st.subheader("HỒI QUY TUYẾN TÍNH")
    st.info("Trong phần này, ứng dụng thực hiện hồi quy tuyến tính trên dữ liệu được chọn.")
    st.info("GHI CHÚ: Nếu không có dữ liệu được tải lên ở Trang Tải Lên Dữ Liệu, trang này sẽ hiển thị một thông báo lỗi.")
    
    if df is not None:
        options = [''] + df.columns.tolist()  
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
            st.pyplot(plt) 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def trang_hoi_quy_logistic():
    st.subheader("HỒI QUY LOGISTIC")
    st.info("Trong phần này, ứng dụng thực hiện hồi quy logistic trên dữ liệu được chọn.")
    st.info("GHI CHÚ: Nếu không có dữ liệu được tải lên ở Trang Tải Lên Dữ Liệu, trang này sẽ hiển thị một thông báo lỗi.")
    
    if df is not None:
        options = [''] + df.columns.tolist()
        X = st.selectbox("Chọn biến độc lập (X)", options)
        Y = st.selectbox("Chọn biến mục tiêu (Y)", options)
    
        if X != '' and Y != '':
            st.subheader("Đồ thị hàm sigmoid")
            x_values = np.linspace(-10, 10, 100)
            y_values = sigmoid(x_values)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x_values, y_values, color='blue')
            ax.axvline(0, color='black', linestyle='--')
            ax.axhline(0.5, color='red', linestyle='--')
            ax.set_xlabel('x')
            ax.set_ylabel('sigmoid(x)')
            ax.set_title('Sigmoid Function')
            st.pyplot(fig)

# Thiết lập thanh menu lựa chọn (sidebar)
with st.sidebar:
    st.image(auto_url, use_column_width=True)
    st.info("Abc")
    selected = option_menu(
        menu_title=None,
        options=["Trang Chủ", "Trang Tải Lên Dữ Liệu", "Trang Phân Tích Dữ Liệu", 
                 
                 "Hồi Quy Tuyến Tính", "Hồi Quy Logistic"],
        icons=["home", "cloud-upload", "clipboard-data", "cpu", "download",  "calculator", "chart-line"],
        styles=css_style
   )
    
if selected == "Trang Chủ":
    home_page()

elif selected == "Trang Tải Lên Dữ Liệu":
    upload_page()

elif selected == "Trang Phân Tích Dữ Liệu":
    trang_phan_tich(df)

elif selected == "Hồi Quy Tuyến Tính":
    trang_hoi_quy_tuyen_tinh()

elif selected == "Hồi Quy Logistic":
    trang_hoi_quy_logistic()