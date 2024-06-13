import streamlit as st
import pandas as pd

def upload_page():
    st.subheader("TẢI LÊN DỮ LIỆU CỦA BẠN ")
    st.info("Trong phần này, người dùng có thể tải lên tập dữ liệu của mình để phân tích và xây dựng mô hình học máy tự động.")
    data = st.file_uploader("Vui lòng tải lên tập dữ liệu của bạn ở đây")
    if data:
        df = pd.read_csv(data, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)
