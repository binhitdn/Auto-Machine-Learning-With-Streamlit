import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np

def recommendation_system():
    st.subheader("Hệ Thống Đề Xuất")
    st.info("Trong phần này, bạn có thể sử dụng hệ thống đề xuất để tìm các mục tương tự dựa trên dữ liệu.")

    if 'df' in st.session_state:
        df = st.session_state.df

        # Chọn các biến cần thiết cho hệ thống đề xuất
        features = st.multiselect("Chọn các biến để sử dụng trong hệ thống đề xuất", df.columns)

        if features:
            data = df[features]

            # Xử lý giá trị thiếu
            if data.isnull().values.any():
                st.warning("Dữ liệu đầu vào chứa giá trị thiếu. Đang xử lý...")
                imputer = SimpleImputer(strategy='mean')
                data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

            # Chuẩn hóa dữ liệu để đưa về cùng phạm vi
            scaler = StandardScaler()
            data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

            # Huấn luyện mô hình KNN để tạo hệ thống đề xuất
            model = NearestNeighbors(metric='euclidean', algorithm='auto')
            model.fit(data)

            # Người dùng nhập vào dữ liệu cho việc dự đoán
            input_data = []
            for feature in features:
                value = st.number_input(f"Nhập giá trị cho '{feature}':")
                input_data.append(value)

            if st.button("Đề Xuất"):
                input_data = np.array(input_data).reshape(1, -1)
                input_data = pd.DataFrame(imputer.transform(input_data), columns=data.columns)
                input_data = pd.DataFrame(scaler.transform(input_data), columns=data.columns)
                distances, indices = model.kneighbors(input_data)

                st.subheader("Các mục được đề xuất:")
                for i in range(len(indices[0])):
                    st.write(f"- Mục {indices[0][i]}")

                st.write("Thông tin chi tiết về các mục được đề xuất:")
                st.write(df.iloc[indices[0]])

    else:
        st.error("Không có dữ liệu được tải lên. Vui lòng tải lên một tệp CSV trong trang Tải Lên Dữ Liệu.")
