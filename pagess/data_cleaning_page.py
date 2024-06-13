import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def phan_tich_va_lam_sach_du_lieu():
    st.subheader("Phân Tích và Làm Sạch Dữ Liệu")
    st.info("Trong phần này, người dùng có thể thực hiện các thao tác làm sạch và phân tích dữ liệu.")

    if 'df' in st.session_state:
        df = st.session_state.df

        st.write("### Xem dữ liệu gốc:")
        st.dataframe(df)

        missing_data = df.isnull().sum()
        st.write("### Missing values in each column:")
        st.write(missing_data)

        st.write("### Xử lý giá trị thiếu và làm sạch dữ liệu:")
        columns = df.columns.tolist()
        fill_methods = {}
        for column in columns:
            st.write(f"#### Cột: {column}")
            col_type = str(df[column].dtype)
            fill_option = st.selectbox(
                f"Chọn phương pháp điền giá trị thiếu cho {column}",
                ("Không điền", "Điền giá trị cụ thể", "Điền giá trị nhiều nhất", "Điền giá trị trung bình", "Điền giá trị trung vị", "Xóa hàng có giá trị thiếu", "Xóa cột có giá trị thiếu"),
                key=f"{column}_fill_option"
            )

            fill_value = None
            if fill_option == "Điền giá trị cụ thể":
                fill_value = st.text_input(f"Điền giá trị cho các ô trống trong {column}", key=f"{column}_fill_value")

            fill_methods[column] = (fill_option, fill_value)

        for column, (fill_option, fill_value) in fill_methods.items():
            if fill_option == "Điền giá trị cụ thể" and fill_value is not None:
                df[column].fillna(fill_value, inplace=True)
            elif fill_option == "Điền giá trị nhiều nhất":
                df[column].fillna(df[column].mode()[0], inplace=True)
            elif fill_option == "Điền giá trị trung bình" and df[column].dtype in ['int64', 'float64']:
                df[column].fillna(df[column].mean(), inplace=True)
            elif fill_option == "Điền giá trị trung vị" and df[column].dtype in ['int64', 'float64']:
                df[column].fillna(df[column].median(), inplace=True)
            elif fill_option == "Xóa hàng có giá trị thiếu":
                df.dropna(subset=[column], inplace=True)
            elif fill_option == "Xóa cột có giá trị thiếu":
                df.drop(columns=[column], inplace=True)

        st.write("Dữ liệu sau khi xử lý giá trị thiếu:")
        st.dataframe(df)

        st.write("### Chuẩn hóa dữ liệu:")
        columns_to_scale = st.multiselect("Chọn các cột để chuẩn hóa", df.select_dtypes(include=['int64', 'float64']).columns)
        if st.button("Chuẩn hóa"):
            scaler = StandardScaler()
            df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
            st.write("Dữ liệu sau khi chuẩn hóa:")
            st.dataframe(df)

            st.write("Phân phối sau khi chuẩn hóa:")
            for column in columns_to_scale:
                plt.figure(figsize=(8, 6))
                sns.histplot(df[column], kde=True)
                plt.title(f"Distribution of {column} (Standardized)")
                plt.xlabel(column)
                plt.ylabel("Frequency")
                st.pyplot()

        st.write("### Xóa cột:")
        columns_to_drop = st.multiselect("Chọn các cột để xóa", df.columns)
        if st.button("Xóa cột đã chọn"):
            df.drop(columns_to_drop, axis=1, inplace=True)
            st.write("Dữ liệu sau khi xóa các cột đã chọn:")
            st.dataframe(df)

        # st.write("### Kiểm tra và xử lý giá trị ngoại lệ:")
        # st.write("Chọn các cột có giá trị số để kiểm tra ngoại lệ:")
        # numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

        # for column in numeric_columns:
        #     st.write(f"#### Cột: {column}")
        #     from_value = st.number_input(f"Nhập giá trị bắt đầu của khoảng cho cột '{column}'", format="%.2f")
        #     to_value = st.number_input(f"Nhập giá trị kết thúc của khoảng cho cột '{column}'", format="%.2f")
        #     outlier_threshold = st.number_input(f"Nhập ngưỡng cho các giá trị ngoại lệ trong khoảng [{from_value}, {to_value}]", min_value=0.0, step=0.1, format="%.1f")

        #     if st.button(f"Xử lý giá trị ngoại lệ cho cột '{column}' trong khoảng [{from_value}, {to_value}]"):
        #         if outlier_threshold > 0:
        #             df.loc[(df[column] < from_value) | (df[column] > to_value), column] = df[column].mean()

        #         st.write("Dữ liệu sau khi xử lý giá trị ngoại lệ:")
        #         st.dataframe(df)

        st.write("### Thay đổi kiểu dữ liệu cho cột:")
        columns_to_change_type = st.multiselect("Chọn các cột để thay đổi kiểu dữ liệu", df.columns)
        new_data_types = {}
        for column in columns_to_change_type:
            st.write(f"#### Cột: {column}")
            current_type = df[column].dtype
            new_type = st.selectbox(
                f"Chọn kiểu dữ liệu mới cho cột '{column}'",
                ['int64', 'float64', 'object', 'bool'],
                index=['int64', 'float64', 'object', 'bool'].index(current_type)
            )
            new_data_types[column] = new_type

        for column, new_type in new_data_types.items():
            if new_type != df[column].dtype:
                try:
                    df[column] = df[column].astype(new_type)
                except ValueError as e:
                    st.warning(f"Không thể chuyển đổi kiểu dữ liệu cho cột '{column}': {str(e)}")

        st.write("Dữ liệu sau khi thay đổi kiểu dữ liệu:")
        st.dataframe(df)

        df.to_csv("sourcedata.csv", index=None)
        st.success("Dữ liệu đã được lưu thành công vào file 'sourcedata.csv'.")
    else:
        st.warning("Vui lòng tải lên dữ liệu trước khi làm sạch.")

# To use this function in a Streamlit app, you can call phan_tich_va_lam_sach_du_lieu() within your main Streamlit script.
