import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import plotly.express as px
import plotly.figure_factory as ff

def trang_phan_tich():
    st.subheader("Phân Tích Dữ Liệu")
    st.info("Trong phần này, ứng dụng thực hiện phân tích và hiển thị dữ liệu một cách trực quan.")

    if 'df' in st.session_state:
        df = st.session_state.df

        st.write("### Thông tin cơ bản về dữ liệu")
        st.write(df.head())

        st.write("### Mô tả thống kê cơ bản")
        st.write(df.describe())

        # Lọc các cột chỉ có dữ liệu số
        numeric_cols = df.select_dtypes(include='number').columns
        df_numeric = df[numeric_cols]

        if not df_numeric.empty:
            st.write("### Phân tích tương quan giữa các biến số")
            corr = df_numeric.corr()

            fig = px.imshow(corr, 
                            labels=dict(x="Biến số", y="Biến số", color="Hệ số tương quan"),
                            x=corr.index, 
                            y=corr.columns,
                            color_continuous_scale=px.colors.diverging.RdYlBu)
            fig.update_layout(title="Heatmap of Correlation Matrix")
            st.plotly_chart(fig)

            st.write("### Mối quan hệ giữa các biến số")
            st.write("Chọn các biến để hiển thị cặp đồ thị:")
            x_var = st.selectbox('Chọn biến X', numeric_cols)
            y_var = st.selectbox('Chọn biến Y', numeric_cols)
            if x_var and y_var:
                fig_scatter = px.scatter(df, x=x_var, y=y_var, title=f'Scatter plot between {x_var} and {y_var}')
                st.plotly_chart(fig_scatter)

            st.write("### Phân phối của các biến số")
            for column in numeric_cols:
                fig_dist = ff.create_distplot([df[column].dropna()], [column], show_hist=False, show_rug=False)
                fig_dist.update_layout(title_text=f'Phân phối của {column}')
                st.plotly_chart(fig_dist)
        else:
            st.warning("Không có biến số nào trong dữ liệu để phân tích tương quan và hiển thị.")

        st.write("### Phân tích thông tin bằng pandas profiling")
        profile = ProfileReport(df, explorative=True)
        st_profile_report(profile)
    else:
        st.warning("Vui lòng tải lên dữ liệu trước khi phân tích.")

# To use this function in a Streamlit app, you can call trang_phan_tich() within your main Streamlit script.
