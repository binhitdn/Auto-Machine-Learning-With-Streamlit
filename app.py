import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_option_menu import option_menu
from streamlit_pandas_profiling import st_profile_report
import os
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report,roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier


from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans


st.title("Machine Learning")
st.write("Trước tiên hãy up file lên")
st.write("Chọn một tùy chọn từ thanh bên trái")
st.set_option('deprecation.showPyplotGlobalUse', False)

css_style = {
    "icon": {"color": "black"},
    "icon_selected": {"color": "white"},
    "nav-link": {"--hover-color": "grey"},
    "nav-link-selected": {"background-color": "#FF4C1B", "color": "white", "font-weight": "400"},
}

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)
    st.session_state.df = df
    st.session_state.columns = df.columns.tolist()

auto_url = "./images/AUTOML.png"

def upload_page(df):
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


                
def hoi_quy(df):
    st.subheader("MÔ HÌNH DỰ ĐOÁN")
    st.info("Trong phần này, người dùng có thể xây dựng mô hình học máy để dự đoán giá trị của một biến phụ thuộc dựa trên các biến độc lập.")
    st.info("GHI CHÚ: Nếu không có dữ liệu được tải lên ở Trang Tải Lên Dữ Liệu, trang này sẽ hiển thị một thông báo lỗi.")

    if df is not None:
        regression_type = st.radio("Chọn loại Mô Hình Dự Đoán:", ("Linear Regression", "Logistic Regression", "KNN", "Decision Tree", "Random Forest", "SVM", "Kmeans"))
        Y = st.selectbox("Chọn biến phụ thuộc (Y)", df.columns)
        X = st.multiselect("Chọn biến độc lập (X)", df.columns)
        
        if X and Y:
            X_data = df[X] 
            y_data = df[Y]  
            
            if regression_type == "Linear Regression":
                model = LinearRegression()
                model.fit(X_data, y_data)

                y_pred = model.predict(X_data)

                mse = mean_squared_error(y_data, y_pred)
                r_squared = r2_score(y_data, y_pred)

                st.write(f"Mean Squared Error: {mse}")
                st.write(f"R-squared: {r_squared}")

                plt.figure(figsize=(10, 6))
                plt.scatter(y_data, y_pred, color='blue', label='Actual vs Predicted')
                plt.plot(y_data, y_data, color='red', label='Perfect prediction') 
                plt.xlabel('Actual')
                plt.ylabel('Predicted')
                plt.title('Linear Regression - Actual vs Predicted')
                plt.legend()
                st.pyplot(plt)

            elif regression_type == "Logistic Regression":
                model = LogisticRegression(C=0.1, penalty='l2')
                model.fit(X_data, y_data)

                y_pred = model.predict(X_data)

                accuracy = accuracy_score(y_data, y_pred)

                st.write(f"Accuracy: {accuracy}")

                cm = confusion_matrix(y_data, y_pred)
                st.write("Confusion Matrix:")
                st.write(cm)

                results_df = pd.DataFrame({'Actual': y_data, 'Predicted': y_pred})
                st.write("Predicted vs Actual:")
                st.write(results_df)

                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.title("Confusion Matrix")
                st.pyplot()

            elif regression_type == "KNN":
                model = KNeighborsClassifier()
                model.fit(X_data, y_data)

                y_pred = model.predict(X_data)

                accuracy = accuracy_score(y_data, y_pred)

                st.write(f"Accuracy: {accuracy}")

                cm = confusion_matrix(y_data, y_pred)
                st.write("Confusion Matrix:")
                st.write(cm)

                results_df = pd.DataFrame({'Actual': y_data, 'Predicted': y_pred})
                st.write("Predicted vs Actual:")
                st.write(results_df)

                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.title("Confusion Matrix")
                st.pyplot()

            elif regression_type == "Decision Tree":
                max_depth = st.slider("Chọn độ sâu tối đa của cây", 1, 20, 5)
                min_samples_split = st.slider("Số lượng mẫu tối thiểu để tách nút", 2, 20, 2)
                min_samples_leaf = st.slider("Số lượng mẫu tối thiểu tại một lá", 1, 20, 1)
                
                if st.button("Dự đoán"):
                    X = df[X]
                    y = df[Y]
    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
                    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
                    model.fit(X_train, y_train)
    
                    y_pred = model.predict(X_test)
    
                    st.subheader("Kết quả dự đoán")
                    result_df = pd.DataFrame({"Thực tế": y_test, "Dự đoán": y_pred})
                    st.write(result_df)
    
                  
                    plt.figure(figsize=(20, 10))
                    plot_tree(model, filled=True, feature_names=X.columns.tolist(), class_names=y.unique().tolist())
                    st.pyplot(plt)
                    
            elif regression_type == "Random Forest":
                n_estimators = st.slider("Chọn số lượng cây (n_estimators)", 10, 200, 100)
                max_depth = st.slider("Chọn độ sâu tối đa của cây", 1, 20, 5)
                min_samples_split = st.slider("Số lượng mẫu tối thiểu để tách nút", 2, 20, 2)
                min_samples_leaf = st.slider("Số lượng mẫu tối thiểu tại một lá", 1, 20, 1)

                if st.button("Dự đoán"):
                    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        random_state=0
                    )
                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_test)

                    st.subheader("Kết quả dự đoán")
                    result_df = pd.DataFrame({"Thực tế": y_test, "Dự đoán": y_pred})
                    st.write(result_df)

                  
                    cm = confusion_matrix(y_test, y_pred)
                    st.write("Confusion Matrix:")
                    st.write(cm)

                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
                    plt.xlabel("Predicted")
                    plt.ylabel("Actual")
                    plt.title("Confusion Matrix")
                    st.pyplot()

                  
                    st.write("Classification Report:")
                    st.text(str(classification_report(y_test, y_pred)))

            elif regression_type == "SVM":
                C = st.slider("Chọn tham số C", 0.01, 10.0, 1.0)
                kernel = st.selectbox("Chọn kernel", ["linear", "poly", "rbf", "sigmoid"])
                degree = st.slider("Chọn bậc đa thức (degree)", 1, 10, 3)
                gamma = st.selectbox("Chọn gamma", ["scale", "auto"])

                if st.button("Dự đoán"):
                    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

                    model = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)
                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_test)

                    st.subheader("Kết quả dự đoán")
                    result_df = pd.DataFrame({"Thực tế": y_test, "Dự đoán": y_pred})
                    st.write(result_df)

                 
                    cm = confusion_matrix(y_test, y_pred)
                    st.write("Confusion Matrix:")
                    st.write(cm)

                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
                    plt.xlabel("Predicted")
                    plt.ylabel("Actual")
                    plt.title("Confusion Matrix")
                    st.pyplot()

                 
                    st.write("Classification Report:")
                    st.text(str(classification_report(y_test, y_pred)))
                 
            elif regression_type == "Kmeans":
                k = st.slider("Chọn số lượng cụm (K)", 1, 10, 2)
                if st.button("Dự đoán"):
                    kmeans = KMeans(n_clusters=k)
                    kmeans.fit(X_data)
                    y_kmeans = kmeans.predict(X_data)
                    centers = kmeans.cluster_centers_
                    plt.scatter(X_data.iloc[:, 0], X_data.iloc[:, 1], c=y_kmeans, s=50, cmap='viridis')
                    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
                    plt.xlabel(X.columns[0])
                    plt.ylabel(X.columns[1])
                    plt.title('KMeans Clustering')
                    st.pyplot(plt)
  



def handle_missing_data(df):
    missing_cols = df.columns[df.isnull().any()].tolist()
    st.write("Cột chứa dữ liệu bị thiếu:")
    st.write(missing_cols)

    
    if st.button("Xóa dòng"):
        df.dropna(axis=0, inplace=True)
    
    if st.button("Điền giá trị trung bình"):
        for col in missing_cols:
            if df[col].dtype == "object":
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].mean(), inplace=True)
    
    st.write("Dữ liệu sau khi xử lý dữ liệu bị thiếu:")
    st.write(df)

def handle_duplicates(df):
    duplicate_rows = df[df.duplicated()]
    st.write("Số lượng dòng trùng lặp:", duplicate_rows.shape[0])

    if duplicate_rows.shape[0] > 0:
        st.write("Dòng trùng lặp:")
        st.write(duplicate_rows)

    if st.button("Xóa dòng trùng lặp"):
        df.drop_duplicates(inplace=True)
    
    st.write("Dữ liệu sau khi xử lý dữ liệu trùng lặp:")
    st.write(df)

def convert_data_type(df):
    columns = df.columns.tolist()
    selected_columns = st.multiselect("Chọn cột để đổi kiểu dữ liệu:", columns)

    if selected_columns:
        for col in selected_columns:
            current_dtype = df[col].dtype
            st.write(f"Kiểu dữ liệu hiện tại của cột '{col}': {current_dtype}")

            new_dtype = st.selectbox(f"Chọn kiểu dữ liệu mới cho cột '{col}':", ["int", "float", "str", "datetime"])

            if new_dtype == "int":
                df[col] = df[col].astype(int)
            elif new_dtype == "float":
                df[col] = df[col].astype(float)
            elif new_dtype == "str":
                df[col] = df[col].astype(str)
            elif new_dtype == "datetime":
                df[col] = pd.to_datetime(df[col])

    st.write("Dữ liệu sau khi đổi kiểu dữ liệu:")
    st.write(df)

def rename_columns(df):
    old_columns = df.columns.tolist()
    new_columns = []

    for col in old_columns:
        new_name = st.text_input(f"Nhập tên mới cho cột '{col}':", value=col)
        new_columns.append(new_name)

    df.columns = new_columns

    st.write("Dữ liệu sau khi đổi tên các cột:")
    st.write(df)

def save_data(df):
    file_format = st.selectbox("Chọn định dạng tệp:", ["csv", "xlsx"])
    file_path = st.text_input("Nhập đường dẫn lưu trữ (để trống để lưu tại thư mục hiện tại):", value="")

    if file_path:
        file_name = st.text_input("Nhập tên tệp:", value=f"data_cleaned.{file_format}")
        save_path = f"{file_path}/{file_name}"
    else:
        file_name = st.text_input("Nhập tên tệp:", value=f"data_cleaned.{file_format}")
        save_path = file_name

    if st.button("Lưu dữ liệu"):
        if file_format == "csv":
            df.to_csv(save_path, index=False)
            # download 
            st.markdown(f'<a href="{save_path}" download="{file_name}">Click để tải tệp về</a>', unsafe_allow_html=True)
        elif file_format == "xlsx":
            df.to_excel(save_path, index=False)

        st.success(f"Đã lưu dữ liệu vào tệp {save_path}")

def phan_tich_va_lam_sach_du_lieu(df):
    st.subheader("Phân Tích và Làm Sạch Dữ Liệu")

    if df is not None:
        st.write("Dữ liệu gốc:")
        st.write(df)

        handle_missing_data(df.copy())
        handle_duplicates(df.copy())
        convert_data_type(df.copy())
        rename_columns(df.copy())
        save_data(df.copy())

    else:
        st.warning("Vui lòng tải lên dữ liệu trước khi thực hiện phân tích và làm sạch dữ liệu.")
        
# Thiết lập thanh menu lựa chọn (sidebar)
with st.sidebar:
    st.info("Abc")
    selected = option_menu(
        menu_title=None,
        options=["Trang Tải Lên Dữ Liệu", "Trang Phân Tích Dữ Liệu", "Mô Hình Dự Đoán", "Phân Tích và Làm Sạch Dữ Liệu"],
        icons=["cloud-upload", "clipboard-data", "cpu", "download", "calculator", "chart-line", "chart-bar"],
        styles=css_style
   )

if selected == "Trang Tải Lên Dữ Liệu":
    upload_page(df)

elif selected == "Trang Phân Tích Dữ Liệu":
    trang_phan_tich(df)

elif selected == "Mô Hình Dự Đoán":
    hoi_quy(df)

elif selected == "Phân Tích và Làm Sạch Dữ Liệu":
    phan_tich_va_lam_sach_du_lieu(df)
