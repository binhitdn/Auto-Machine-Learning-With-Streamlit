from sklearn.linear_model import LinearRegression
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

if 'file' not in st.session_state:
    st.session_state.file = None
if 'columns' not in st.session_state:
    st.session_state.columns = []
if 'df' not in st.session_state:
    st.session_state.df = None

# Page layout
st.set_page_config(page_title="Data Analysis App", layout="wide")


st.title("Data Analysis App")

select_file = st.sidebar.file_uploader("SELECT FILE")
if select_file is not None:
    st.session_state.file = select_file.getvalue()

button = st.sidebar.button("Đọc file")

if button:
    if st.session_state.file is not None:
        if select_file.name.endswith('.csv'):
            df = pd.read_csv(select_file)
            st.session_state.df = df  
            st.session_state.columns = df.columns.tolist()
        elif select_file.name.endswith('.xlsx'):
            df = pd.read_excel(select_file)
            st.session_state.df = df  
            st.session_state.columns = df.columns.tolist()
    else:
        st.warning('Vui lòng chọn dataset')

# Function to predict with selected algorithm
def predict_with_algorithm(df, selected_variables, target_variable, selected_algorithm):
    X = df[[var for var in selected_variables if var != target_variable]]
    y = df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # if selected_algorithm == "Logistic Regression":
    #     model = LogisticRegression()
    # elif selected_algorithm == "Linear Regression":
    #     model = LinearRegression()
    # elif selected_algorithm == "KNN":
    #     model = KNeighborsClassifier(n_neighbors=5)
    
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    
    # if selected_algorithm == "Logistic Regression":
    #     st.subheader("Confusion Matrix")
    #     conf_matrix = confusion_matrix(y_test, y_pred)
    #     st.write(conf_matrix)
        
    #     st.subheader("Accuracy")
    #     accuracy = accuracy_score(y_test, y_pred)
    #     st.write(accuracy)
        
    #     plt.figure(figsize=(6, 6))
    #     sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    #     st.pyplot()
        
    # elif selected_algorithm == "Linear Regression":
    #     mse = mean_squared_error(y_test, y_pred)
    #     r2 = r2_score(y_test, y_pred)
    #     st.write("Mean Squared Error:", mse)
    #     st.write("R-squared:", r2)
        
    # elif selected_algorithm == "KNN":
    #     accuracy = accuracy_score(y_test, y_pred)
    #     st.write("Accuracy:", accuracy)
    #     confusion_mat = confusion_matrix(y_test, y_pred)
    #     st.write("Confusion Matrix:")
    #     st.write(confusion_mat)
    if selected_algorithm == "Logistic Regression":
        model = LogisticRegression()
    elif selected_algorithm == "Linear Regression":
        model = LinearRegression()
    elif selected_algorithm == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
    elif selected_algorithm == "Decision Tree":
        model = DecisionTreeClassifier()
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if selected_algorithm == "Logistic Regression":
        st.subheader("Confusion Matrix")
        conf_matrix = confusion_matrix(y_test, y_pred)
        st.write(conf_matrix)
        
        st.subheader("Accuracy")
        accuracy = accuracy_score(y_test, y_pred)
        st.write(accuracy)
        
        plt.figure(figsize=(6, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        st.pyplot()
        
    # elif selected_algorithm == "Linear Regression":
    #     mse = mean_squared_error(y_test, y_pred)
    #     r2 = r2_score(y_test, y_pred)
    #     st.write("Mean Squared Error:", mse)
    #     st.write("R-squared:", r2)

    elif selected_algorithm == "Linear Regression":
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write("Mean Squared Error:", mse)
        st.write("R-squared:", r2)
        
        # st.subheader("Confusion Matrix")
        # conf_matrix = confusion_matrix(y_test, y_pred)
        # st.write(conf_matrix)
        
    # elif selected_algorithm == "KNN":
    #     accuracy = accuracy_score(y_test, y_pred)
    #     st.write("Accuracy:", accuracy)
    #     confusion_mat = confusion_matrix(y_test, y_pred)
    #     st.write("Confusion Matrix:")
    #     st.write(confusion_mat)
    elif selected_algorithm == "KNN":
        # accuracy = accuracy_score(y_test, y_pred)
        # st.write("Accuracy:", accuracy)
        st.subheader("Confusion Matrix")
        conf_matrix = confusion_matrix(y_test, y_pred)
        st.write(conf_matrix)
        
        st.subheader("Accuracy")
        accuracy = accuracy_score(y_test, y_pred)
        st.write(accuracy)
        
        plt.figure(figsize=(6, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        st.pyplot()
    elif selected_algorithm == "Decision Tree":
        st.subheader("Confusion Matrix")
        conf_matrix = confusion_matrix(y_test, y_pred)
        st.write(conf_matrix)
        
        st.subheader("Accuracy")
        accuracy = accuracy_score(y_test, y_pred)
        st.write(accuracy)
        
        plt.figure(figsize=(6, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        st.pyplot()

def clean_data():  
    st.header("CleanData")
    if st.sidebar.button("Display"):
        st.subheader("Display")
        st.write(st.session_state.df) 
    if st.sidebar.button("Display Null Counts"):
        st.subheader("Null Counts")
        null_counts = st.session_state.df.isnull().sum()
        st.write(null_counts)
    if st.sidebar.button("Delete Rows with Null"):
        st.session_state.df.dropna(inplace=True)
        st.success("All rows with null values have been deleted.")
    selected_column = st.sidebar.selectbox("Choose column", st.session_state.columns)
    if st.sidebar.button("Display Unique Values"):
        st.subheader(f"Unique values in column '{selected_column}'")
        unique_values = st.session_state.df[selected_column].unique()
        st.write(unique_values)
    if st.sidebar.button("Replace Null with Mode"):
        mode_value = st.session_state.df[selected_column].mode()[0]
        st.session_state.df[selected_column].fillna(mode_value, inplace=True)
        st.success(f"Successfully replaced null values in column '{selected_column}' with mode.")
    if st.sidebar.button("Replace Null with Mean"):
        mean_value = st.session_state.df[selected_column].mean()
        st.session_state.df[selected_column].fillna(mean_value, inplace=True)
        st.success(f"Successfully replaced null values in column '{selected_column}' with mean.")
    if st.sidebar.button("Delete Rows with Null of column"):
        st.session_state.df.dropna(subset=[selected_column], inplace=True)
        st.success(f"Rows with null values in column '{selected_column}' have been deleted.")
    if st.sidebar.button("Delete Column"):
        st.session_state.df.drop(labels=[selected_column],axis = 1, inplace=True)
        st.success(f"Column '{selected_column}' has been deleted.")
    replace_value = st.sidebar.text_input("Enter value to replace", "", key="replace_value_input")
    numeric_value = st.sidebar.text_input("Enter numeric value", "", key="numeric_value_input")
    if st.sidebar.button("Convert", key="convert_button"):
        if replace_value == "":
            st.warning("Please enter value to replace.")
            return
        if numeric_value == "":
            st.warning("Please enter numeric value.")
            return
        replace_dict = {replace_value: float(numeric_value)}
        st.session_state.df[selected_column].replace(replace_dict, inplace=True)
        st.success(f"Categorical data in column '{selected_column}' converted to numeric.")
    if st.sidebar.button("Histogram Chart"):
        numeric_columns = st.session_state.df.select_dtypes(include=['int', 'float']).columns
        fig, axes = plt.subplots(len(numeric_columns), 1, figsize=(8, len(numeric_columns) * 4))
        for i, col in enumerate(numeric_columns):
            st.session_state.df[col].hist(ax=axes[i])
            axes[i].set_title(col)
        st.pyplot(fig)
    
    if st.sidebar.button("Pie Chart of Column"):
        value_counts = st.session_state.df[selected_column].value_counts(normalize=True)  
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=140)
        ax.set_title(selected_column, fontsize=14)  # Điều chỉnh kích thước của tiêu đề
        ax.axis('equal')  # Đảm bảo biểu đồ tròn
        ax.legend(loc="right", fontsize=8)  # Thêm chú thích bên cạnh
        ax.set_ylabel("")  # Xóa nhãn trục y
        ax.tick_params(labelsize=8)  # Điều chỉnh cỡ chữ của tất cả trong biểu đồ
        st.pyplot(fig)
    if st.sidebar.button("Line Chart of Column"):
        value_counts = st.session_state.df[selected_column].value_counts(normalize=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        value_counts.plot(kind='line', ax=ax)
        ax.set_title(selected_column, fontsize=14)  # Điều chỉnh kích thước của tiêu đề
        ax.set_xlabel(selected_column)  # Đặt nhãn trục x là tên cột
        ax.set_ylabel("Percentage")  # Đặt nhãn trục y
        ax.tick_params(labelsize=8)  # Điều chỉnh cỡ chữ của tất cả trong biểu đồ
        st.pyplot(fig)
    if st.sidebar.button("Box plot of Column"):
        q1, q3 = np.percentile(st.session_state.df[selected_column], [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - (iqr)
        upper_bound = q3 + (iqr)
        clean_data = st.session_state.df[(st.session_state.df[selected_column] >= lower_bound) & (st.session_state.df[selected_column] <= upper_bound)]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.boxplot(clean_data[selected_column], vert=False)
        ax.set_title('Box plot')
        # Thêm chú thích
        ax.annotate(f'Min: {lower_bound:.2f}', xy=(lower_bound, 1), xytext=(lower_bound, 1.1), fontsize=10)
        ax.annotate(f'Q1: {q1:.2f}', xy=(q1, 1), xytext=(q1, 1.1), fontsize=10)

        ax.annotate(f'Q3: {q3:.2f}', xy=(q3, 1), xytext=(q3, 1.1), fontsize=10)

        ax.annotate(f'Max: {upper_bound:.2f}', xy=(upper_bound, 1), xytext=(upper_bound, 1.1), fontsize=10)

        st.pyplot(fig)
    selected_2_variables = st.sidebar.multiselect("Select variables(1 or 2)", st.session_state.columns)
    if st.sidebar.button("Bar Chart of 3 Column"):
        df_gr = st.session_state.df.groupby(selected_column).agg({selected_2_variables[0]:'count', selected_2_variables[1]:'mean'}).reset_index()
        df_gr = df_gr.sort_values(by=selected_2_variables[0])
        ax = df_gr.plot(kind='bar', x=selected_column)
        plt.grid()
        st.pyplot(ax.figure)
    if st.sidebar.button("Bar Chart of 2 Column"):
        df_gr = st.session_state.df.groupby(selected_column).agg({ selected_2_variables[0]:'mean'}).reset_index()
        df_gr = df_gr.sort_values(by=selected_2_variables[0])
        ax = df_gr.plot(kind='bar', x=selected_column, y = selected_2_variables[0])
        plt.grid()
        st.pyplot(ax.figure)
    if st.sidebar.button("Scatter Plot"):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(st.session_state.df[selected_column], st.session_state.df[selected_2_variables[0]], color='purple', alpha=0.5)
        ax.set_title('Scatter Plot')
        ax.set_xlabel(selected_column)
        ax.set_ylabel(selected_2_variables[0])
        ax.grid(True)
        st.pyplot(fig)
    if st.sidebar.button("Save dataframe"):
        # Chuyển DataFrame thành CSV
        csv = st.session_state.df.to_csv(index=False)
        # Tạo nút tải xuống cho CSV
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='dataframe_clean.csv',
            mime='text/csv',
        )


def training_data():
    st.header("Huấn luyện mô hình")  
    selected_column = st.sidebar.selectbox("Choose column", st.session_state.columns)
    selected_algorithm = st.sidebar.selectbox("Choose algorithm", ["Logistic Regression", "Linear Regression", "KNN", "Decision Tree"])            
    st.sidebar.subheader("Choose variables for prediction")
    selected_variables = st.sidebar.multiselect("Select variables", st.session_state.columns)

    if st.sidebar.button("Predict"):
        predict_with_algorithm(st.session_state.df, selected_variables, selected_column, selected_algorithm)      

def main():
    menu = ["Clean Data", "Training Data"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Clean Data":
        clean_data()
    elif choice == "Training Data":
        training_data()


if __name__ == "__main__":
    main()
