import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from streamlit_pandas_profiling import st_profile_report
from sklearn.neighbors import KNeighborsClassifier
from ydata_profiling import ProfileReport



# Load data
@st.cache
def load_data():
    return pd.read_csv("StudentsPerformance.csv")

data = load_data()
st.set_option('deprecation.showPyplotGlobalUse', False)

# Function for data preprocessing
def preprocess_data(df):
    # Converting string to numeric data
    df = df.replace({'male': 1, 'female': 0})
    df = df.replace({'group A': 0, 'group B': 1, 'group C': 2, 'group D': 3, 'group E': 4})
    df = df.replace({'free/reduced': 0, 'standard': 1})
    df = df.replace({'none': 0, 'completed': 1})
    df = df.replace({'some college': 3, "associate's degree": 2, 'high school': 4, 
                     'some high school': 5, "bachelor's degree": 1, "master's degree": 0})
    df = df.drop('lunch', axis=1)
    df['average_score'] = (df['math score'] + df['reading score'] + df['writing score']) // 3
    return df

def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot()

data = preprocess_data(data)

# Sidebar menu
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Home", "Upload Data", "Data Analysis", "Linear Regression", "Logistic Regression", "KNN"])

if menu == "Home":
    st.title("Machine Learning")
    st.write("First, upload your data file or choose an option from the sidebar.")

elif menu == "Upload Data":
    st.subheader("Upload Your Data")
    st.info("In this section, you can upload your own dataset for analysis and machine learning.")
    uploaded_file = st.file_uploader("Please upload your dataset here", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.dataframe(data)

elif menu == "Data Analysis":
    st.subheader("Data Analysis")
    st.info("Trong phần này, ứng dụng thực hiện một phân tích dữ liệu tự động trên dữ liệu. Điều này giúp người dùng hiểu biết và nắm bắt dữ liệu của mình hơn")
    st.info("GHI CHÚ: Nếu không có dữ liệu được tải lên ở Trang Tải Lên Dữ Liệu, trang này sẽ hiển thị một thông báo lỗi.")
    st.text(data.info())
    st.write("Thông tin cơ bản về dữ liệu:")
    st.write(data.info())

    st.write("Mô tả thống kê cơ bản:")
    st.write(data.describe())
    
    st.write("Mối quan hệ giữa các biến:")
    # sns.pairplot(data)
    # st.pyplot()
    
    # Phân tích phân phối của các biến
    st.write("Phân phối của các biến:")
    for column in data.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(data[column], kde=True)
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        st.pyplot()
   # Tạo báo cáo phân tích dữ liệu
    profile = ProfileReport(data, title="Pandas Profiling Report")
    st_profile_report(profile)
    
elif menu == "Linear Regression":
    st.subheader("Linear Regression")
    st.info("In this section, linear regression is performed on the selected data.")
    if not data.empty:
        options = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        selected_features = st.multiselect("Select independent variables (X)", options)
        selected_target = st.selectbox("Select target variable (Y)", options)
        
        if selected_features and selected_target:
            X_data = data[selected_features]
            y_data = data[selected_target]
            
            # Standardize the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_data)
            
            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_data, test_size=0.3, random_state=42)
            
            # Train the model
            reg = LinearRegression()
            reg.fit(X_train, y_train)
            
            # Predictions
            y_pred = reg.predict(X_test)
            
            # Model evaluation
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            st.write("Mean Squared Error:", mse)
            st.write("R-squared:", r2)
            
            # Plotting
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred)
            ax.plot(y_test, y_test, color='red', linestyle='--')  # Add the line for perfect prediction
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Linear Regression - Actual vs Predicted")
            st.pyplot(fig)
            
            # Displaying predicted vs actual results
            results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
            st.write("Predicted vs Actual Results:")
            st.write(results_df)
        else:
            st.warning("Please select at least one independent variable and one target variable.")



elif menu == "KNN":
    st.subheader("K-Nearest Neighbors")
    st.info("In this section, K-Nearest Neighbors is performed on the selected data.")
    if not data.empty:
        options = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        selected_features = st.multiselect("Select independent variables (X)", options)
        selected_target = st.selectbox("Select target variable (Y)", options)
        
        if selected_features and selected_target:
            X_data = data[selected_features]
            y_data = data[selected_target]
            
            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)
            
            # Train the model
            model = KNeighborsClassifier(n_neighbors=5)
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Model evaluation
            accuracy = model.score(X_test, y_test)
            cm = confusion_matrix(y_test, y_pred)
            classification_rep = classification_report(y_test, y_pred)
            
            st.write("Accuracy:", accuracy)
            st.write("Confusion Matrix:", cm)
            st.write("Classification Report:", classification_rep)
            
            # Plot confusion matrix
            plot_confusion_matrix(cm, model.classes_)
        else:
            st.warning("Please select at least one independent variable and one target variable.")
        