
import streamlit as st
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.naive_bayes import MultinomialNB 


def hoi_quy():
    st.subheader("MÔ HÌNH DỰ ĐOÁN")
    st.info("Trong phần này, người dùng có thể xây dựng mô hình học máy để dự đoán giá trị của một biến phụ thuộc dựa trên các biến độc lập.")
    st.info("GHI CHÚ: Nếu không có dữ liệu được tải lên ở Trang Tải Lên Dữ Liệu, trang này sẽ hiển thị một thông báo lỗi.")

    if 'df' in st.session_state:
        df = st.session_state.df
        regression_type = st.radio("Chọn loại Mô Hình Dự Đoán:", ("Linear Regression", "Logistic Regression", "KNN", "Decision Tree", "Random Forest", "SVM", "Kmeans", "Naive Bayes"))
        Y = st.selectbox("Chọn biến phụ thuộc (Y)", df.columns)
        X = st.multiselect("Chọn biến độc lập (X)", df.columns)

        if X and Y:
            X_data = df[X]
            y_data = df[Y]

            if regression_type == "Linear Regression":
                if st.button("Dự đoán"):
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
                if st.button("Dự đoán"):
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
                if st.button("Dự đoán"):
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
                    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

                    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
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

                    st.write("Decision Tree Visualization:")
                    fig, ax = plt.subplots(figsize=(12, 8))
                    plot_tree(model, feature_names=X, filled=True, ax=ax)
                    st.pyplot(fig)

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
                    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
                    plt.title("KMeans Clustering")
                    plt.xlabel(X[0])
                    plt.ylabel(X[1])
                    st.pyplot(plt)
                    st.write("Centers of clusters:")
                    st.write(centers)

            elif regression_type == "Naive Bayes":
                if st.button("Dự đoán"):
                    model = MultinomialNB()
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

                    st.write("Classification Report:")
                    st.text(str(classification_report(y_data, y_pred)))
    else:
        st.error("Không có dữ liệu được tải lên. Vui lòng tải lên một tệp CSV trong trang Tải Lên Dữ Liệu.")
