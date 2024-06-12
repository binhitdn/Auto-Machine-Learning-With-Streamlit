# def hoi_quy(df):
#     st.subheader("HỒI QUY")
#     st.info("Trong phần này, bạn có thể chọn biến độc lập và biến phụ thuộc trước khi chọn loại hồi quy.")
#     st.info("GHI CHÚ: Nếu không có dữ liệu được tải lên ở Trang Tải Lên Dữ Liệu, trang này sẽ hiển thị một thông báo lỗi.")
    
#     if df is not None:
#         options = [''] + df.columns.tolist()  
#         X = st.multiselect("Chọn biến độc lập (X)", options)
#         Y = st.selectbox("Chọn biến phụ thuộc (Y)", options)
        
#         # Kiểm tra xem có biến nào được chọn không
#         if X and Y:
#             X_data = df[X]  # Chọn các biến độc lập
#             y_data = df[Y]  # Chọn biến phụ thuộc
            
#             # Lựa chọn loại hồi quy
#             regression_type = st.radio("Chọn loại hồi quy:", ("Linear Regression", "Logistic Regression", "KNN", "Decision Tree","Random Forest","SVM"))
            
#             if regression_type == "Linear Regression":
#                 # Train the linear regression model
#                 model = LinearRegression()
#                 model.fit(X_data, y_data)
    
#                 # Make predictions
#                 y_pred = model.predict(X_data)
    
#                 # Model evaluation
#                 mse = mean_squared_error(y_data, y_pred)
#                 r_squared = r2_score(y_data, y_pred)
    
#                 # Display results
#                 st.write(f"Mean Squared Error: {mse}")
#                 st.write(f"R-squared: {r_squared}")
    
#                 # Plotting
#                 plt.figure(figsize=(10, 6))
#                 plt.scatter(y_data, y_pred, color='blue', label='Actual vs Predicted')
#                 plt.plot(y_data, y_data, color='red', label='Perfect prediction')  # Add the line for perfect prediction
#                 plt.xlabel('Actual')
#                 plt.ylabel('Predicted')
#                 plt.title('Linear Regression - Actual vs Predicted')
#                 plt.legend()
#                 st.pyplot(plt)

            
#             elif regression_type == "Logistic Regression":
#                 # Train the logistic regression model
#                 model = LogisticRegression(C=0.1, penalty='l2')
#                 model.fit(X_data, y_data)

#                 # Make predictions
#                 y_pred = model.predict(X_data)

#                 # Model evaluation
#                 accuracy = accuracy_score(y_data, y_pred)

#                 # Display results
#                 st.write(f"Accuracy: {accuracy}")

#                 # Plotting (if applicable)
#                 # Plot confusion matrix
#                 cm = confusion_matrix(y_data, y_pred)
#                 st.write("Confusion Matrix:")
#                 st.write(cm)

#                 # Display predicted and actual values in a table
#                 results_df = pd.DataFrame({'Actual': y_data, 'Predicted': y_pred})
#                 st.write("Predicted vs Actual:")
#                 st.write(results_df)

#                 # Plot confusion matrix as heatmap
#                 plt.figure(figsize=(8, 6))
#                 sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
#                 plt.xlabel("Predicted")
#                 plt.ylabel("Actual")
#                 plt.title("Confusion Matrix")
#                 st.pyplot()

#             elif regression_type == "KNN":
#                 # Train the KNN model
#                 model = KNeighborsClassifier()
#                 model.fit(X_data, y_data)
    
#                 # Make predictions
#                 y_pred = model.predict(X_data)
    
#                 # Model evaluation
#                 accuracy = accuracy_score(y_data, y_pred)
    
#                 # Display results
#                 st.write(f"Accuracy: {accuracy}")
    
#                 # Plotting (if applicable)
#                 # Plot confusion matrix
#                 cm = confusion_matrix(y_data, y_pred)
#                 st.write("Confusion Matrix:")
#                 st.write(cm)

#                 # Display predicted and actual values in a table
#                 results_df = pd.DataFrame({'Actual': y_data, 'Predicted': y_pred})
#                 st.write("Predicted vs Actual:")
#                 st.write(results_df)
    
#                 # Plot confusion matrix as heatmap
#                 plt.figure(figsize=(8, 6))
#                 sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
#                 plt.xlabel("Predicted")
#                 plt.ylabel("Actual")
#                 plt.title("Confusion Matrix")
#                 st.pyplot()

#             elif regression_type == "Decision Tree":
#                 dependent_var = st.selectbox("Chọn biến phụ thuộc (Chỉ phân loại)", df.columns)
#                 independent_vars = st.multiselect("Chọn biến độc lập", df.columns.drop(dependent_var))
#                 max_depth = st.slider("Chọn độ sâu tối đa của cây", 1, 20, 5)
#                 min_samples_split = st.slider("Số lượng mẫu tối thiểu để tách nút", 2, 20, 2)
#                 min_samples_leaf = st.slider("Số lượng mẫu tối thiểu tại một lá", 1, 20, 1)
                
#                 if st.button("Dự đoán"):
#                     X = df[independent_vars]
#                     y = df[dependent_var]
    
#                     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#                     model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
#                     model.fit(X_train, y_train)
    
#                     y_pred = model.predict(X_test)
    
#                     st.subheader("Kết quả dự đoán")
#                     result_df = pd.DataFrame({"Thực tế": y_test, "Dự đoán": y_pred})
#                     st.write(result_df)
    
#                     # In ra cây quyết định
#                     plt.figure(figsize=(20, 10))
#                     plot_tree(model, filled=True, feature_names=X.columns.tolist(), class_names=y.unique().tolist())
#                     st.pyplot(plt)
#             elif regression_type == "Random Forest":
#                 dependent_var = st.selectbox("Chọn biến phụ thuộc (Chỉ phân loại)", df.columns)
#                 independent_vars = st.multiselect("Chọn biến độc lập", df.columns.drop(dependent_var))
#                 n_estimators = st.slider("Chọn số lượng cây (n_estimators)", 10, 200, 100)
#                 max_depth = st.slider("Chọn độ sâu tối đa của cây", 1, 20, 5)
#                 min_samples_split = st.slider("Số lượng mẫu tối thiểu để tách nút", 2, 20, 2)
#                 min_samples_leaf = st.slider("Số lượng mẫu tối thiểu tại một lá", 1, 20, 1)

#                 if st.button("Dự đoán"):
#                     X = df[independent_vars]
#                     y = df[dependent_var]

#                     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#                     model = RandomForestClassifier(
#                         n_estimators=n_estimators,
#                         max_depth=max_depth,
#                         min_samples_split=min_samples_split,
#                         min_samples_leaf=min_samples_leaf,
#                         random_state=0
#                     )
#                     model.fit(X_train, y_train)

#                     y_pred = model.predict(X_test)

#                     st.subheader("Kết quả dự đoán")
#                     result_df = pd.DataFrame({"Thực tế": y_test, "Dự đoán": y_pred})
#                     st.write(result_df)

#                     # Confusion Matrix
#                     cm = confusion_matrix(y_test, y_pred)
#                     st.write("Confusion Matrix:")
#                     st.write(cm)

#                     plt.figure(figsize=(8, 6))
#                     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
#                     plt.xlabel("Predicted")
#                     plt.ylabel("Actual")
#                     plt.title("Confusion Matrix")
#                     st.pyplot()

#                     # Classification Report
#                     st.write("Classification Report:")
#                     st.text(str(classification_report(y_test, y_pred)))

#             elif regression_type == "SVM":
#                 dependent_var = st.selectbox("Chọn biến phụ thuộc (Chỉ phân loại)", df.columns)
#                 independent_vars = st.multiselect("Chọn biến độc lập", df.columns.drop(dependent_var))
#                 C = st.slider("Chọn tham số C", 0.01, 10.0, 1.0)
#                 kernel = st.selectbox("Chọn kernel", ["linear", "poly", "rbf", "sigmoid"])
#                 degree = st.slider("Chọn bậc đa thức (degree)", 1, 10, 3)
#                 gamma = st.selectbox("Chọn gamma", ["scale", "auto"])

#                 if st.button("Dự đoán"):
#                                         X = df[independent_vars]
#                                         y = df[dependent_var]

#                                         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#                                         model = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)
#                                         model.fit(X_train, y_train)

#                                         y_pred = model.predict(X_test)

#                                         st.subheader("Kết quả dự đoán")
#                                         result_df = pd.DataFrame({"Thực tế": y_test, "Dự đoán": y_pred})
#                                         st.write(result_df)

#                                         # Confusion Matrix
#                                         cm = confusion_matrix(y_test, y_pred)
#                                         st.write("Confusion Matrix:")
#                                         st.write(cm)

#                                         plt.figure(figsize=(8, 6))
#                                         sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
#                                         plt.xlabel("Predicted")
#                                         plt.ylabel("Actual")
#                                         plt.title("Confusion Matrix")
#                                         st.pyplot()

#                                         # Classification Report
#                                         st.write("Classification Report:")
#                                         st.text(str(classification_report(y_test, y_pred)))