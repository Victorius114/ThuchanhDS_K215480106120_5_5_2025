'''Bài 1. Sử dụng dữ liệu từ nguồn: https://www.kaggle.com/competitions/titanic/data
Yêu cầu:
+ Đối với cột age, thay vì chỉ điền giá trị trung bình hoặc trung vị, xây dựng một mô hình hồi quy tuyến tính
(Linear Regression) để dự đoán giá trị thiếu dựa trên các đặc trưng khác như pclass, sex, sibsp, và parch.
+ Đối với cột embarked, sử dụng thuật toán K-Nearest Neighbors (KNN) để điền giá trị thiếu dựa trên các đặc trưng liên quan.
+ Xử lý cột cabin (thường có nhiều giá trị thiếu): trích xuất thông tin từ ký tự đầu tiên của cabin (ví dụ: 'C', 'B') để tạo cột mới deck,
sau đó mã hóa cột này.
+ Tạo cột family_size như bài gốc (tổng sibsp + parch + 1).
+ Tạo cột is_alone (1 nếu hành khách đi một mình, 0 nếu đi cùng gia đình).
+ Trích xuất danh xưng (title) từ cột name (ví dụ: Mr, Mrs, Miss) và tạo cột title. Gộp các danh xưng hiếm (như Dr, Rev) thành nhóm "Rare".
+ Tạo cột fare_per_person bằng cách chia fare cho family_size để thể hiện giá vé trung bình mỗi người.
+ Vẽ biểu đồ boxplot để kiểm tra phân phối của fare_per_person theo pclass và survived.
+ Sử dụng kiểm định thống kê (ví dụ: t-test hoặc ANOVA) để kiểm tra xem fare_per_person có khác biệt đáng kể giữa các nhóm survived (sống sót hay không).
+ Xây dựng một pipeline xử lý dữ liệu (sử dụng sklearn.pipeline) để tự động hóa quá trình làm sạch, mã hóa, và chuẩn hóa dữ liệu.
+ Huấn luyện mô hình Random Forest để dự đoán survived dựa trên các đặc trưng đã xử lý và tạo mới.
+ Sử dụng Grid Search hoặc Random Search để tối ưu hóa siêu tham số của mô hình (ví dụ: n_estimators, max_depth).
+ Đánh giá mô hình bằng cross-validation (5-fold) và báo cáo các chỉ số: accuracy, precision, recall, F1-score, và ROC-AUC.
+ Tạo biểu đồ feature importance để xác định các đặc trưng quan trọng nhất ảnh hưởng đến khả năng sống sót.'''


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder

train_data = pd.read_csv(r'bai1/train.csv')
test_data = pd.read_csv(r'bai1/test.csv')

# 1. Đối với cột age, thay vì chỉ điền giá trị trung bình hoặc trung vị, xây dựng một mô hình hồi quy tuyến tính
# (Linear Regression) để dự đoán giá trị thiếu dựa trên các đặc trưng khác như pclass, sex, sibsp, và parch.
features = ['Age', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
data = train_data[features].copy()

data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

known_age = data[data['Age'].notna()]
unknown_age = data[data['Age'].isna()]

X = known_age.drop('Age', axis = 1)
y = known_age['Age']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
print(f'Model score (R^2): {model.score(X_test, y_test): .2f}')

if not unknown_age.empty:
    X_missing = unknown_age.drop('Age', axis = 1)
    predicted_age = model.predict(X_missing)
    train_data.loc[train_data['Age'].isna(), 'Age'] = predicted_age

print(f"Số giá trị còn thiếu trong Age: {train_data['Age'].isna().sum}")

#2. Đối với cột embarked, sử dụng thuật toán K-Nearest Neighbors (KNN) để điền giá trị thiếu dựa trên các đặc trưng liên quan.
known_embarked = data[data['Embarked'].notna()]
unknown_embarked = data[data['Embarked'].isna()]
X_train_embarked = known_embarked.drop('Embarked', axis = 1)
y_train_embarked = known_embarked['Embarked_encoded']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_embarked)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train_embarked)

if not unknown_embarked.empty:
    X_test = unknown_embarked.drop(['Embarked', 'Embarked_encoded'], axis = 10)