import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score

# Định nghĩa tên cột theo bộ dữ liệu gốc
column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal", "num"
]

# Đọc dữ liệu từ file
file_path = "../dataset/processed.cleveland.data"
df = pd.read_csv(file_path, header=None, names=column_names)

# Thay thế '?' bằng NaN
df.replace('?', np.nan, inplace=True)

# Chuyển đổi các cột số về kiểu float
numeric_columns = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca", "thal"]
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Chuyển cột num thành nhị phân (0: Không bệnh, 1: Có bệnh)
df["num"] = df["num"].apply(lambda x: 1 if int(x) > 0 else 0)

# Kiểm tra số lượng giá trị NaN trong từng cột
print("Số lượng giá trị NaN trong mỗi cột trước khi xử lý:\n", df.isnull().sum())

# Xử lý giá trị NaN: Điền bằng trung vị (median) thay vì mean để giảm ảnh hưởng của ngoại lai
df.fillna(df.median(numeric_only=True), inplace=True)

# Xác định giá trị ngoại lai bằng phương pháp IQR
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

outlier_counts = {col: len(detect_outliers_iqr(df, col)) for col in numeric_columns}
print("Số lượng giá trị ngoại lai trong từng cột:", outlier_counts)

# Chuẩn hóa dữ liệu bằng MinMaxScaler
scaler = MinMaxScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Phân tích tương quan
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Heatmap tương quan giữa các đặc trưng")
plt.show()

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X = df.drop(columns=["num"])
y = df["num"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Huấn luyện mô hình Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

# Đánh giá mô hình
print("Accuracy (Logistic Regression):", accuracy_score(y_test, y_pred))
print("Classification Report (Logistic Regression):\n", classification_report(y_test, y_pred))
print("Confusion Matrix (Logistic Regression):\n", confusion_matrix(y_test, y_pred))

# Huấn luyện mô hình Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Đánh giá mô hình Random Forest
print("Accuracy (Random Forest):", accuracy_score(y_test, y_pred_rf))
print("Classification Report (Random Forest):\n", classification_report(y_test, y_pred_rf))
print("Confusion Matrix (Random Forest):\n", confusion_matrix(y_test, y_pred_rf))

cm = confusion_matrix(y_test, y_pred_rf)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
print(f"Specificity: {specificity:.2f}")

# Tính ROC-AUC: cần xác suất dự đoán của lớp 1
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC: {roc_auc:.2f}")

# Vẽ biểu đồ ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", color="blue")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()