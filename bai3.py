import os
import cv2
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Đường dẫn tới folder chứa ảnh và file labels.csv
image_folder = r"E:\EAUT\nam4\ki1\xulyhinhanh\baitap\anh"
label_file_path = r"E:\EAUT\nam4\ki1\xulyhinhanh\baitap\labels/labels.csv"

# Đọc file nhãn
labels_df = pd.read_csv(label_file_path)


# Hàm để đọc ảnh và chuyển đổi thành vector
def load_images_and_labels(image_folder, labels_df, image_size=(30, 30)):
    images = []
    labels = []

    for _, row in labels_df.iterrows():
        image_path = os.path.join(image_folder, row['image_name'])
        label = row['label']

        # Đọc ảnh và resize
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.resize(image, image_size)  # Resize về kích thước cố định
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Chuyển sang ảnh xám
            images.append(image.flatten())  # Chuyển thành vector
            labels.append(label)

    return np.array(images), np.array(labels)


# Tải ảnh và nhãn
X, y = load_images_and_labels(image_folder, labels_df)

# Chia dữ liệu thành tập huấn luyện và kiểm thử
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Hàm để huấn luyện và đánh giá mô hình
def evaluate_model(model, X_train, X_test, y_train, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Dự đoán trên tập kiểm thử
    y_pred = model.predict(X_test)

    # Tính các độ đo
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=1)

    return training_time, accuracy, precision, recall


# Khởi tạo các mô hình
models = {
    'SVM': SVC(kernel='linear'),
    'KNN': KNeighborsClassifier(n_neighbors=1),  # Giảm n_neighbors xuống 1
    'Decision Tree': DecisionTreeClassifier(max_depth=5)
}

# Đánh giá từng mô hình
results = {}
for model_name, model in models.items():
    training_time, accuracy, precision, recall = evaluate_model(model, X_train, X_test, y_train, y_test)
    results[model_name] = {
        'Time': training_time,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall
    }

# In kết quả
for model_name, metrics in results.items():
    print(f"{model_name} results:")
    print(f" - Training Time: {metrics['Time']:.4f} seconds")
    print(f" - Accuracy: {metrics['Accuracy']:.4f}")
    print(f" - Precision: {metrics['Precision']:.4f}")
    print(f" - Recall: {metrics['Recall']:.4f}")
    print()
image_size = (300, 300)  # Điều chỉnh kích thước theo ý muốn

# Đọc file labels.csv
import pandas as pd
labels_df = pd.read_csv(label_file_path)

# Tạo dictionary để ánh xạ tên ảnh với nhãn
labels_dict = dict(zip(labels_df["image_name"], labels_df["label"]))

# Lấy danh sách các ảnh
image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

# Tính toán số hàng và cột cho lưới
num_images = len(image_files)
num_cols = 4  # Số cột, điều chỉnh nếu muốn
num_rows = (num_images + num_cols - 1) // num_cols  # Tính số hàng cần thiết

# Tạo khung hình trống
combined_image = np.zeros((image_size[1] * num_rows + 20 * num_rows, image_size[0] * num_cols, 3), dtype=np.uint8)

# Đọc và chèn từng ảnh vào khung hình lớn
for idx, image_name in enumerate(image_files):
    # Đọc ảnh và resize
    image_path = os.path.join(image_folder, image_name)
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.resize(image, image_size)

        # Lấy nhãn từ dictionary
        label = labels_dict.get(image_name, "unknown")

        # Vẽ nhãn trên đầu ảnh
        cv2.putText(image, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Xác định vị trí hàng và cột
        row = idx // num_cols
        col = idx % num_cols

        # Xác định vị trí chèn ảnh vào khung hình lớn
        y_start = row * (image_size[1] + 20)
        y_end = y_start + image_size[1]
        x_start = col * image_size[0]
        x_end = x_start + image_size[0]

        # Chèn ảnh vào vị trí xác định trong khung hình lớn
        combined_image[y_start:y_end, x_start:x_end] = image

# Hiển thị khung hình lớn chứa toàn bộ ảnh với nhãn
cv2.imshow("Combined Image with Labels", combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()