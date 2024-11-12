from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dữ liệu IRIS
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hàm đánh giá mô hình
def evaluate_model(model, X_train, X_test, y_train, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, train_time

# KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn_accuracy, knn_time = evaluate_model(knn, X_train, X_test, y_train, y_test)

# SVM
svm = SVC(kernel='linear')
svm_accuracy, svm_time = evaluate_model(svm, X_train, X_test, y_train, y_test)

# ANN
ann = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])
ann.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

start_time = time.time()
ann.fit(X_train, y_train, epochs=50, batch_size=4, verbose=0)
ann_time = time.time() - start_time
ann_accuracy = ann.evaluate(X_test, y_test, verbose=0)[1]

# In kết quả
print(f"KNN Accuracy: {knn_accuracy}, Time: {knn_time}")
print(f"SVM Accuracy: {svm_accuracy}, Time: {svm_time}")
print(f"ANN Accuracy: {ann_accuracy}, Time: {ann_time}")
