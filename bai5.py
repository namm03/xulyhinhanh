import cv2
import numpy as np
from sklearn.cluster import KMeans
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Load ảnh vệ tinh
image = cv2.imread('anh/anhvetinh.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# K-means clustering
k = 3  # số lượng cụm
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(pixel_values)
segmented_image_kmeans = labels.reshape(image.shape[:2])

# Fuzzy C-means clustering
n_clusters = 3
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(pixel_values.T, n_clusters, 2, error=0.005, maxiter=1000)
cluster_membership = np.argmax(u, axis=0)
segmented_image_fcm = cluster_membership.reshape(image.shape[:2])

# Hiển thị kết quả
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title('Original Image')
plt.subplot(1, 3, 2)
plt.imshow(segmented_image_kmeans, cmap='viridis')
plt.title('K-means Clustering')
plt.subplot(1, 3, 3)
plt.imshow(segmented_image_fcm, cmap='viridis')
plt.title('Fuzzy C-means Clustering')
plt.show()
