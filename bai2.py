import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh gốc
image = cv2.imread('anh/bai1.jpg', cv2.IMREAD_GRAYSCALE) # Đọc ảnh dưới dạng grayscale (ảnh xám)

# 1. Dò biên với toán tử Sobel
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Gradient theo trục x
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Gradient theo trục y
sobel_combined = cv2.magnitude(sobel_x, sobel_y)  # Tổng hợp gradient theo cả hai hướng

# 2. Dò biên với toán tử Laplace Gaussian
log_kernel = np.array([[0, 0, -1, 0, 0],
                       [0, -1, -2, -1, 0],
                       [-1, -2, 16, -2, -1],
                       [0, -1, -2, -1, 0],
                       [0, 0, -1, 0, 0]])

log_image = cv2.filter2D(image, -1, log_kernel)

# Hiển thị các kết quả
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray'), plt.title('Ảnh xám')
plt.subplot(1, 3, 2), plt.imshow(sobel_combined, cmap='gray'), plt.title('Biên với Sobel')
plt.subplot(1, 3, 3), plt.imshow(log_image, cmap='gray'), plt.title('Biên với LoG')
plt.tight_layout()
plt.show()
