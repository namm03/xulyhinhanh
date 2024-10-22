import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('anh/bai1.jpg', cv2.IMREAD_GRAYSCALE)  # Đọc ảnh dưới dạng grayscale (ảnh xám)

# 1. Negative image
negative_image = 255 - image  # Ảnh âm bản (Negative image)

# 2. Tăng cường độ tương phản (using contrast stretching)
def increase_contrast(image, alpha=1.5, beta=0):
    # alpha: hệ số điều chỉnh độ tương phản (>1 để tăng độ tương phản)
    # beta: điều chỉnh độ sáng (brightness)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

contrast_image = increase_contrast(image)  # Tăng độ tương phản ảnh

# 3. Biến đổi log (Logarithmic transformation)
c = 255 / np.log(1 + np.max(image))  # Hệ số c để scale kết quả vào khoảng 0-255
log_image = c * np.log1p(image.astype(np.float32))  # Áp dụng phép biến đổi log
log_image = np.uint8(log_image)  # Chuyển đổi kết quả trở lại dạng uint8

# 4. Cân bằng histogram (Histogram equalization)
equalized_image = cv2.equalizeHist(image)  # Thực hiện cân bằng histogram

# Hiển thị các ảnh đã xử lý
images = [image, negative_image, contrast_image, log_image, equalized_image]
titles = ['Ảnh gốc', 'Ảnh âm bản', 'Ảnh tăng độ tương phản', 'Biến đổi log', 'Cân bằng Histogram']

plt.figure(figsize=(10, 8))
for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])  # Hiển thị tiêu đề cho mỗi ảnh
    plt.axis('off')  # Ẩn các trục tọa độ

plt.tight_layout()
plt.show()
