import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar imagen en escala de grises
img = cv2.imread('C:\\Users\\Ricardo\\Downloads\\BasesDeDatos\\galleta1.jpg', cv2.IMREAD_GRAYSCALE)

# a) Harris Corner Detector con parámetros variables
def harris_detector(img, k=0.04, sigma=1):
    img_float = np.float32(img)
    dst = cv2.cornerHarris(img_float, 2, 3, k)
    dst = cv2.dilate(dst, None)
    img_harris = img.copy()
    img_harris[dst > 0.01 * dst.max()] = 255
    return img_harris

harris_img = harris_detector(img, k=0.04, sigma=1)

# b) Comparación entre detectores de esquinas (Harris, Shi-Tomasi y Noble)
def compare_corner_detectors(img):
    harris = cv2.cornerHarris(np.float32(img), 2, 3, 0.04)
    harris_img = img.copy()
    harris_img[harris > 0.01 * harris.max()] = 255

    shi_tomasi = cv2.goodFeaturesToTrack(img, 100, 0.01, 10)
    shi_tomasi = np.intp(shi_tomasi)
    shi_img = img.copy()
    for i in shi_tomasi:
        x, y = i.ravel()
        cv2.circle(shi_img, (x, y), 3, 255, -1)

    plt.subplot(1, 2, 1)
    plt.imshow(harris_img, cmap='gray')
    plt.title('Harris Corners')

    plt.subplot(1, 2, 2)
    plt.imshow(shi_img, cmap='gray')
    plt.title('Shi-Tomasi Corners')
    plt.show()

compare_corner_detectors(img)

# c) Implementación del detector de Moravec
def moravec_detector(img, window_size=3, threshold=100):
    h, w = img.shape
    output = np.zeros((h, w), np.float32)
    offset = window_size // 2

    for y in range(offset, h - offset):
        for x in range(offset, w - offset):
            roi = img[y - offset:y + offset + 1, x - offset:x + offset + 1]
            min_eigen = float('inf')
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                # Comprobar límites de la imagen antes de realizar el cálculo
                if (0 <= y + dy - offset < h - offset and 0 <= x + dx - offset < w - offset):
                    shifted_roi = img[y + dy - offset:y + dy + offset + 1, x + dx - offset:x + dx + offset + 1]
                    if roi.shape == shifted_roi.shape:  # Verificar que ambas tengan el mismo tamaño
                        ssd = np.sum((roi - shifted_roi) ** 2)
                        min_eigen = min(min_eigen, ssd)
            if min_eigen > threshold:
                output[y, x] = min_eigen

    return output

moravec_img = moravec_detector(img, window_size=3, threshold=100)
plt.imshow(moravec_img, cmap='gray')
plt.title("Moravec Corner Detector")
plt.show()
