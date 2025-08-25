import cv2
from machinevisiontoolbox import Image
from matplotlib import pyplot as plt
import numpy as np

# Leer la imagen en escala de grises
object_img = Image.Read('C:\\Users\\Ricardo\\Downloads\\BasesDeDatos\\galleta1.jpg')
gray_img = object_img.image  # Obtener la imagen en escala de grises como numpy array

# Aplicar el detector de bordes Canny
edges = cv2.Canny(gray_img, 100, 200)

# Detectar contornos usando OpenCV
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Crear una copia de la imagen para mostrar los contornos
contour_img = np.zeros_like(gray_img)
cv2.drawContours(contour_img, contours, -1, 255, 1)  # Dibujar todos los contornos en blanco

# Mostrar la imagen original y los contornos
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(gray_img, cmap='gray')
plt.title('Imagen original')

plt.subplot(1, 2, 2)
plt.imshow(contour_img, cmap='gray')
plt.title('Contornos detectados')
plt.show()

# Detección de blobs con SimpleBlobDetector de OpenCV
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 50
params.filterByCircularity = True
params.minCircularity = 0.1
detector = cv2.SimpleBlobDetector_create(params)

# Detectar blobs en la imagen
keypoints = detector.detect(gray_img)

# Dibujar blobs en la imagen
blob_img = cv2.drawKeypoints(gray_img, keypoints, np.zeros_like(gray_img), (0, 255, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Mostrar la imagen con los blobs
plt.imshow(blob_img, cmap='gray')
plt.title('Detección de blobs')
plt.show()
