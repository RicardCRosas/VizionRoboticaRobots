from machinevisiontoolbox import Image, Camera
import matplotlib.pyplot as plt


k_values = [2, 3, 4, 5]
color_img = Image.Read('C:\\Users\\Ricardo\\Downloads\\BasesDeDatos\\galleta1.jpg')

# Aplicar k-means clustering con diferentes valores de k
for k in k_values:
    clusters = color_img.colorkmeans(k)
    clusters.disp(title=f'Clasificación de color con k={k}')
    plt.show()

for init_type in ['random', 'spread']:
    clusters = color_img.colorkmeans(3, init=init_type)
    clusters.disp(title=f'k-means con inicialización {init_type}')
    plt.show()

# Suponiendo que el color objetivo tiene un tono específico
target_color = [255, 255, 0]  # Ejemplo para amarillo
cluster_means = clusters.mean_colors()
target_cluster = min(range(len(cluster_means)), key=lambda i: np.linalg.norm(cluster_means[i] - target_color))
print(f'El clúster más cercano al color objetivo es el número {target_cluster}')

# Inicializar la cámara
cam = Camera(device=0)

while True:
    frame = cam.snapshot()

    # Aplicar clasificación de color en tiempo real
    clusters = frame.colorkmeans(3)
    clusters.disp(title='Clasificación en tiempo real')
    plt.pause(0.1)

    # Romper el bucle al cerrar la ventana
    if not plt.fignum_exists(plt.gcf().number):
        break
cam.release()