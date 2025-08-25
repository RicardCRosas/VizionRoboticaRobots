from machinevisiontoolbox import Image
import matplotlib.pyplot as plt

# Cargar imágenes
castle_img = Image.Read('C:\\Users\\Ricardo\\Downloads\\BasesDeDatos\\galleta1.jpg')
castle2_img = Image.Read('C:\\Users\\Ricardo\\Downloads\\BasesDeDatos\\galleta2.jpg')

# Aplicar umbralización interactiva
bw_castle = castle_img.threshold_interactive()
bw_castle2 = castle2_img.threshold_interactive()

# Mostrar resultados
plt.subplot(1, 2, 1)
bw_castle.disp(title='Umbralización interactiva en galleta1.jpg')

plt.subplot(1, 2, 2)
bw_castle2.disp(title='Umbralización interactiva en galleta2.jpg')
plt.show()

# Valores de k y tamaño de ventana para Niblack
k_values = [-0.2, 0, 0.2]
window_sizes = [15, 31, 45]

# Aplicar Niblack con diferentes valores de k y tamaño de ventana
for k in k_values:
    for window in window_sizes:
        bw_niblack = castle2_img.niblack(k=k, window=window)
        bw_niblack.disp(title=f'Niblack k={k}, window={window}')
        plt.show()

# Aplicar MSER
mser_regions = castle2_img.imser(delta=5, min_area=0.0001)

# Aplicar blobs para encontrar regiones de interés
blobs = mser_regions.blobs()

# Filtrar blobs basados en el tamaño de las cajas contenedoras
letter_blobs = [blob for blob in blobs if 10 < blob.width < 50 and 10 < blob.height < 50]

# Mostrar blobs que corresponden a letras
castle2_img.disp()
for blob in letter_blobs:
    blob.plot_box(color='green')
plt.show()

# MSER con diferentes valores de delta y min_area
delta_values = [5, 10, 15]
min_area_values = [0.0001, 0.0005, 0.001]

for delta in delta_values:
    for min_area in min_area_values:
        regions = castle2_img.imser(delta=delta, min_area=min_area)
        regions.disp(title=f'MSER Delta={delta}, MinArea={min_area}')
        plt.show()

# Aplicar Graph Cut
segmented_img = castle2_img.igraphcut(sigma=0.5, lambda_=5)

# Mostrar resultado
segmented_img.disp(title='Segmentación con Graph Cut en galleta2.jpg')
plt.show()
