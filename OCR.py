from machinevisiontoolbox import ImageRegionFeatures, CentralCamera
from matplotlib import pyplot as plt
from machinevisiontoolbox import Image

# Leer imágenes en modo monocromático
view1 = Image.Read("C:\\Users\\Ricardo\\Downloads\\BasesDeDatos\\galleta1.jpg", mono=True)
view2 = Image.Read("C:\\Users\\Ricardo\\Downloads\\BasesDeDatos\\galleta2.jpg", mono=True)

# Extraer características SIFT de ambas imágenes
sf1 = view1.SIFT()
sf2 = view2.SIFT()

# Realizar coincidencias entre las características SIFT
matches = sf1.match(sf2)
print(matches)

# Mostrar una tabla con las primeras coincidencias
matches[1:5].table()

# Plotear una muestra de coincidencias en blanco
matches.subset(100).plot("w")
plt.title("Coincidencias (blanco)")
plt.show()

# Estimar la matriz fundamental usando RANSAC
F, resid = matches.estimate(CentralCamera.points2F, method="ransac", confidence=0.99, seed=0)
print("Matriz fundamental (F):\n", F)

# Apilar imágenes horizontalmente y mostrar con coincidencias
stacked_image = Image.Hstack((view1, view2))
stacked_image.disp(title="Imágenes apiladas con coincidencias")

# Coincidencias correctas en verde
matches.inliers.subset(100).plot("g", ax=plt.gca())
plt.title("Coincidencias correctas (verde)")
plt.show()

# Coincidencias incorrectas en rojo
matches.outliers.subset(100).plot("r", ax=plt.gca())
plt.title("Coincidencias incorrectas (rojo)")
plt.show()
