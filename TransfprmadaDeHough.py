from matplotlib import pyplot as plt
from machinevisiontoolbox import Image

# Leer la imagen en modo monocromático
im = Image.Read("C:\\Users\\Ricardo\\Downloads\\BasesDeDatos\\galleta1.jpg", mono=True)

# Aplicar el detector de bordes Canny
edges = im.canny()

# Calcular la Transformada de Hough sobre los bordes
h = edges.Hough()

# Detectar líneas utilizando la Transformada de Hough Probabilística
lines = h.lines_p(100, minlinelength=200, maxlinegap=5, seed=0)

# Visualizar la imagen original con las líneas detectadas
fig, ax = plt.subplots()
im.disp(ax=ax, darken=True)  # Mostrar la imagen original atenuada
h.plot_lines(lines, 'r--')  # Trazar líneas en rojo punteado
plt.title("Líneas detectadas con la Transformada de Hough")
plt.show()
