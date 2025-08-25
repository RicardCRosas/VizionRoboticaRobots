import cv2
import numpy as np
from robodk.robolink import Robolink
from robolink import *  # API para interactuar con RoboDK
from robodk import *  # Funciones de robot

# Conexión a RoboDK
RDK = Robolink()
robot = RDK.Item('UR5')  # Cambia el nombre si el robot tiene otro en tu proyecto de RoboDK

# Configuración de los ángulos constantes para cada junta (basados en la captura de pantalla proporcionada)
theta2 = -90.40
theta3 = -82.52
theta4 = -97.08
theta5 = 90.00
theta6 = 0.00

# Configuración del color rojo en HSV
lower_red = np.array([150, 20, 0])
upper_red = np.array([180, 250, 250])


def detect_red_angle(frame):
    """
    Detecta el color rojo en el marco y calcula el ángulo respecto al eje X.
    """
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_red = cv2.inRange(frame_hsv, lower_red, upper_red)

    # Encuentra contornos del color rojo
    contours, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Seleccionar el contorno más grande (supone que es el objetivo)
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)

        if M['m00'] != 0:
            # Centroide del contorno
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])

            # Calcular el ángulo respecto al eje X (suponiendo que el eje X es horizontal)
            angle = np.degrees(np.arctan2(cY - frame.shape[0] // 2, cX - frame.shape[1] // 2))
            return angle
    return None


def mover_robot(theta1):
    """
    Mueve el robot a una posición con theta1 variable y las otras juntas en posiciones fijas.
    """
    # Definir la lista de ángulos de la posición objetivo
    joint_target = [theta1, theta2, theta3, theta4, theta5, theta6]

    # Mover el robot a la posición objetivo
    robot.MoveJ(joint_target)


def main():
    # Inicializar la captura de video (cambia el índice según la cámara que uses)
    cap = cv2.VideoCapture(1)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("No se pudo capturar la imagen de la cámara")
                break

            # Detecta el ángulo del color rojo respecto al eje X
            angle = detect_red_angle(frame)

            if angle is not None:
                # Ajustar theta1 al ángulo detectado (limitar al rango permitido para theta1)
                theta1 = np.clip(angle, -180, 180)
                mover_robot(theta1)
                print(f"Ángulo detectado: {angle:.2f}°, moviendo base a {theta1}°")

            # Mostrar la imagen de la cámara (opcional para visualización)
            cv2.imshow('Camera', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
