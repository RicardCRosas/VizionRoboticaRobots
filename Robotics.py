# -*- coding: utf-8 -*-
"""
Robotics.py — Solo PyPlot: DH PUMA560, FK SCARA, IK 2R (L1=1, L2=0.8)
+ Visualización y animación SCARA
Autor: Ricardo Carballido Rosas (ID 174926)
"""

import math
import time
import pandas as pd

# Forzar backend de ventana nativa (requiere PyQt5)
import matplotlib
matplotlib.use("Qt5Agg")

import roboticstoolbox as rtb
from roboticstoolbox import DHRobot, RevoluteDH, PrismaticDH


# ---------------- Utilidad de impresión ----------------
def print_df(title: str, df: pd.DataFrame):
    print("\n=== " + title + " ===")
    print(df.to_string(index=False))


# =================== Parte 1: DH PUMA560 ===================
def parte1_puma_dh():
    puma = rtb.models.DH.Puma560()
    rows = []
    for i, L in enumerate(puma.links, start=1):
        is_rev = getattr(L, "isrevolute", True)  # PUMA es rotacional
        theta_field = f"θ{i}" if is_rev else float(L.theta)
        rows.append({
            "i": i,
            "d_i (m)": float(L.d),
            "theta_i": theta_field,
            "a_i (m)": float(L.a),
            "alpha_i (rad)": float(L.alpha)
        })
    df = pd.DataFrame(rows)
    print_df("DH PUMA560", df)
    return puma


# ============ Parte 2: FK SCARA y modelo con qlim ============
def scara_fk(L1, L2, theta1_deg, theta2_deg, d3):
    """SCARA: x,y en plano; z = -d3 (convención: positivo hacia abajo)."""
    t1 = math.radians(theta1_deg)
    t2 = math.radians(theta2_deg)
    x = L1*math.cos(t1) + L2*math.cos(t1 + t2)
    y = L1*math.sin(t1) + L2*math.sin(t1 + t2)
    z = -d3
    return x, y, z


def parte2_fk_scara():
    cases = [
        {"Ejercicio": 1, "L1": 0.4, "L2": 0.3,  "θ1 (deg)": 30,  "θ2 (deg)": -45, "d3 (m)": 0.10},
        {"Ejercicio": 2, "L1": 0.5, "L2": 0.3,  "θ1 (deg)": 40,  "θ2 (deg)": 25,  "d3 (m)": 0.12},
        {"Ejercicio": 3, "L1": 0.5, "L2": 0.25, "θ1 (deg)": -30, "θ2 (deg)": 110, "d3 (m)": 0.06},
    ]
    rows = []
    for c in cases:
        x, y, z = scara_fk(c["L1"], c["L2"], c["θ1 (deg)"], c["θ2 (deg)"], c["d3 (m)"])
        rows.append({**c, "x (m)": round(x, 6), "y (m)": round(y, 6), "z (m)": round(z, 6)})
    print_df("Parte 2 - Resultados FK SCARA (x,y,z)", pd.DataFrame(rows))


def model_scara(L1=0.4, L2=0.3, d3_min=0.0, d3_max=0.20):
    """SCARA 2R+1P con límites (qlim).
       Rotacionales: ±180° en rad; Prismática d3: [d3_min, d3_max] m.
    """
    return DHRobot([
        RevoluteDH(a=L1, alpha=0, qlim=[math.radians(-180), math.radians(180)]),
        RevoluteDH(a=L2, alpha=0, qlim=[math.radians(-180), math.radians(180)]),
        PrismaticDH(theta=0, alpha=0, qlim=[d3_min, d3_max])
    ], name="SCARA")


# ======= Parte 3: IK 2R (L1=1.0, L2=0.8) y modelo con qlim =======
def ik_2r(x, y, L1, L2, elbow="up"):
    """Cinemática inversa planar 2R. Retorna (θ1, θ2) en grados."""
    r2 = x*x + y*y
    c2 = (r2 - L1*L1 - L2*L2) / (2*L1*L2)
    c2 = max(-1.0, min(1.0, c2))
    s2 = math.sqrt(max(0.0, 1 - c2*c2))
    if elbow == "down":
        s2 = -s2
    t2 = math.atan2(s2, c2)
    t1 = math.atan2(y, x) - math.atan2(L2*math.sin(t2), L1 + L2*math.cos(t2))
    return math.degrees(t1), math.degrees(t2)


def parte3_ik_2r():
    # AJUSTE según tu diapositiva: L1=1.0, L2=0.8
    L1 = 1.0
    L2 = 0.8

    points = [
        {"Posición": 1, "x": 1.0,  "y": 1.0},
        {"Posición": 2, "x": 1.6,  "y": 0.2},
        {"Posición": 3, "x": -0.5, "y": 1.2},
    ]
    rows = []
    for p in points:
        for elbow in ("up", "down"):
            th1, th2 = ik_2r(p["x"], p["y"], L1, L2, elbow)
            rows.append({**p, "Solución": elbow,
                         "θ1 (deg)": round(th1, 3),
                         "θ2 (deg)": round(th2, 3)})
    print_df("Parte 3 - IK 2R (L1=1.0, L2=0.8)", pd.DataFrame(rows))


def model_planar_2r(L1=1.0, L2=0.8):  # default L2=0.8 para coincidir con la lámina
    return DHRobot([
        RevoluteDH(a=L1, alpha=0, qlim=[math.radians(-180), math.radians(180)]),
        RevoluteDH(a=L2, alpha=0, qlim=[math.radians(-180), math.radians(180)])
    ], name="Planar 2R")


# ==================== Visualización (solo PyPlot) ====================
def show_with_pyplot():
    # PUMA
    puma = rtb.models.DH.Puma560()
    print("Abriendo PUMA560 (PyPlot)... cierra la ventana para continuar.")
    puma.plot([0, 0, 0, 0, 0, 0], backend='pyplot', block=True)

    # SCARA (caso 1)
    scara = model_scara(L1=0.4, L2=0.3, d3_min=0.0, d3_max=0.20)
    print("Abriendo SCARA (PyPlot)...")
    scara.plot([math.radians(30), math.radians(-45), 0.10], backend='pyplot', block=True)

    # Planar 2R (L1=1.0, L2=0.8)
    planar = model_planar_2r(1.0, 0.8)
    print("Abriendo Planar 2R (PyPlot, L1=1.0, L2=0.8)...")
    planar.plot([math.radians(45), math.radians(30)], backend='pyplot', block=True)


# ============== Animación de movimientos del SCARA (PyPlot) ==============
def animar_scara_pyplot():
    """Anima el SCARA en PyPlot recorriendo varias poses (trayectorias en q)."""
    from roboticstoolbox.tools.trajectory import jtraj
    from roboticstoolbox.backends.PyPlot import PyPlot

    # Modelo SCARA y límites
    L1, L2 = 0.4, 0.3
    scara = model_scara(L1=L1, L2=L2, d3_min=0.0, d3_max=0.20)

    # Backend PyPlot
    env = PyPlot()
    env.launch()
    env.add(scara)

    # Waypoints (articulares): [θ1(rad), θ2(rad), d3(m)]
    q_home = [0.0, 0.0, 0.10]
    q_e1   = [math.radians(30),  math.radians(-45), 0.10]
    q_e2   = [math.radians(40),  math.radians(25),  0.12]
    q_e3   = [math.radians(-30), math.radians(110), 0.06]
    q_down = [q_e3[0], q_e3[1],  0.18]
    q_up   = [q_e3[0], q_e3[1],  0.06]

    tramos = [
        (q_home, q_e1,  60),
        (q_e1,   q_e2,  60),
        (q_e2,   q_e3,  60),
        (q_e3,   q_down,40),
        (q_down, q_up,  40),
        (q_up,   q_home,60),
    ]

    dt = 0.02  # ~50 FPS
    for q0, qf, n in tramos:
        traj = jtraj(q0, qf, n)  # interpola en espacio articular
        for q in traj.q:
            scara.q = q
            env.step(dt)
            time.sleep(dt)

    print(" Animación SCARA terminada. Cierra la ventana para continuar.")


# ====================== MAIN ======================
if __name__ == "__main__":
    # Cálculo/impresión para el reporte
    parte1_puma_dh()
    parte2_fk_scara()
    parte3_ik_2r()

    # Visualizaciones estáticas (PyPlot)
    show_with_pyplot()

    # Animación del SCARA
    animar_scara_pyplot()

    print("\nListo. Si no ves ventanas, verifica PyQt5 y la config de PyCharm (plots externos).")
