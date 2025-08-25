import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import math
from math import pi
np.set_printoptions(linewidth=120, formatter={'float': lambda x: f"{0:8.4g}" if abs(x) < 1e-10 else f"{x:8.4g}"})

np.random.seed(0)

from machinevisiontoolbox.base import *
from machinevisiontoolbox import *
from spatialmath.base import *
from spatialmath import *

camera = CentralCamera.Default(pose=SE3.Trans(1, 1, -2));

P = mkgrid(2, 0.5)

p = camera.project_point(P, objpose=SE3.Tz(1))

Te_C_G = camera.estpose(P, p, frame="camera");
Te_C_G.printline()

T_Cd_G = SE3.Tz(1);

T_delta = Te_C_G * T_Cd_G.inv();
T_delta.printline()

camera.pose = camera.pose * T_delta.interp1(0.05);

camera = CentralCamera.Default(pose=SE3.Trans(1, 1, -2));

T_Cd_G = SE3.Tz(1);

pbvs = PBVS(camera, P=P, pose_g=SE3.Trans(-1, -1, 2), pose_d=T_Cd_G, plotvol=[-1, 2, -1, 2, -3, 2.5])

pbvs.run(200);

# Plot the required trajectories
plt.figure()
pbvs.plot_p()      # Image plane trajectory
plt.title("Image Plane Trajectory")

plt.figure()
pbvs.plot_vel()    # Camera velocity
plt.title("Camera Velocity")

plt.figure()
pbvs.plot_pose()   # Camera pose trajectory
plt.title("Camera Pose Trajectory")

plt.show()