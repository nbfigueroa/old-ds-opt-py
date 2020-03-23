from nonlinear_ds import *
import numpy as np
import numpy.linalg as LA
from numpy import pi
import matplotlib.pyplot as plt

n_grid = 100
x_lim = [-1, 0.25]
y_lim = [-0.5, 0.25]
dim = 2 
ds_lpv1 = np.zeros((dim, n_grid, n_grid))
pos = np.zeros((dim, n_grid, n_grid))

x_vals = np.linspace(x_lim[0], x_lim[1], n_grid)
y_vals = np.linspace(y_lim[0], y_lim[1], n_grid)

import_dir = "./models/"
import_file = "record_ft_a_v3_ds0.yml"
lpv_ds_func1 = lpv_DS(filename=import_dir+import_file)

# Create figure/environment to draw trajectories on
# fig, ax = plt.subplots()
# plt.subplots_adjust(bottom=0.2)  
for ix in range(n_grid):
    for iy in range(n_grid):
        pos[:,ix,iy] = [x_vals[ix], y_vals[iy]]
        ds_lpv1[:,ix,iy] = lpv_ds_func1.get_ds(pos[:,ix,iy])

# plt.ion()
plt.subplots()
plt.quiver(pos[0,:,:], pos[1,:,:], ds_lpv1[0,:,:], ds_lpv1[1,:,:])
plt.show()
