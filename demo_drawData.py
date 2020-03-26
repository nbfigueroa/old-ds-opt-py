from __future__ import print_function
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.widgets import Button
from ds_tools.mousetrajectory_gui import MouseTrajectory


rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

'''
 Brings up an empty world environment to draw "human-demonstrated" trajectories
'''

if __name__ == '__main__':

    # Create figure/environment to draw trajectories on
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    points,   = ax.plot([], [], 'ro', markersize=2, lw=2)
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('$x_1$',fontsize=15)
    plt.ylabel('$x_2$',fontsize=15)
    plt.title('Draw trajectories to learn a motion policy:',fontsize=15)
          

    # Add UI buttons for data/figure manipulation
    store_btn  = plt.axes([0.67, 0.05, 0.075, 0.05])
    clear_btn  = plt.axes([0.78, 0.05, 0.075, 0.05])
    snap_btn   = plt.axes([0.15, 0.05, 0.075, 0.05])    
    bstore     = Button(store_btn, 'store')    
    bclear     = Button(clear_btn, 'clear')    
    bsnap      = Button(snap_btn, 'snap')


    # Calling class to draw data on top of environment
    indexing  = 1 # Set to 1 if you the snaps/data to be indexed with current time-stamp
    store_mat = 0 # Set to 1 if you want to store data in .mat structure for MATLAB
    draw_data = MouseTrajectory(points, indexing, store_mat)
    draw_data.connect()
    bstore.on_clicked(draw_data.store_callback)
    bclear.on_clicked(draw_data.clear_callback)
    bsnap.on_clicked(draw_data.snap_callback)

    # Show
    plt.show()