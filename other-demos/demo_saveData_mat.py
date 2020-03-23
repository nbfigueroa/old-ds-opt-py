from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import mousetrajectory_gui as mt
import scipy.io as sio

rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

'''
 Brings up an empty world environment with the drawn trajectories using MouseTrajectory GUI
'''

if __name__ == '__main__':

    # Create figure/environment to draw trajectories on
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)    
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('$x_1$',fontsize=15)
    plt.ylabel('$x_2$',fontsize=15)
    plt.title('Draw trajectories to learn a motion policy:',fontsize=15)
          
    # Load trajectories from file and plot
    dir_name  = '../data/'    
    data_name = 'human_demonstrated_trajectories'
    
    file_name = dir_name + data_name + '.dat'
    l,t,x,y   = mt.load_trajectories(file_name)
    ax.plot(x, y, 'ro', markersize=2, lw=2)

    # Show
    plt.show()

    # Create a dictionary
    adict = {}
    adict['labels']      = l
    adict['timestamps'] = t
    adict['x']    = x
    adict['y']    = y
    file_name = dir_name + data_name + '.mat'
    sio.savemat(file_name, adict)