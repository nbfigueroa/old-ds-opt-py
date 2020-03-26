from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import mousetrajectory_gui as mt

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
    plt.title('Trajectories drawn by human with mouse GUI:',fontsize=15)


    # Add Obstacles and target in environment
    x_target = np.array([0.9, 0.8])
    ax.plot(x_target[0], x_target[1], 'rd', markersize=10, lw=2)
    ax.add_artist(plt.Circle((0.2, 0.8), 0.15))
    ax.add_artist(plt.Rectangle((0.50, 0.25), 0.4, 0.1))
    ax.add_artist(plt.Rectangle((0.65, 0.1), 0.1, 0.4))
    ax.add_artist(plt.Rectangle((0.5, 0.55), 0.2, 0.3))

          
    # Load trajectories from file and plot
    file_name = './data/human_demonstrated_trajectories_Mar22_23:06:56.dat'
    l,t,x,y   = mt.load_trajectories(file_name)
    ax.plot(x, y, 'ro', markersize=2, lw=2)

    # Show
    plt.show()